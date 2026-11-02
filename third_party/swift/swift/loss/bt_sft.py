# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-CG BT-SFT (Bradley-Terry verbalized scoring) for the TRANSFORMERS backend.

Port of the Megatron-backend objective (swift/megatron/model/qwen3_vl_aip.py +
bt_score.py) to a plain HuggingFace causal-LM training path. The model learns to emit,
per content-guideline (CG), ``<cg_c><score_k>`` in one assistant turn; the continuous
score for CG c is read off the LM head at the ``<cg_c>`` position (its next-token IS the
score) as the softmax-expected bucket, normalized to ``[0, 1]``, and ranked pairwise with
a Bradley-Terry loss against the continuous GT (``regression_labels``). Total loss =
lm_ce + bt_loss_weight * bt_loss.

Backend-agnostic core here (readout + BT loss + queue); the swift ``BaseLoss`` wrapper in
mapping registration routes ``input_ids``/``cg_ids``/``regression_labels`` into it.
"""
import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def expected_score_from_logits(score_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """E[score] = Σ_k k·softmax(score_logits/T)_k over the last dim (score-token logits,
    ascending bucket order). Returns a differentiable value in ``[0, N-1]``. Matches
    ``swift.megatron.model.bt_score.expected_score_from_logits`` exactly."""
    probs = torch.softmax(score_logits.float() / temperature, dim=-1)
    k = torch.arange(score_logits.shape[-1], device=score_logits.device, dtype=probs.dtype)
    return (probs * k).sum(dim=-1)


def read_verbalized_scores(logits: torch.Tensor, input_ids: torch.Tensor, cg_marker_ids: List[int],
                           score_token_ids: List[int], temperature: float = 1.0, logit_offset: int = 0) -> torch.Tensor:
    """Read the per-CG normalized score off the LM logits.

    input_ids: ``[B, S]`` (full). logits: ``[B, S, V]`` (full, logit_offset=0) OR ``[B, K, V]``
    when the trainer truncated to the response region via ``logits_to_keep`` (then
    ``logit_offset = S - K`` and logits[j] == full-logits[j + offset]). A ``<cg_c>`` marker at
    full position ``p`` is read at logit index ``p - offset`` (markers are in the response, so
    they are always inside the kept window). This is the memory win: gemma4's ``[B,S,262k]``
    logits collapse to ``[B, ~30, 262k]`` -- the transformers analogue of the Megatron sparse head.

    Reading logit COLUMNS at the marker position is identical to Megatron's
    ``hidden[pos] @ score_rows.T`` because ``logits = hidden @ out_emb.T`` (softcap off).
    """
    B, S = input_ids.shape
    K = logits.shape[1]
    C = len(cg_marker_ids)
    N = len(score_token_ids)
    ar = torch.arange(S, device=logits.device)
    b_idx = torch.arange(B, device=logits.device)
    score_ids = torch.as_tensor(score_token_ids, device=logits.device, dtype=torch.long)
    out = logits.new_zeros(B, C)
    for c in range(C):
        mask = input_ids == cg_marker_ids[c]                       # [B, S] over FULL positions
        pos = torch.where(mask, ar.unsqueeze(0), torch.full_like(input_ids, -1)).max(dim=1).values
        jpos = (pos - logit_offset).clamp(0, K - 1)                # into the (maybe truncated) logits
        sl = logits[b_idx, jpos][:, score_ids]                     # [B, N]
        out[:, c] = expected_score_from_logits(sl, temperature)
    return out / float(max(N - 1, 1))


def dense_cg_labels(regression_labels: torch.Tensor, cg_ids: Optional[torch.Tensor], num_cg: int,
                    ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Project per-item CG labels to a dense ``[N, num_cg]`` matrix (NaN = unlabeled).
    1-D scores + cg_ids -> one non-NaN entry per row; already-dense input returned as-is.
    Mirrors ``qwen3_vl_aip._dense_cg_labels``."""
    if regression_labels is None:
        return None
    N = ref.size(0)
    rl = regression_labels.float().to(ref.device)
    if rl.dim() == 2:
        return rl
    rl = rl.reshape(-1)
    idx = (cg_ids.to(device=ref.device, dtype=torch.long).reshape(-1)
           if cg_ids is not None else torch.zeros(N, dtype=torch.long, device=ref.device))
    idx = idx[:N].clamp(0, num_cg - 1)
    dense = torch.full((N, num_cg), float('nan'), device=ref.device, dtype=torch.float32)
    if N > 0:
        dense[torch.arange(N, device=ref.device), idx] = rl[:N]
    return dense


def bradley_terry_loss_multilabel(logits2d: torch.Tensor, labels2d: torch.Tensor, margin: float,
                                  queue: Optional[dict] = None) -> torch.Tensor:
    """Item-granular multi-CG Bradley-Terry ranking loss, averaged over CGs with a valid pair.

    logits2d / labels2d: ``[N, num_cg]``; labels2d is NaN where a CG is unlabeled. Vectorized,
    sync-free (matches the vectorized Megatron ``_bradley_terry_loss_multilabel``). ``queue`` is
    an optional dict-backed cross-batch memory bank {'logits','labels','count'} appended in place.
    """
    cur_l, cur_y = logits2d, labels2d
    if queue is not None and queue.get('count', 0) > 0:
        all_l = torch.cat([logits2d, queue['logits'][:queue['count']].to(logits2d.device).detach()], 0)
        all_y = torch.cat([labels2d, queue['labels'][:queue['count']].to(labels2d.device).detach()], 0)
    else:
        all_l, all_y = logits2d, labels2d
    if queue is not None:
        _enqueue(queue, logits2d.detach(), labels2d.detach())

    vmask = ~torch.isnan(all_y)
    labels_f = torch.where(vmask, all_y, all_y.new_zeros(()))
    logit_diff = all_l.unsqueeze(1) - all_l.unsqueeze(0)                    # [M,M,C]
    label_diff = labels_f.unsqueeze(1) - labels_f.unsqueeze(0)
    pair = vmask.unsqueeze(1) & vmask.unsqueeze(0) & (label_diff > margin)
    pair_f = pair.to(logit_diff.dtype)
    n_valid = pair_f.sum(dim=(0, 1))
    loss_per_cg = -(F.logsigmoid(logit_diff) * pair_f).sum(dim=(0, 1)) / n_valid.clamp(min=1)
    has_pairs = (n_valid > 0).to(logit_diff.dtype)
    return (loss_per_cg * has_pairs).sum() / has_pairs.sum().clamp(min=1)


def _enqueue(queue: dict, logits2d: torch.Tensor, labels2d: torch.Tensor) -> None:
    """Append the current batch's per-item [num_cg] logit/label rows into a ring buffer."""
    Q = queue['size']
    if Q <= 0:
        return
    C = logits2d.size(-1)
    if 'logits' not in queue:
        queue['logits'] = torch.zeros(Q, C, device=logits2d.device)
        queue['labels'] = torch.full((Q, C), float('nan'), device=labels2d.device)
        queue['ptr'] = 0
        queue['count'] = 0
    b = logits2d.size(0)
    if b >= Q:
        queue['logits'].copy_(logits2d[-Q:]); queue['labels'].copy_(labels2d[-Q:])
        queue['ptr'] = 0; queue['count'] = Q
        return
    ptr = queue['ptr']; end = ptr + b
    if end <= Q:
        queue['logits'][ptr:end] = logits2d; queue['labels'][ptr:end] = labels2d
    else:
        first = Q - ptr
        queue['logits'][ptr:] = logits2d[:first]; queue['labels'][ptr:] = labels2d[:first]
        queue['logits'][:b - first] = logits2d[first:]; queue['labels'][:b - first] = labels2d[first:]
    queue['ptr'] = end % Q
    queue['count'] = min(queue['count'] + b, Q)


# ---------------------------------------------------------------------------
# swift transformers-backend loss wrapper (registered as loss_type='bt_sft')
# ---------------------------------------------------------------------------
import os

from .base import BaseLoss


class BTSFTLoss(BaseLoss):
    """Multi-CG BT-SFT loss for the transformers backend.

    total = mean_CE(lm, <score_unk> masked) + BT_SFT_LOSS_WEIGHT * bradley_terry.

    num_cg / num_buckets are DERIVED from the registered ``<cg_N>`` / ``<score_N>`` special
    tokens (add them with ``--new_special_tokens``). Weights are env-configured (matching the
    env-var convention of the embedding/reranker losses):
      BT_SFT_LOSS_WEIGHT (0.2)  BT_SFT_MARGIN (0.05)  BT_SFT_TEMPERATURE (1.0)  BT_SFT_QUEUE_SIZE (32)

    Requires full LM logits aligned with input_ids (i.e. use_logits_to_keep=False, which is
    forced for this loss). regression_labels/cg_ids/input_ids are routed in by the trainer's
    compute_loss (see Seq2SeqTrainer.compute_loss).
    """

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.bt_loss_weight = float(os.environ.get('BT_SFT_LOSS_WEIGHT', '0.2'))
        self.margin = float(os.environ.get('BT_SFT_MARGIN', '0.05'))
        self.temperature = float(os.environ.get('BT_SFT_TEMPERATURE', '1.0'))
        self.queue = {'size': int(os.environ.get('BT_SFT_QUEUE_SIZE', '32'))}
        self._ids = None
        self._softcap_checked = False

    def _maybe_disable_gemma_softcap(self):
        """gemma4 verbalized scoring reads PRE-softcap logits (matches Megatron BT-SFT, which
        nulls ``final_logit_softcapping`` for training). Softcap is monotonic so ranking is
        preserved either way, but disabling keeps the read/CE consistent + parity with the
        Megatron-trained checkpoints. Keep it with GEMMA4_TRAIN_SOFTCAP=1.

        Runs lazily on the first loss call (the model is loaded by then, unlike at __init__)."""
        if self._softcap_checked:
            return
        self._softcap_checked = True
        if os.environ.get('GEMMA4_TRAIN_SOFTCAP', '0') == '1':
            return
        model = getattr(self.trainer, 'model', None)
        if model is None:
            self._softcap_checked = False   # retry next call
            return
        # HF gemma4 reads final_logit_softcapping from CONFIG (self.config + its text_config),
        # not a module attr (unlike mcore). Null it on every reachable config object.
        n = 0
        configs = []
        for m in list(model.modules()):
            cfg = getattr(m, 'config', None)
            if cfg is not None:
                configs.append(cfg)
                if hasattr(cfg, 'get_text_config'):
                    try:
                        configs.append(cfg.get_text_config())
                    except Exception:
                        pass
                configs.append(getattr(cfg, 'text_config', None))
        for cfg in configs:
            if cfg is not None and getattr(cfg, 'final_logit_softcapping', None) is not None:
                cfg.final_logit_softcapping = None
                n += 1
        if n:
            import logging
            logging.getLogger(__name__).info(
                f'[bt_sft] disabled final_logit_softcapping on {n} module(s) for training parity '
                f'(set GEMMA4_TRAIN_SOFTCAP=1 to keep it).')

    def _resolve_ids(self):
        if self._ids is not None:
            return self._ids
        tok = self.trainer.template.tokenizer
        unk_tok = getattr(tok, 'unk_token_id', None)

        def series(fmt):
            out = []
            i = 0
            while True:
                tid = tok.convert_tokens_to_ids(fmt.format(i))
                if tid is None or tid == unk_tok:
                    break
                out.append(int(tid))
                i += 1
            return out

        cg = series('<cg_{}>')
        score = series('<score_{}>')
        unk = tok.convert_tokens_to_ids('<score_unk>')
        if len(cg) < 1 or len(score) < 2:
            raise ValueError('bt_sft loss: no <cg_N>/<score_N> tokens registered. '
                             'Pass --new_special_tokens with the CG markers + score buckets.')
        self._ids = (cg, score, (None if unk == unk_tok else int(unk)))
        return self._ids

    @staticmethod
    def _log_bt_metrics(trainer, lm_loss, bt_loss, reg_logits):
        """Surface bt_loss/lm_loss (+ mean predicted score) into the HF Trainer logs, via the same
        custom_metrics channel the trainer uses for MoE aux_loss. Keys auto-create + DDP-sync in log()."""
        if trainer is None or not hasattr(trainer, 'custom_metrics'):
            return
        try:
            mode = 'train' if trainer.model.training else 'eval'
            m = trainer.custom_metrics[mode]
            m['bt_loss'].update(bt_loss.detach())
            m['lm_loss'].update(lm_loss.detach())
            m['reg_pred_mean'].update(reg_logits.detach().mean())
        except Exception:  # logging must never break training
            pass

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, trainer=None,
                 regression_labels=None, cg_ids=None, input_ids=None, **kwargs):
        from swift.trainers import per_token_loss_func
        self._maybe_disable_gemma_softcap()
        cg_ids_tok, score_ids, unk_id = self._resolve_ids()

        # --- LM cross-entropy (mask <score_unk> placeholders, like the Megatron forward) ---
        lm_labels = labels if unk_id is None else torch.where(labels == unk_id,
                                                              torch.full_like(labels, -100), labels)
        token_loss = per_token_loss_func(outputs, lm_labels)
        denom = num_items_in_batch if num_items_in_batch is not None else (lm_labels[:, 1:] != -100).sum()
        lm_loss = token_loss.sum() / torch.clamp(denom if torch.is_tensor(denom)
                                                 else torch.tensor(float(denom), device=token_loss.device), min=1)

        # --- Bradley-Terry ranking on the verbalized per-CG scores ---
        if regression_labels is None or input_ids is None:
            return lm_loss
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        # logits may be response-region-truncated (use_logits_to_keep) -> map marker positions.
        # Requires the truncation to be a contiguous suffix (holds for batch>1; BT-SFT uses it).
        logit_offset = input_ids.shape[1] - logits.shape[1]
        reg_logits = read_verbalized_scores(logits, input_ids, cg_ids_tok, score_ids, self.temperature,
                                            logit_offset=logit_offset)
        rl2d = dense_cg_labels(regression_labels, cg_ids, len(cg_ids_tok), reg_logits)
        if rl2d is None or not torch.isfinite(rl2d).any():
            self._log_bt_metrics(trainer, lm_loss, reg_logits.new_zeros(()), reg_logits)
            return lm_loss + reg_logits.sum() * 0.0     # keep head in graph, zero contribution
        bt_loss = bradley_terry_loss_multilabel(reg_logits, rl2d, self.margin, queue=self.queue)
        # Report the lm/bt loss components to the trainer logs (mirrors Megatron's reg_bt_loss).
        self._log_bt_metrics(trainer, lm_loss, bt_loss, reg_logits)
        if os.environ.get('BT_SFT_DEBUG') == '1':
            n_lab = int(torch.isfinite(rl2d).sum())
            print(f'[bt_sft] lm={float(lm_loss):.4f} bt={float(bt_loss):.4f} '
                  f'bt_w={self.bt_loss_weight} n_labeled={n_lab} queue={self.queue.get("count", 0)} '
                  f'reg_pred_mean={float(reg_logits.mean()):.4f}', flush=True)
        # grad-accum: bt is a per-batch ranking term (not per-token); scale like the trainer would
        if num_items_in_batch is not None:
            bt_loss = bt_loss / max(getattr(self.args, 'gradient_accumulation_steps', 1), 1)
        return lm_loss + self.bt_loss_weight * bt_loss
