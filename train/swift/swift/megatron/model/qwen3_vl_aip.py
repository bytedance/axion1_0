# Copyright (c) ModelScope Contributors. All rights reserved.
"""Qwen3-VL AIP regression head extension for mcore_bridge 0.17.1."""
import logging
import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core import parallel_state

from mcore_bridge.model.mm_gpt_model import MultimodalGPTModel
from mcore_bridge.model.mm_gpts.qwen3_vl import Qwen3VLLoader
from swift.utils import get_logger

logger = get_logger()

def _last_assistant_nl_idx(labels_1d: torch.Tensor, cu_seqlens: torch.Tensor,
                            num_samples: int) -> torch.Tensor:
    """Return the index of \\n after 'assistant' for each packed sample.

    The labels tensor is unshifted: ``labels[i] = token_id`` for response
    positions, ``-100`` for prompt positions.  The ``\\n`` token that immediately
    follows ``<|im_start|>assistant`` in the chat template has label ``-100``
    (it is part of the prompt prefix, not the trainable response).  It therefore
    sits at the last ``-100`` position before the first labeled token of the last
    assistant turn, i.e. ``first_labeled_in_last_turn - 1``.

    Args:
        labels_1d: ``[total_tokens]`` flat label tensor.
        cu_seqlens: ``[num_samples + 1]`` cumulative sequence lengths.
        num_samples: number of real samples (may be less than ``len(cu_seqlens) - 1``
            due to SP-alignment padding).

    Returns:
        ``[num_samples]`` int64 tensor of global token positions.
    """
    indices = []
    for i in range(num_samples):
        start = int(cu_seqlens[i])
        end = int(cu_seqlens[i + 1])
        seq_labels = labels_1d[start:end]
        is_labeled = seq_labels != -100
        # transitions[j] is True at the start of each labeled run
        prev = torch.cat([is_labeled.new_zeros(1), is_labeled[:-1]])
        transitions = is_labeled & ~prev
        t_pos = transitions.nonzero(as_tuple=False).view(-1)
        if t_pos.numel() == 0:
            # No labeled tokens — fall back to last position in the sequence.
            indices.append(max(start, end - 1))
        else:
            last_start_local = int(t_pos[-1])
            # One position before the first labeled token of the last turn.
            nl_local = max(0, last_start_local - 1)
            indices.append(start + nl_local)
    return torch.tensor(indices, device=labels_1d.device, dtype=torch.long)


def _last_assistant_nl_idx_batched(labels: torch.Tensor,
                                    device: torch.device) -> torch.Tensor:
    """Non-packing variant: ``labels`` is ``[b, s]``, returns ``[b]`` positions."""
    b, s = labels.shape
    indices = []
    for bi in range(b):
        seq_labels = labels[bi]
        is_labeled = seq_labels != -100
        prev = torch.cat([is_labeled.new_zeros(1), is_labeled[:-1]])
        transitions = is_labeled & ~prev
        t_pos = transitions.nonzero(as_tuple=False).view(-1)
        if t_pos.numel() == 0:
            indices.append(s - 1)
        else:
            last_start = int(t_pos[-1])
            indices.append(max(0, last_start - 1))
    return torch.tensor(indices, device=device, dtype=torch.long)


class Qwen3VLAIPModel(MultimodalGPTModel):
    """Qwen3VLGPTModel with an additional MLP regression head for AIP tasks.

    The regression head ``nn.Sequential(Dropout(0.1), Linear(hidden_size, regression_dim))``
    is applied to the decoder's final hidden states on the last pipeline stage
    (post_process=True).  Forward returns a ``(lm_output, reg_logits)`` tuple on
    that stage so the training loop can compute both language-model loss and a
    regression loss.  On all other pipeline stages the return value is identical
    to ``Qwen3VLGPTModel``.

    ``regression_dim`` is read from ``get_args().regression_dim`` (default 1).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_args = getattr(self.config, 'aip_train_args', None)
        if train_args is None:
            from types import SimpleNamespace
            train_args = SimpleNamespace(
                regression_dim=1,
                regression_hidden_dim=512,
                regression_loss_type='mse',
                bt_queue_size=0,
                decision_thresholds='',
                decision_threshold=0.5,
                l2_lambda=0.1,
                focal_beta=0.0,
                focal_mode='boundary',
                continuous_pos_weight=1.0,
                threshold_soft_bce_scale=0.0,
                threshold_soft_bce_tau=0.03,
                bt_sft=False,
                bt_sft_temperature=1.0,
                bt_sft_score_norm=True,
                bt_sft_per_cg_score_tokens=False,
                bt_loss_weight=1.0,
                huber_loss_weight=0.0,
            )
        regression_dim = getattr(train_args, 'regression_dim', 1)
        # Store AIP-specific args on the model so compute_regression_loss can
        # read them reliably even if get_args() is called in a different context.
        self._regression_loss_type: str = getattr(train_args, 'regression_loss_type', 'mse')
        self._bt_queue_size: int = getattr(train_args, 'bt_queue_size', 0)
        # BT-SFT: verbalized scoring off the LM head instead of a regression head.
        self._bt_sft: bool = getattr(train_args, 'bt_sft', False)
        self._bt_sft_temperature: float = getattr(train_args, 'bt_sft_temperature', 1.0)
        self._num_score_buckets: int = getattr(train_args, 'num_score_buckets', 11)
        # The verbalized expected score lands in [0, N-1]; when bt_sft_score_norm (default) we
        # normalize it to [0, 1] by dividing by (N-1) before the BT sigmoid, so the ranking loss is
        # invariant to N. Set bt_sft_score_norm=False to feed the raw [0, N-1] score (denom=1) —
        # the pre-normalization baseline (A/B toggle). No logit temperature either way.
        _bt_sft_score_norm = getattr(train_args, 'bt_sft_score_norm', True)
        self._bt_score_denom: float = float(max(self._num_score_buckets - 1, 1)) if _bt_sft_score_norm else 1.0
        # Independent BT-SFT loss weights (these REPLACE reg_loss_weight for bt_sft): total reg loss
        # is bt_loss_weight * bt_ranking + huber_loss_weight * huber(proxy, gt_norm_score). The Huber
        # term (weight 0 = off) is a pointwise regression of the normalized score proxy against the
        # (normalized) GT score; it assumes regression_labels are on the same [0, 1] scale as the
        # proxy (true for ts_score_norm data).
        self._bt_loss_weight: float = float(getattr(train_args, 'bt_loss_weight', 1.0))
        self._huber_loss_weight: float = float(getattr(train_args, 'huber_loss_weight', 0.0))
        # Huber target column: 'regression_labels' (default; raw normalized GT score) or
        # 'score_percentile' (mid-rank empirical percentile, a uniform [0,1] target matched to the
        # proxy). Only the Huber term switches; the BT ranking loss always uses regression_labels.
        self._huber_target: str = str(getattr(train_args, 'huber_target', 'regression_labels'))
        self._huber_use_percentile: bool = (self._huber_target == 'score_percentile')
        self._score_token_ids: Optional[torch.Tensor] = None  # lazily resolved from the tokenizer
        # Multi-CG BT-SFT: emit <cg_c><score_k> per category in one generation. When
        # _num_cg > 1, _bt_sft_score reads a score at each <cg_c> marker -> [N, num_cg]
        # and compute_regression_loss selects the labeled column by cg_ids.
        self._num_cg: int = getattr(train_args, 'bt_sft_num_cg', 1)
        self._cg_marker_ids: Optional[torch.Tensor] = None  # [num_cg], set by the trainer
        self._score_unk_id: Optional[int] = None  # placeholder token id, set by the trainer
        # Per-CG score tokens: each CG reads its own <cg_c_score_k> rows instead of the shared
        # <score_k>. When on, _per_cg_score_token_ids is a [num_cg, num_buckets] id matrix (trainer-set).
        self._per_cg_score_tokens: bool = bool(getattr(train_args, 'bt_sft_per_cg_score_tokens', False))
        self._per_cg_score_token_ids: Optional[torch.Tensor] = None  # [num_cg, num_buckets], set by the trainer
        logger.info(
            f'[Qwen3VLAIPModel.__init__] bt_sft={self._bt_sft}  '
            f'regression_loss_type={self._regression_loss_type!r}  '
            f'bt_queue_size={self._bt_queue_size}  regression_dim={regression_dim}  '
            f'bt_score_denom={self._bt_score_denom} (norm={_bt_sft_score_norm})  '
            f'bt_loss_weight={self._bt_loss_weight}  huber_loss_weight={self._huber_loss_weight}  '
            f'huber_target={self._huber_target!r}  '
            f'per_cg_score_tokens={self._per_cg_score_tokens}'
        )
        self._regression_hidden: Optional[torch.Tensor] = None
        if self.post_process:
            if self._bt_sft:
                # No regression head: the score is read off the existing LM head at
                # the <score_k> position. Nothing to build here; score token ids are
                # resolved lazily in forward() from the megatron tokenizer.
                logger.info('[Qwen3VLAIPModel] BT-SFT enabled: verbalized scoring on the LM head '
                            '(no regression head); expects --regression_loss_type bradley_terry.')
            else:
                hidden_size = self.config.hidden_size
                hidden_dim = getattr(train_args, 'regression_hidden_dim', 512)
                self.regression_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Linear(hidden_size, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_dim, regression_dim),
                )
            # Capture hidden states after the final layer-norm via a hook (used by
            # both the regression head and BT-SFT verbalized scoring).
            # The hook fires on the last pipeline stage only (where final_layernorm
            # is not None), so no guard is needed in forward().
            decoder = self.language_model.decoder
            if decoder.final_layernorm is not None:
                decoder.final_layernorm.register_forward_hook(self._capture_hidden)

            # Memory bank (FIFO ring buffer) for BT ranking loss augmentation.
            bt_queue_size = getattr(train_args, 'bt_queue_size', 0)
            if bt_queue_size > 0:
                self.register_buffer('_logit_queue', torch.zeros(bt_queue_size), persistent=False)
                self.register_buffer('_label_queue', torch.zeros(bt_queue_size), persistent=False)
                self.register_buffer('_task_queue', torch.zeros(bt_queue_size, dtype=torch.long), persistent=False)
                # Plain Python ints — no GPU sync needed on reads (persistent=False anyway).
                self._py_queue_ptr: int = 0
                self._py_queue_count: int = 0

            # Per-task decision thresholds for threshold-aware soft BCE and
            # focal/confidence weighting.  Parsed from --decision-thresholds
            # (comma-separated floats) or a single --decision-threshold float.
            raw = getattr(train_args, 'decision_thresholds', None)
            if raw:
                if isinstance(raw, str):
                    t_vals = [float(x.strip()) for x in raw.split(',') if x.strip()]
                else:
                    t_vals = [float(x) for x in raw]
            else:
                t_vals = [float(getattr(train_args, 'decision_threshold', 0.5))]
            self.register_buffer(
                'decision_thresholds',
                torch.tensor(t_vals, dtype=torch.float32),
                persistent=False,
            )
            _thresh_mode = 'broadcast' if len(t_vals) == 1 else 'per_task'
            logger.info(
                f'[Qwen3VLAIPModel] decision_thresholds registered: {t_vals}  '
                f'mode={_thresh_mode}  '
                f'(raw --decision_thresholds={raw!r}, '
                f'--decision_threshold={getattr(train_args, "decision_threshold", 0.5)!r})'
            )
            logger.info(
                f'[Qwen3VLAIPModel] combined_bt_mse config: '
                f'threshold_mode={_thresh_mode}  threshold_values={t_vals}  '
                f'l2_lambda={getattr(train_args, "l2_lambda", 0.1)}  '
                f'focal_beta={getattr(train_args, "focal_beta", 0.0)}  '
                f'focal_mode={getattr(train_args, "focal_mode", "boundary")!r}  '
                f'continuous_pos_weight={getattr(train_args, "continuous_pos_weight", 1.0)}  '
                f'threshold_soft_bce_scale={getattr(train_args, "threshold_soft_bce_scale", 0.0)}  '
                f'threshold_soft_bce_tau={getattr(train_args, "threshold_soft_bce_tau", 0.03)}'
            )

    def _capture_hidden(self, module, input, output):
        self._regression_hidden = output

    @torch.no_grad()
    def _enqueue(self, logits: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor):
        """FIFO ring-buffer enqueue of detached samples for BT memory bank."""
        if not hasattr(self, '_logit_queue'):
            return
        batch_size = logits.shape[0]
        if batch_size == 0:
            return
        queue_size = self._logit_queue.shape[0]
        # Lazy device migration for data buffers only; ptr/count are Python ints.
        for attr in ('_logit_queue', '_label_queue', '_task_queue'):
            buf = getattr(self, attr)
            if buf.device != logits.device:
                setattr(self, attr, buf.to(logits.device))
        if batch_size >= queue_size:
            self._logit_queue.copy_(logits[-queue_size:])
            self._label_queue.copy_(labels[-queue_size:])
            self._task_queue.copy_(task_ids[-queue_size:])
            self._py_queue_ptr = 0
            self._py_queue_count = queue_size
            return
        ptr = self._py_queue_ptr
        end = ptr + batch_size
        if end <= queue_size:
            self._logit_queue[ptr:end] = logits
            self._label_queue[ptr:end] = labels
            self._task_queue[ptr:end] = task_ids
        else:
            first_len = queue_size - ptr
            self._logit_queue[ptr:] = logits[:first_len]
            self._label_queue[ptr:] = labels[:first_len]
            self._task_queue[ptr:] = task_ids[:first_len]
            second_len = batch_size - first_len
            self._logit_queue[:second_len] = logits[first_len:]
            self._label_queue[:second_len] = labels[first_len:]
            self._task_queue[:second_len] = task_ids[first_len:]
        self._py_queue_ptr = end % queue_size
        self._py_queue_count = min(self._py_queue_count + batch_size, queue_size)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'[bt_queue] enqueued {batch_size} samples  '
                f'ptr={self._py_queue_ptr}  '
                f'fill={self._py_queue_count}/{queue_size}'
            )

    def _resolve_score_token_ids(self, device) -> torch.Tensor:
        """Lazily resolve and cache the 11 <score_k> vocab ids as a [11] long tensor."""
        if self._score_token_ids is None:
            from megatron.training import get_tokenizer
            from .bt_score import get_score_token_ids
            tok = get_tokenizer()
            hf = getattr(tok, 'tokenizer', tok)  # megatron tokenizer wraps the HF tokenizer
            ids = get_score_token_ids(hf, self._num_score_buckets)
            self._score_token_ids = torch.tensor(ids, dtype=torch.long, device=device)
            logger.info(f'[Qwen3VLAIPModel] BT-SFT score_token_ids({self._num_score_buckets})={ids}')
        elif self._score_token_ids.device != device:
            self._score_token_ids = self._score_token_ids.to(device)
        return self._score_token_ids

    def _score_rows(self, device, score_ids=None) -> torch.Tensor:
        """TP-safe ``[len(score_ids), h]`` LM-head rows for the given score-token ids
        (defaults to the shared ``<score_k>`` ids).

        The LM head / embedding weight is vocab-parallel: each TP rank holds a
        contiguous ``[padded_vocab_size // tp, h]`` shard, while the score ids are
        GLOBAL vocab ids. Map them to local rows, zero the rows this rank does not
        own, then autograd-aware all-reduce so every rank gets the full rows (and
        gradient reaches the owning rank's embedding rows).
        """
        if score_ids is None:
            score_ids = self._resolve_score_token_ids(device)  # [num_buckets]
        lm = self.language_model
        if lm.share_embeddings_and_output_weights:
            out_w = lm.shared_embedding_or_output_weight()
        else:
            out_w = lm.output_layer.weight
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            from torch.distributed.nn import all_reduce as _tp_all_reduce
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            v_local = out_w.shape[0]
            vocab_start = tp_rank * v_local
            owned = (score_ids >= vocab_start) & (score_ids < vocab_start + v_local)
            local_ids = (score_ids - vocab_start).clamp(0, v_local - 1)
            rows = out_w.index_select(0, local_ids).float()
            rows = rows * owned.unsqueeze(1).to(rows.dtype)
            rows = _tp_all_reduce(rows, group=parallel_state.get_tensor_model_parallel_group())
        else:
            rows = out_w.index_select(0, score_ids).float()
        return rows  # [len(score_ids), h]

    def _score_rows_per_cg(self, device) -> torch.Tensor:
        """``[num_cg, num_buckets, h]`` LM-head rows -- each CG's own ``<cg_c_score_k>`` tokens.
        Flattens the [num_cg, num_buckets] id matrix through the TP-safe ``_score_rows`` path."""
        if self._per_cg_score_token_ids is None:
            raise ValueError('per-CG BT-SFT requires _per_cg_score_token_ids (set by the trainer).')
        ids = self._per_cg_score_token_ids.to(device)  # [num_cg, num_buckets]
        num_cg, num_buckets = ids.shape
        rows = self._score_rows(device, ids.reshape(-1))  # [num_cg*num_buckets, h]
        return rows.view(num_cg, num_buckets, -1)  # [num_cg, num_buckets, h]

    def _bt_sft_score(self, hidden, labels, packed_seq_params, input_ids=None) -> torch.Tensor:
        """Expected verbalized score(s) read off the LM head, normalized to ``[0, 1]``.

        The softmax-expected value is in ``[0, N-1]``; we divide by ``N-1`` so the
        Bradley-Terry sigmoid sees a normalized-score difference in ``[-1, 1]``,
        making the ranking loss invariant to ``num_score_buckets`` (no logit
        temperature). ``bt_margin`` stays on the (continuous) label scale.

        Single-CG (num_cg == 1): read at the ``<score_k>`` position -> ``[N, 1]``.
        Multi-CG (num_cg > 1): read at each ``<cg_c>`` marker (whose next-token IS
        that category's score, so the read is aligned with the CE that shapes it)
        -> ``[N, num_cg]``; compute_regression_loss selects the labeled column by
        cg_ids and the others are produced for inference.
        """
        from .bt_score import expected_score_from_logits
        if labels is None:
            raise ValueError('BT-SFT requires labels to locate the score position(s).')
        # Per-CG multi-CG uses each CG's own <cg_c_score_k> rows (shared <score_k> tokens are not even
        # registered then), so defer to _score_rows_per_cg below; otherwise use the shared score rows.
        per_cg = self._per_cg_score_tokens and self._num_cg > 1
        score_rows = None if per_cg else self._score_rows(hidden.device)  # [num_buckets, h]

        if self._num_cg > 1:
            if self._cg_marker_ids is None:
                raise ValueError('multi-CG BT-SFT requires _cg_marker_ids (set by the trainer).')
            cg_marker_ids = self._cg_marker_ids.to(hidden.device)              # [num_cg]
            src = input_ids if input_ids is not None else labels  # markers live in the assistant output
            if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
                # Padding-free/packing: hidden is [total_tokens, 1, h]; each marker
                # appears once per sample, in sample order, so nonzero gives the
                # per-sample positions for that marker.
                flat = src.reshape(-1)
                cols = []
                for c in range(self._num_cg):
                    pos_c = (flat == cg_marker_ids[c]).nonzero(as_tuple=False).flatten()  # [num_samples]
                    cols.append(hidden[pos_c, 0, :])                           # [num_samples, h]
                pooled = torch.stack(cols, dim=1)                             # [num_samples, num_cg, h]
            else:
                # Non-packing: hidden is [s, b, h], src is [b, s].
                src = src if src.ndim == 2 else src.unsqueeze(0)
                b, s = src.shape
                ar = torch.arange(s, device=src.device)
                b_idx = torch.arange(b, device=hidden.device)
                pooled = hidden.new_zeros((b, self._num_cg, hidden.shape[-1]))  # [b, num_cg, h]
                for c in range(self._num_cg):
                    mask = src == cg_marker_ids[c]
                    masked = torch.where(mask, ar.unsqueeze(0), torch.full_like(src, -1))
                    pos = masked.max(dim=1).values.clamp(min=0)
                    pooled[:, c, :] = hidden[pos, b_idx, :]
            if per_cg:
                score_rows_pc = self._score_rows_per_cg(hidden.device)         # [num_cg, num_buckets, h]
                # each CG's hidden dotted with ITS OWN score-token rows
                score_logits = torch.einsum('ncd,ckd->nck', pooled.float(), score_rows_pc)  # [N, num_cg, num_buckets]
            else:
                score_logits = pooled.float() @ score_rows.t()                # [N, num_cg, num_buckets]
            s = expected_score_from_logits(score_logits, self._bt_sft_temperature)  # [N, num_cg], in [0, N-1]
            return s / self._bt_score_denom                                   # normalize to [0, 1]

        # Single-CG path: read at the <score_k> position.
        score_ids = self._resolve_score_token_ids(hidden.device)
        if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
            lab = labels.view(-1)
            pos = torch.isin(lab, score_ids).nonzero(as_tuple=False).flatten()
            pooled = hidden[pos, 0, :]
        else:
            lbs = labels if labels.ndim == 2 else labels.unsqueeze(0)
            b, s = lbs.shape
            mask = torch.isin(lbs, score_ids)
            ar = torch.arange(s, device=lbs.device)
            masked = torch.where(mask, ar.unsqueeze(0), torch.full_like(lbs, -1))
            pos = masked.max(dim=1).values.clamp(min=0)
            b_idx = torch.arange(b, device=hidden.device)
            pooled = hidden[pos, b_idx, :]
        score_logits = pooled.float() @ score_rows.t()
        s = expected_score_from_logits(score_logits, self._bt_sft_temperature)  # in [0, N-1]
        return (s / self._bt_score_denom).unsqueeze(-1)  # [N, 1], normalized to [0, 1]

    def _sparse_lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """LM cross-entropy computed ONLY at supervised (label != -100) positions.

        ``hidden`` is the post-final-layernorm state ``[s, b, h]`` (output_layer NOT yet
        applied). With BT-SFT only a handful of positions per sample are labeled (the
        ``<cg_k>``/``<score_k>`` tokens), so projecting just those through the vocab head
        turns the dense ``[s, vocab]`` logits/softmax (the memory hog) into ``[M, vocab]``
        with M = #labeled tokens. Returns a full-length ``[b, s]`` per-token loss (0 at
        ignored positions) so the trainer's loss_func (``sum(loss * (labels != -100))``) is
        unchanged. Numerically identical to the dense LM CE (CE already ignores -100).
        """
        lm = self.language_model  # output_layer / CE live on the inner GPTModel
        lbs = labels if labels.ndim == 2 else labels.unsqueeze(0)   # [b, s]
        b, s = lbs.shape
        full_loss = hidden.new_zeros((b, s), dtype=torch.float32)
        mask = lbs != -100                                          # [b, s]
        output_weight = (lm.shared_embedding_or_output_weight()
                         if lm.share_embeddings_and_output_weights else None)
        if not bool(mask.any()):
            # No supervised token this micro-batch: keep the head in the autograd graph
            # (so grads stay defined for DDP) but contribute zero.
            w = output_weight if output_weight is not None else lm.output_layer.weight
            return full_loss + 0.0 * w.sum()
        h_sup = hidden.permute(1, 0, 2)[mask]                       # [M, h] in (b, s) order
        labels_sup = lbs[mask]                                      # [M]    in (b, s) order
        logits_sup, _ = lm.output_layer(h_sup.unsqueeze(1), weight=output_weight)  # [M, 1, vocab]
        loss_sup = lm.compute_language_model_loss(labels_sup.unsqueeze(0), logits_sup)  # [1, M]
        full_loss[mask] = loss_sup.squeeze(0).to(full_loss.dtype)
        return full_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params=None,
        packed_seq_params=None,
        **kwargs,
    ):
        self._regression_hidden = None
        if self._bt_sft and self._num_cg > 1 and self._score_unk_id is not None and labels is not None:
            # Mask the <score_unk> placeholders (unlabeled CG slots) from the LM CE
            # so we don't train the model to emit them; the <cg_c> markers and the
            # labeled <score_k> remain supervised.
            unk_mask = labels == self._score_unk_id
            labels = torch.where(unk_mask, torch.full_like(labels, -100), labels)
            if os.environ.get('BT_SFT_DEBUG') and parallel_state.get_data_parallel_rank() == 0:
                logger.info(f'[bt_dbg] label_mask: masked <score_unk>={int(unk_mask.sum())} '
                            f'remaining_labeled_tokens={int((labels != -100).sum())}')

        # ── Sparse LM head (memory/compute-efficient BT-SFT) ─────────────────────────
        # The dense LM CE projects EVERY packed position through the ~152k-wide vocab head
        # (the [seq, vocab] logits/softmax dominate activation memory), yet only ~8/sample
        # are supervised. Here we get the post-LN hidden WITHOUT the full output_layer (by
        # toggling post_process off for the base forward -- the decoder still applies
        # final_layernorm) and project only the supervised positions. Numerically identical.
        # Guarded to TP=1/CP=1 (the current config); falls back to the dense path otherwise.
        _lm = self.language_model  # the inner GPTModel that owns the output_layer
        if (self._bt_sft and getattr(_lm, 'post_process', False) and labels is not None
                and os.environ.get('BT_SFT_SPARSE_LM_HEAD', '1') != '0'
                and parallel_state.get_tensor_model_parallel_world_size() == 1
                and parallel_state.get_context_parallel_world_size() == 1):
            # Toggle the INNER GPTModel's post_process (MultimodalGPTModel.forward just
            # returns self.language_model(...)), so super().forward skips the output_layer
            # and returns post-LN hidden [s, b, h] instead of full [b, s, vocab] logits.
            saved_pp = _lm.post_process
            _lm.post_process = False
            try:
                hidden = super().forward(
                    input_ids, position_ids, attention_mask=attention_mask,
                    decoder_input=decoder_input, labels=None, inference_params=inference_params,
                    packed_seq_params=packed_seq_params, **kwargs)
            finally:
                _lm.post_process = saved_pp
            self._regression_hidden = None
            lm_output = self._sparse_lm_loss(hidden, labels)
            reg_logits = self._bt_sft_score(hidden, labels, packed_seq_params, input_ids=input_ids)
            return lm_output, reg_logits

        lm_output = super().forward(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
        if self._regression_hidden is not None:
            hidden = self._regression_hidden
            self._regression_hidden = None  # release ref immediately

            # Sequence-parallel compatibility: when TP > 1 and sequence_parallel is
            # enabled, final_layernorm's output is [s/TP, b, h] — the sequence is
            # scattered across the tensor-parallel group.  last_idx values are global
            # token positions, so we must gather the full [s, b, h] tensor first.
            # With TP=1 (current default) this branch is skipped entirely.
            if (parallel_state.get_tensor_model_parallel_world_size() > 1
                    and getattr(self.config, "sequence_parallel", False)):
                from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
                hidden = gather_from_sequence_parallel_region(hidden)

            if self._bt_sft:
                # Verbalized scoring: read the N <score_k> logits off the LM head at
                # the score-token position and return the softmax-expected value,
                # normalized to [0, 1], as a 1-D regression logit ([N, 1]) for the
                # bradley_terry loss.
                reg_logits = self._bt_sft_score(hidden, labels, packed_seq_params, input_ids=input_ids)
                return lm_output, reg_logits

            if packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
                # Packing mode: hidden is [total_tokens, 1, h].
                cu_seqlens = packed_seq_params.cu_seqlens_q
                num_samples = getattr(packed_seq_params, 'num_samples', cu_seqlens.shape[0] - 1)
                if labels is not None:
                    # Select the hidden state at the \n token right after "assistant"
                    # in the chat template (<|im_start|>assistant\n).  This is the last
                    # position with label=-100 before the first labeled token of the
                    # last assistant turn, i.e. first_labeled_in_last_turn - 1.
                    pool_idx = _last_assistant_nl_idx(labels.view(-1), cu_seqlens, num_samples)
                else:
                    pool_idx = cu_seqlens[1:num_samples + 1] - 1
                pooled = hidden[pool_idx, 0, :]           # [N, h]
                reg_logits = self.regression_head(pooled) # [N, regression_dim]
                # reg_logits is 2-D; compute_regression_loss detects this.
            else:
                # Non-packing: hidden is [s, b, h], labels is [b, s].
                if labels is not None:
                    lbs = labels if labels.ndim == 2 else labels.unsqueeze(0)
                    b = lbs.shape[0]
                    pool_idx = _last_assistant_nl_idx_batched(lbs, hidden.device)  # [b]
                    b_idx = torch.arange(b, device=hidden.device)
                    pooled = hidden[pool_idx, b_idx, :]   # [b, h]
                else:
                    pooled = hidden[-1, :, :]             # [b, h] fallback
                reg_logits = self.regression_head(pooled) # [b, regression_dim]
                # Treat as 2-D (pre-pooled) so compute_regression_loss skips pooling.
            return lm_output, reg_logits
        return lm_output


def _get_thresholds_for_task_ids(
    thresholds: torch.Tensor,
    task_ids: torch.Tensor,
) -> torch.Tensor:
    """Return a per-sample threshold vector, looked up by task_id.

    If ``thresholds`` has only one element it is broadcast to all samples.
    Out-of-range task_ids are clamped to the last valid index.
    """
    thresholds = thresholds.to(device=task_ids.device, dtype=torch.float32)
    if thresholds.numel() == 1:
        return thresholds.expand(task_ids.size(0))
    task_ids = task_ids.clamp(0, thresholds.numel() - 1)
    return thresholds[task_ids]


def _threshold_aware_soft_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
    task_ids: torch.Tensor,
    scale: float,
    tau: float,
) -> torch.Tensor:
    """Soft-label BCE that places the decision boundary at the per-task threshold.

    Returns a zero-gradient scalar when ``scale <= 0`` or there are no samples.
    """
    if scale <= 0 or logits.numel() == 0:
        return logits.sum() * 0.0

    sample_thresh = _get_thresholds_for_task_ids(thresholds, task_ids)

    eps = 1e-6
    # Shift logits so the decision boundary (p = threshold) lies at 0
    threshold_logits = torch.logit(sample_thresh.clamp(eps, 1.0 - eps))
    cls_logits = logits - threshold_logits

    # Soft targets: 0 → below threshold, 1 → above, smooth transition of width tau
    tau = max(float(tau), eps)
    soft_targets = torch.sigmoid((labels - sample_thresh) / tau)

    loss = F.binary_cross_entropy_with_logits(cls_logits, soft_targets, reduction='none')
    return scale * loss.mean()


def _bradley_terry_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    args,
    task_ids: Optional[torch.Tensor] = None,
    model=None,
) -> torch.Tensor:
    """Pairwise Bradley-Terry ranking loss with optional memory bank augmentation.

    When ``model`` has a populated ``_logit_queue`` (i.e. ``--bt-queue-size > 0``),
    the current-batch logits/labels are concatenated with the queue contents before
    computing pairwise losses. The batch is then enqueued *after* reading so it
    does not pair with itself. Loss is averaged across unique task groups.
    """
    contrastive_margin = getattr(args, 'bt_margin', 0.05)

    if task_ids is None:
        task_ids = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    # ── Memory bank: augment with recent samples ────────────────────────────────
    use_queue = (model is not None and hasattr(model, '_logit_queue')
                 and getattr(model, 'training', True))
    if use_queue:
        q_size = model._py_queue_count  # Python int — no GPU sync
        if q_size > 0:
            bt_logits = torch.cat([logits, model._logit_queue[:q_size].detach()])
            bt_labels = torch.cat([labels, model._label_queue[:q_size].detach()])
            bt_task_ids = torch.cat([task_ids, model._task_queue[:q_size].detach()])
        else:
            bt_logits, bt_labels, bt_task_ids = logits, labels, task_ids
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'[bt_loss] queue active  batch={logits.size(0)}  '
                f'queue_fill={q_size}  total_for_bt={bt_logits.size(0)}'
            )
        # Enqueue AFTER reading so the batch doesn't pair with itself.
        if logits.numel() > 0:
            model._enqueue(logits.detach(), labels.detach(), task_ids.detach())
    else:
        bt_logits, bt_labels, bt_task_ids = logits, labels, task_ids

    # ── Per-task pairwise BT loss ────────────────────────────────────────────────
    if bt_logits.size(0) < 2:
        return logits.sum() * 0.0

    # Accumulate on-GPU without any .any()/.item() sync inside the loop.
    bt_loss_total = logits.new_zeros(())
    tasks_with_pairs_t = logits.new_zeros(())
    for task_id in torch.unique(bt_task_ids):
        mask = bt_task_ids == task_id
        t_logits = bt_logits[mask]
        t_labels = bt_labels[mask]
        n = t_logits.size(0)
        if n < 2:
            continue
        sorted_idx = torch.argsort(t_labels, descending=True)
        s_logits = t_logits[sorted_idx]
        s_labels = t_labels[sorted_idx]
        logit_diff = s_logits.unsqueeze(1) - s_logits.unsqueeze(0)
        label_diff = s_labels.unsqueeze(1) - s_labels.unsqueeze(0)
        upper = torch.triu(torch.ones(n, n, device=logits.device, dtype=torch.bool), diagonal=1)
        valid = upper & (label_diff > contrastive_margin)
        # Avoid valid.any() (GPU→CPU sync); use tensor masking instead.
        n_valid = valid.sum()
        has_pairs = (n_valid > 0).to(logit_diff.dtype)
        task_loss = -(F.logsigmoid(logit_diff) * valid.to(logit_diff.dtype)).sum() / n_valid.clamp(min=1)
        bt_loss_total = bt_loss_total + task_loss * has_pairs
        tasks_with_pairs_t = tasks_with_pairs_t + has_pairs
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'[bt_loss] task_id={int(task_id.item())}  '
                f'n_samples={n}  valid_pairs={int(n_valid.item())}  margin={contrastive_margin}'
            )

    final_bt_loss = bt_loss_total / tasks_with_pairs_t.clamp(min=1)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[bt_loss] tasks_with_pairs={tasks_with_pairs_t.item():.0f}  bt_loss={final_bt_loss.item():.6f}'
        )
    if os.environ.get('BT_SFT_DEBUG') and parallel_state.get_data_parallel_rank() == 0:
        q_fill = int(getattr(model, '_py_queue_count', 0)) if use_queue else 0
        per_cg = {int(t.item()): int((bt_task_ids == t).sum().item()) for t in torch.unique(bt_task_ids)}
        logger.info(f'[bt_dbg] bt_queue: cur_batch={logits.size(0)} queue_fill={q_fill} '
                    f'total_for_bt={bt_logits.size(0)} per_cg_counts={per_cg} '
                    f'tasks_with_pairs={int(tasks_with_pairs_t.item())} bt_loss={float(final_bt_loss.item()):.4f}')
    return final_bt_loss


def _mc_enqueue(model, logits2d: torch.Tensor, labels2d: torch.Tensor) -> None:
    """Circular enqueue for the item-granular multi-label BT memory bank.

    Each entry is one item's ``[num_cg]`` logit/label vector (label NaN where that CG is
    unlabeled), so a queued item carries its own list of cg_ids = its non-NaN columns.
    Lazily allocates the buffers on first use (shape ``[bt_queue_size, num_cg]``).
    """
    Q = int(getattr(model, '_bt_queue_size', 0))
    if Q <= 0:
        return
    num_cg = logits2d.size(-1)
    if not hasattr(model, '_mc_logit_queue'):
        dev = logits2d.device
        model.register_buffer('_mc_logit_queue', torch.zeros(Q, num_cg, device=dev), persistent=False)
        model.register_buffer('_mc_label_queue', torch.full((Q, num_cg), float('nan'), device=dev),
                              persistent=False)
        model._mc_queue_ptr = 0
        model._mc_queue_count = 0
    b = logits2d.size(0)
    if b >= Q:
        model._mc_logit_queue.copy_(logits2d[-Q:])
        model._mc_label_queue.copy_(labels2d[-Q:])
        model._mc_queue_ptr = 0
        model._mc_queue_count = Q
        return
    ptr = model._mc_queue_ptr
    end = ptr + b
    if end <= Q:
        model._mc_logit_queue[ptr:end] = logits2d
        model._mc_label_queue[ptr:end] = labels2d
    else:
        first = Q - ptr
        model._mc_logit_queue[ptr:] = logits2d[:first]
        model._mc_label_queue[ptr:] = labels2d[:first]
        model._mc_logit_queue[:b - first] = logits2d[first:]
        model._mc_label_queue[:b - first] = labels2d[first:]
    model._mc_queue_ptr = end % Q
    model._mc_queue_count = min(model._mc_queue_count + b, Q)


def _bradley_terry_loss_multilabel(logits2d: torch.Tensor, labels2d: torch.Tensor, args,
                                   model=None) -> torch.Tensor:
    """BT loss for items that may carry SEVERAL CG scores.

    ``logits2d`` / ``labels2d`` are ``[N, num_cg]``; ``labels2d`` is NaN where a CG is
    unlabeled for that item. Each item keeps its OWN list of labeled CGs (its non-NaN
    columns). The memory bank is item-granular: each queued entry is one item's
    ``[num_cg]`` logit/label vector (so it carries that item's cg_ids). Within each CG
    column, all labeled entries (current batch + queue) are ranked pairwise with
    Bradley-Terry; the loss is averaged over CGs that have a valid pair.
    """
    margin = getattr(args, 'bt_margin', 0.05)
    num_cg = logits2d.size(-1)

    use_queue = (model is not None and int(getattr(model, '_bt_queue_size', 0)) > 0
                 and getattr(model, 'training', True))
    if use_queue:
        q = int(getattr(model, '_mc_queue_count', 0)) if hasattr(model, '_mc_logit_queue') else 0
        if q > 0:
            all_logits = torch.cat([logits2d, model._mc_logit_queue[:q].detach()], dim=0)
            all_labels = torch.cat([labels2d, model._mc_label_queue[:q].detach()], dim=0)
        else:
            all_logits, all_labels = logits2d, labels2d
        _mc_enqueue(model, logits2d.detach(), labels2d.detach())  # enqueue AFTER reading
    else:
        all_logits, all_labels = logits2d, labels2d

    # Fully vectorized over CGs -- no per-CG Python loop, no ``.item()`` / boolean-gather /
    # ``nonzero`` (each of those forces a device sync; a profiler trace showed ~64 such syncs
    # per step here). For every CG, all ordered pairs (i>j by >margin) are scored at once via a
    # multiplicative mask, so the result is numerically identical to the per-CG sorted-triu form
    # (verified across randomized trials). Cost is a [M, M, num_cg] tensor with M = batch+queue
    # (<= a few hundred), which is negligible.
    vmask = ~torch.isnan(all_labels)                                      # [M, num_cg]
    labels_f = torch.where(vmask, all_labels, all_labels.new_zeros(()))   # NaN -> 0 (masked out below)
    logit_diff = all_logits.unsqueeze(1) - all_logits.unsqueeze(0)        # [M, M, num_cg] = logit_i - logit_j
    label_diff = labels_f.unsqueeze(1) - labels_f.unsqueeze(0)            # [M, M, num_cg] = label_i - label_j
    pair = vmask.unsqueeze(1) & vmask.unsqueeze(0) & (label_diff > margin)  # both labeled AND i outranks j
    pair_f = pair.to(logit_diff.dtype)
    n_valid = pair_f.sum(dim=(0, 1))                                      # [num_cg]
    loss_per_cg = -(F.logsigmoid(logit_diff) * pair_f).sum(dim=(0, 1)) / n_valid.clamp(min=1)  # [num_cg]
    has_pairs = (n_valid > 0).to(logit_diff.dtype)                       # [num_cg]
    return (loss_per_cg * has_pairs).sum() / has_pairs.sum().clamp(min=1)


def _dense_cg_labels(regression_labels, cg_ids, num_cg: int, ref: torch.Tensor):
    """Project per-item CG labels into a dense ``[N, num_cg]`` matrix (NaN = unlabeled).

    Unifies the single-label and multi-label BT-SFT cases. Single-label data provides a
    1-D ``[N]`` score plus ``cg_ids`` selecting each item's one labeled CG — that is just
    the dense form with exactly one non-NaN entry per row. Already-dense ``[N, num_cg]``
    input (merged multi-label data) is returned unchanged.
    """
    if regression_labels is None:
        return None
    N = ref.size(0)
    rl = regression_labels.float().to(ref.device)
    if rl.dim() == 2:
        return rl
    rl = rl.reshape(-1)
    if rl.numel() != N:  # packed: align scalar-per-sample to N rows
        repeat = math.ceil(N / max(rl.numel(), 1))
        rl = rl.repeat_interleave(repeat)[:N]
    if cg_ids is not None:
        idx = cg_ids.to(device=ref.device, dtype=torch.long).reshape(-1)
        if idx.numel() != N:
            repeat = math.ceil(N / max(idx.numel(), 1))
            idx = idx.repeat_interleave(repeat)[:N]
    else:
        idx = torch.zeros(N, dtype=torch.long, device=ref.device)
    idx = idx.clamp(0, num_cg - 1)
    dense = torch.full((N, num_cg), float('nan'), device=ref.device, dtype=torch.float32)
    if N > 0:
        dense[torch.arange(N, device=ref.device), idx] = rl
    return dense


def _compute_combined_bt_mse_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    args,
    model,
    thresholds: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full combined BT + Huber regression loss.

    Components
    ----------
    regression_loss
        Per-sample Huber loss, optionally reweighted by focal and
        positive-class factors.
    bt_loss
        Bradley-Terry pairwise ranking loss scaled by ``--l2-lambda``.
    threshold_bce_loss
        Soft BCE centred at the per-task decision threshold
        (``--threshold-soft-bce-scale``).
    """
    preds = torch.sigmoid(logits)

    # ── Per-sample Huber loss ────────────────────────────────────────────────
    per_sample_loss = F.huber_loss(preds, labels, delta=0.1, reduction='none')

    # ── Focal weighting ──────────────────────────────────────────────────────
    focal_beta = getattr(args, 'focal_beta', 0.0)
    focal_mode = getattr(args, 'focal_mode', 'boundary')
    if focal_beta > 0 and focal_mode != 'off':
        sample_thresh = _get_thresholds_for_task_ids(thresholds, task_ids)
        if focal_mode == 'confidence':
            max_dist = torch.maximum(sample_thresh, 1.0 - sample_thresh).clamp(min=1e-6)
            distance = ((labels - sample_thresh) / max_dist).pow(2)
            focal_weight = 1.0 + focal_beta * distance
        else:  # 'boundary': upweight samples near extremes (0 or 1)
            focal_weight = 1.0 + focal_beta * (1.0 - (2.0 * labels - 1.0).pow(2))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'[combined_bt_mse] focal: mode={focal_mode!r}  beta={focal_beta}  '
                f'focal_weight=[{focal_weight.min().item():.3f},{focal_weight.max().item():.3f}]'
                f' mean={focal_weight.mean().item():.3f}'
            )
        per_sample_loss = focal_weight * per_sample_loss

    # ── Positive-class rebalancing ───────────────────────────────────────────
    cont_pos_weight = getattr(args, 'continuous_pos_weight', 1.0)
    if cont_pos_weight != 1.0:
        sample_thresh = _get_thresholds_for_task_ids(thresholds, task_ids)
        ramp = torch.sigmoid((labels - sample_thresh) / 0.05)
        cont_weight = 1.0 + (cont_pos_weight - 1.0) * ramp
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f'[combined_bt_mse] cont_pos_weight={cont_pos_weight}  '
                f'ramp={[round(x, 3) for x in ramp.tolist()]}  '
                f'cont_weight={[round(x, 3) for x in cont_weight.tolist()]}'
            )
        per_sample_loss = per_sample_loss * cont_weight

    regression_loss = per_sample_loss.mean() if per_sample_loss.numel() > 0 else logits.sum() * 0.0

    # ── BT ranking loss ───────────────────────────────────────────────────────
    l2_lambda = getattr(args, 'l2_lambda', 0.1)
    bt_loss = _bradley_terry_loss(logits, labels, args, task_ids=task_ids, model=model)

    # ── Threshold-aware soft BCE ──────────────────────────────────────────────
    tbce_scale = float(os.environ.get('BCE_SCALE', str(getattr(args, 'threshold_soft_bce_scale', 0.0))))
    tbce_tau = getattr(args, 'threshold_soft_bce_tau', 0.03)
    threshold_bce_loss = _threshold_aware_soft_bce(
        logits, labels, thresholds, task_ids, scale=tbce_scale, tau=tbce_tau)

    total_loss = regression_loss + l2_lambda * bt_loss + threshold_bce_loss

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[combined_bt_mse] SUMMARY  n={logits.numel()}  '
            f'reg_loss={regression_loss.item():.6f}  '
            f'bt_loss={bt_loss.item():.6f} (x{l2_lambda})  '
            f'tbce_loss={threshold_bce_loss.item():.6f}  '
            f'total={total_loss.item():.6f}  '
            f'focal_beta={focal_beta}  '
            f'cont_pos_weight={cont_pos_weight}'
        )

    with torch.no_grad():
        metrics: Dict[str, torch.Tensor] = {
            'reg_huber_loss':         regression_loss.detach(),
            'reg_bt_loss':            bt_loss.detach(),
            'reg_threshold_bce_loss': threshold_bce_loss.detach(),
            'reg_pred_mean':          preds.detach().mean(),
            'reg_pred_std':           preds.detach().std() if preds.numel() > 1 else preds.new_zeros(()),
            'reg_logit_mean':         logits.detach().mean(),
            'reg_logit_std':          logits.detach().std() if logits.numel() > 1 else logits.new_zeros(()),
        }

    return total_loss, metrics


def compute_regression_loss(
    reg_logits: torch.Tensor,
    regression_labels: torch.Tensor,
    cg_ids: Optional[torch.Tensor],
    packed_seq_params,
    model=None,
    attention_mask: Optional[torch.Tensor] = None,
    score_percentile: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute regression loss from the AIP regression head output.

    Args:
        reg_logits: ``[b, s, regression_dim]`` tensor (mcore [s,b,h] transposed).
            In padding-free mode ``b=1`` and ``s`` is the total token count.
        regression_labels: ``[batch_size]`` float tensor of per-sample targets.
        score_percentile: optional per-sample mid-rank percentile target(s), same shape
            family as ``regression_labels``. When the model has ``huber_target='score_percentile'``
            this replaces ``regression_labels`` as the Huber target (BT ranking still uses
            ``regression_labels``).
        cg_ids: optional ``[batch_size]`` int64 tensor selecting which head output
            to use for each sample when ``regression_dim > 1``.
        packed_seq_params: Megatron ``PackedSeqParams`` for padding-free mode.
        model: optional ``Qwen3VLAIPModel`` instance for the BT memory bank.
        attention_mask: optional ``[b, s]`` int/bool tensor for padded batching.
            Used to find the actual last real token rather than always taking
            position ``-1``.

    Returns:
        Tuple of (scalar loss, metrics dict).
    """
    args = getattr(getattr(model, "config", None), "aip_train_args", None)
    if args is None:
        from types import SimpleNamespace
        args = SimpleNamespace(regression_loss_type='mse', decision_threshold=0.5, bt_margin=0.05, l2_lambda=0.1, focal_beta=0.0, focal_mode='boundary', continuous_pos_weight=1.0, threshold_soft_bce_scale=0.0, threshold_soft_bce_tau=0.03)
    # Prefer model-stored loss_type (captured during __init__ when get_args() is
    # definitely correct) over the live get_args() call, as extra_args forwarding
    # may not reliably reach validate_args for all Megatron builds.
    if model is not None and hasattr(model, '_regression_loss_type'):
        loss_type = model._regression_loss_type
    else:
        loss_type = getattr(args, 'regression_loss_type', 'mse')
    # ── Pool to one vector per sample ──────────────────────────────────────────
    if reg_logits.ndim == 2:
        # forward() already pooled to the \n-after-'assistant' position and ran
        # the regression head.  Shape is [N, regression_dim] (packed) or
        # [b, regression_dim] (non-packed).  No further pooling needed.
        pooled = reg_logits
    elif packed_seq_params is not None and packed_seq_params.cu_seqlens_q is not None:
        # Fallback packing path (reg_logits still [1, total_tokens, regression_dim]).
        # Use num_samples from packed_seq_params (set by _prepare_batch); cu_seqlens_q
        # may have extra entries for SP-alignment padding tokens.
        cu_seqlens = packed_seq_params.cu_seqlens_q
        num_samples = getattr(packed_seq_params, 'num_samples', cu_seqlens.shape[0] - 1)
        last_idx = cu_seqlens[1:num_samples + 1] - 1  # [N]
        pooled = reg_logits[0, last_idx]              # [N, regression_dim]
    else:
        # Padded batching: find the actual last non-padding token per sample.
        if attention_mask is not None:
            seq_lens = attention_mask.to(device=reg_logits.device).sum(dim=1).long()
            last_idx = (seq_lens - 1).clamp(min=0)
            b_idx = torch.arange(reg_logits.size(0), device=reg_logits.device)
            pooled = reg_logits[b_idx, last_idx]      # [b, regression_dim]
        else:
            pooled = reg_logits[:, -1, :]

    # ── Unified multi-CG BT-SFT loss ────────────────────────────────────────────
    # Single-label and multi-label BT-SFT run through ONE path. We project the per-item
    # CG labels into a dense [N, num_cg] matrix (NaN = unlabeled) and run the item-granular
    # Bradley-Terry loss over every labeled (item, cg) pair. Single-label data (1-D scores
    # + cg_ids) is just the special case of one non-NaN entry per row; merged multi-label
    # data is already dense. The memory bank is item-granular: each queued entry is one
    # item's [num_cg] logit/label vector, carrying that item's cg_ids = its non-NaN columns.
    if loss_type == 'bradley_terry' or (regression_labels is not None and regression_labels.dim() == 2):
        logits2d = pooled.float() if pooled.dim() == 2 else pooled.float().reshape(pooled.size(0), -1)
        num_cg = logits2d.size(-1)
        rl2d = _dense_cg_labels(regression_labels, cg_ids, num_cg, logits2d)
        if rl2d is None or not torch.isfinite(rl2d).any():
            z = logits2d.new_zeros(())
            return logits2d.sum() * 0.0, {
                'reg_bt_loss': z, 'reg_huber_loss': z, 'reg_pred_mean': z, 'reg_logit_mean': z}
        bt_loss = _bradley_terry_loss_multilabel(logits2d, rl2d, args, model=model)
        labeled = ~torch.isnan(rl2d)
        lab_logits = logits2d[labeled] if labeled.any() else logits2d.reshape(-1)[:0]
        # Independent BT-SFT loss weights: total = bt_loss_weight * bt_loss + huber_loss_weight * huber,
        # where huber is a pointwise Huber on the normalized score proxy vs the (normalized) GT score
        # over labeled (item, cg) entries. The trainer applies weight 1.0 for bt_sft (NOT
        # reg_loss_weight), so these two weights are the absolute contributions of each term.
        bt_w = float(getattr(model, '_bt_loss_weight', 1.0)) if model is not None else 1.0
        huber_w = float(getattr(model, '_huber_loss_weight', 0.0)) if model is not None else 0.0
        # Huber target: percentile (if opted in AND the column is present) else regression_labels.
        # Only the Huber term switches; BT ranking above always used rl2d (regression_labels).
        use_pct = bool(getattr(model, '_huber_use_percentile', False)) if model is not None else False
        huber_target_2d = rl2d
        if use_pct and score_percentile is not None:
            pct2d = _dense_cg_labels(score_percentile, cg_ids, num_cg, logits2d)
            if pct2d is not None:
                huber_target_2d = pct2d
        huber_loss = logits2d.new_zeros(())
        # Regress only where the chosen target is finite (percentile is null-aligned with
        # regression_labels by construction, but guard against any mismatch).
        hmask = labeled & torch.isfinite(huber_target_2d)
        if huber_w > 0 and hmask.any():
            huber_loss = F.huber_loss(logits2d[hmask], huber_target_2d[hmask].float(), delta=0.1)
        total_loss = bt_w * bt_loss + huber_w * huber_loss
        return total_loss, {
            'reg_bt_loss':    bt_loss.detach(),      # raw (unweighted) component losses
            'reg_huber_loss': huber_loss.detach(),
            'reg_pred_mean':  torch.sigmoid(lab_logits).detach().mean() if lab_logits.numel() else logits2d.new_zeros(()),
            'reg_logit_mean': lab_logits.detach().mean() if lab_logits.numel() else logits2d.new_zeros(()),
        }

    # ── Build task_ids ─────────────────────────────────────────────────────────
    num_samples = pooled.size(0)
    if cg_ids is not None:
        cg_ids_flat = cg_ids.to(device=pooled.device, dtype=torch.long).reshape(-1)
        if cg_ids_flat.numel() != num_samples:
            # Packed sequences: cg_ids has micro_batch_size entries but num_samples packed
            # sequences. The round-robin sampler keeps CGs consistent within a step, so
            # repeating is correct.
            repeat = math.ceil(num_samples / cg_ids_flat.numel())
            cg_ids_flat = cg_ids_flat.repeat_interleave(repeat)[:num_samples]
        task_ids = cg_ids_flat
    else:
        task_ids = torch.zeros(num_samples, dtype=torch.long, device=pooled.device)

    unique_task_ids, task_counts = task_ids.unique(return_counts=True)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[compute_reg_loss] num_samples={num_samples}  regression_dim={pooled.size(-1)}  '
            f'cg_ids_input={cg_ids.tolist() if cg_ids is not None else None}  '
            f'task_ids={task_ids.tolist()}  '
            f'task_distribution={dict(zip(unique_task_ids.tolist(), task_counts.tolist()))}'
        )

    # ── Select head output by cg_id for multi-output heads ─────────────────────
    if pooled.size(-1) > 1:
        if cg_ids is not None:
            batch_idx = torch.arange(num_samples, device=pooled.device)
            logits = pooled[batch_idx, task_ids]
        else:
            logits = pooled[:, 0]
    else:
        logits = pooled.squeeze(-1)

    logits = logits.float()
    if logits.numel() == 0:
        # No score position in this micro-batch (e.g. the <score_k> token was
        # truncated at max_length). Skip the BT loss for this item and do NOT
        # enqueue it (we return before _bradley_terry_loss). Return a graph-
        # connected zero loss plus zero-valued metrics with the SAME keys the
        # bradley_terry path emits — Megatron requires every micro-batch to
        # report identical metric keys when reducing across the batch.
        z = logits.new_zeros(())
        return logits.sum() * 0.0, {'reg_bt_loss': z, 'reg_pred_mean': z, 'reg_logit_mean': z}
    regression_labels = regression_labels.float().to(logits.device).reshape_as(logits)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[compute_reg_loss] n_samples={logits.numel()}  '
            f'logits=[{logits.min().item():.4f}, {logits.max().item():.4f}]  '
            f'labels=[{regression_labels.min().item():.4f}, {regression_labels.max().item():.4f}]  '
            f'labels={regression_labels.tolist()}'
        )

    if logits.numel() == 0:
        return logits.sum() * 0.0, {}

    # ── Retrieve per-task decision thresholds ──────────────────────────────────
    if model is not None and hasattr(model, 'decision_thresholds'):
        thresholds = model.decision_thresholds.to(device=logits.device, dtype=torch.float32)
        threshold_source = 'model.decision_thresholds'
    else:
        thresholds = logits.new_tensor([float(getattr(args, 'decision_threshold', 0.5))])
        threshold_source = 'args.decision_threshold'
    per_sample_thresholds = thresholds[task_ids.clamp(max=thresholds.size(0) - 1)]
    _thresh_mode = 'broadcast' if thresholds.numel() == 1 else 'per_task'
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f'[compute_reg_loss] thresholds={thresholds.tolist()}  source={threshold_source}  '
            f'mode={_thresh_mode}  '
            + (f'broadcast_value={thresholds.item():.4f}' if thresholds.numel() == 1
               else f'per_sample_thresholds={per_sample_thresholds.tolist()}')
        )

    # ── Dispatch ───────────────────────────────────────────────────────────────
    if loss_type == 'combined_bt_mse':
        return _compute_combined_bt_mse_loss(
            logits, regression_labels, task_ids, args, model, thresholds)

    # Simple pointwise losses
    pred = torch.sigmoid(logits)
    if loss_type == 'mse':
        loss = F.mse_loss(pred, regression_labels)
    elif loss_type == 'mae':
        loss = F.l1_loss(pred, regression_labels)
    else:  # huber
        loss = F.huber_loss(pred, regression_labels, delta=0.1)
    metrics = {
        'reg_huber_loss': loss.detach(),
        'reg_pred_mean':  pred.detach().mean(),
        'reg_logit_mean': logits.detach().mean(),
    }
    return loss, metrics


class Qwen3VLAIPLoader(Qwen3VLLoader):
    model_cls = Qwen3VLAIPModel


def make_aip_loader(base_loader_cls):
    """Wrap an arbitrary registered mcore loader so it builds the AIP model.

    ``Qwen3VLAIPModel`` subclasses ``MultimodalGPTModel`` and is model-agnostic:
    it only touches generic mcore attributes (``self.config``, ``self.post_process``,
    ``self.language_model.decoder``, ``self.language_model.output_layer`` /
    ``shared_embedding_or_output_weight``). The same model class therefore works for
    any multimodal GPT base (``qwen3_vl``, ``qwen3_5``, ...). We only need to swap in
    ``model_cls`` on top of the model's own registered loader so that all of the base
    loader's layer-spec / transformer-block / GDN wiring is preserved.
    """
    if getattr(base_loader_cls, 'model_cls', None) is Qwen3VLAIPModel:
        return base_loader_cls
    return type(f'{base_loader_cls.__name__}AIP', (base_loader_cls, ), {'model_cls': Qwen3VLAIPModel})
