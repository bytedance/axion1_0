# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reasoning BT-SFT: 1/K reasoning CE weight on top of the AIP BT-SFT model.

Recipe (one-hot supervision over the joint multi-CG reasoning format): a training
item labeled on K CGs is expanded at data-prep time into K copies. Every copy keeps
the SAME assistant sequence -- the joint reasoning trace over all CGs followed by the
full ``<cg_0><score?>...<cg_{C-1}><score?>`` block -- but copy j puts the real
``<score_k>`` only in CG j's slot (``<score_unk>`` elsewhere), so CE/BT gradients are
routed one CG at a time (the single-label dynamics that outperform multi-label).

The side effect this file corrects: the shared reasoning trace is then CE-supervised
K times per epoch instead of once, over-weighting exactly the multi-violation items.
``Qwen3VLAIPReasoningModel`` therefore scales the per-token LM CE of every
*reasoning* position by ``1/K``, while *structural* positions -- the ``<cg_c>``
markers and the ``<score_k>`` tokens (shared or per-CG), whose supervision is unique
to each copy -- keep full weight. Positions with label ``-100`` (prompt,
``<score_unk>`` slots) are unaffected either way.

K reaches the model as a per-sample ``num_cg_labels`` batch field (int, the number of
labeled CGs of the ORIGINAL item; every copy carries the same value). The field rides
the same dataset -> template -> collator path as ``cg_ids`` and lands in
``model.forward`` as a kwarg; when it is absent (plain single-label data, eval) the
model behaves exactly like ``Qwen3VLAIPModel``.

Enable with ``--bt_sft_reasoning_ce true`` (routes the loader here, see
``swift/megatron/model/utils.py``) plus a ``num_cg_labels`` column in the dataset.
"""
import os
from typing import Optional

import torch
from megatron.core import parallel_state

from mcore_bridge.model.mm_gpts.qwen3_vl import Qwen3VLLoader
from swift.utils import get_logger

from .qwen3_vl_aip import Qwen3VLAIPModel

logger = get_logger()


class Qwen3VLAIPReasoningModel(Qwen3VLAIPModel):
    """``Qwen3VLAIPModel`` + per-token 1/K CE weight on reasoning tokens.

    Both LM-CE paths of the parent return a full-length per-token loss tensor
    ([b, s], zeros at ignored positions): the sparse LM head path via
    ``_sparse_lm_loss`` and the dense path via mcore's
    ``compute_language_model_loss``. This subclass therefore only has to scale that
    tensor elementwise before it reaches the trainer's ``loss_func`` (which computes
    ``sum(loss * (labels != -100))``); the token-count denominator is untouched, so a
    reasoning token simply contributes 1/K of its CE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_args = getattr(self.config, 'aip_train_args', None)
        # The class itself is the opt-in; the flag only exists to allow an explicit
        # off-switch on a checkpoint that was wired to this loader.
        self._reasoning_ce_1_over_k: bool = bool(getattr(train_args, 'bt_sft_reasoning_ce', True)) \
            if train_args is not None else True
        self._structural_ids: Optional[torch.Tensor] = None  # lazily built, device-cached
        self._reasoning_dbg_logged: bool = False
        logger.info(f'[Qwen3VLAIPReasoningModel.__init__] reasoning_ce_1_over_k={self._reasoning_ce_1_over_k}  '
                    f'(bt_sft={self._bt_sft}  num_cg={self._num_cg})')

    def _structural_token_ids(self, device) -> torch.Tensor:
        """Vocab ids whose CE keeps full weight: score tokens + <cg_c> markers.

        ``<score_unk>`` is included for completeness only -- its labels are already
        masked to -100 by the parent's forward, so its weight never matters.
        Built lazily on the first post-process forward (after the trainer has stashed
        the marker/score ids in ``prepare_model``) and cached per device.
        """
        if self._structural_ids is not None:
            if self._structural_ids.device != device:
                self._structural_ids = self._structural_ids.to(device)
            return self._structural_ids
        parts = []
        if self._per_cg_score_tokens and self._num_cg > 1:
            if self._per_cg_score_token_ids is None:
                raise ValueError('per-CG BT-SFT requires _per_cg_score_token_ids (set by the trainer).')
            parts.append(self._per_cg_score_token_ids.reshape(-1))
        else:
            parts.append(self._resolve_score_token_ids(device))
        if self._cg_marker_ids is not None:
            parts.append(self._cg_marker_ids.reshape(-1))
        if self._score_unk_id is not None:
            parts.append(torch.tensor([int(self._score_unk_id)], dtype=torch.long))
        ids = torch.cat([p.to(device=device, dtype=torch.long) for p in parts]).unique()
        self._structural_ids = ids
        logger.info(f'[Qwen3VLAIPReasoningModel] structural token ids ({ids.numel()}): {ids.tolist()}')
        return ids

    def _reasoning_ce_scale(self, labels: torch.Tensor, num_cg_labels: torch.Tensor,
                            packed_seq_params) -> torch.Tensor:
        """Per-token CE scale, same shape as ``labels``: 1/K at reasoning positions,
        1.0 at structural positions (and everywhere unsupervised -- a no-op there).

        Works for both layouts the parent supports: packed/padding-free
        (labels [1, total_tokens], per-sample spans from ``cu_seqlens_q``) and padded
        batching (labels [b, s], one K per row). No ``.item()``/``nonzero`` syncs.
        """
        lbs = labels if labels.ndim == 2 else labels.unsqueeze(0)  # [b, s]
        w = 1.0 / num_cg_labels.to(device=lbs.device, dtype=torch.float32).reshape(-1).clamp(min=1)

        if packed_seq_params is not None and getattr(packed_seq_params, 'cu_seqlens_q', None) is not None:
            cu_seqlens = packed_seq_params.cu_seqlens_q
            num_samples = int(getattr(packed_seq_params, 'num_samples', cu_seqlens.shape[0] - 1))
            # Under CP the sequence dim is sliced per rank while cu_seqlens counts the
            # full sequence (same convention as _compute_channel_loss).
            cp = parallel_state.get_context_parallel_world_size()
            cu = cu_seqlens[:num_samples + 1] // cp
            if w.numel() < num_samples:  # missing entries default to K=1 (weight 1.0)
                w = torch.cat([w, w.new_ones(num_samples - w.numel())])
            seg_len = cu[1:] - cu[:-1]                                   # [num_samples]
            tok_w = torch.repeat_interleave(w[:num_samples], seg_len)    # [cu[-1]]
            scale = lbs.new_ones(lbs.shape, dtype=torch.float32).reshape(-1)
            # SP-alignment padding tail (positions past cu[-1]) stays at 1.0; its
            # labels are -100 so the value is irrelevant anyway.
            scale[:tok_w.numel()] = tok_w
            scale = scale.view(lbs.shape)                                # [1, total]
        else:
            b, s = lbs.shape
            if w.numel() < b:
                w = torch.cat([w, w.new_ones(b - w.numel())])
            scale = w[:b].view(b, 1).expand(b, s)                        # [b, s]

        structural = torch.isin(lbs, self._structural_token_ids(lbs.device))
        scale = torch.where(structural, scale.new_ones(()), scale)

        if os.environ.get('BT_SFT_DEBUG') and not self._reasoning_dbg_logged \
                and parallel_state.get_data_parallel_rank() == 0:
            self._reasoning_dbg_logged = True  # one-time: the .item()s below force a sync
            sup = lbs != -100
            down = sup & ~structural & (scale < 1.0)
            logger.info(f'[reasoning_ce_dbg] K={num_cg_labels.reshape(-1).tolist()}  '
                        f'supervised={int(sup.sum())}  structural={int((sup & structural).sum())}  '
                        f'downweighted={int(down.sum())}  '
                        f'mean_scale_on_supervised={float(scale[sup].mean()):.4f}')
        return scale

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params=None,
        packed_seq_params=None,
        num_cg_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output = super().forward(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **kwargs,
        )
        # labels is None on non-last PP stages and at inference -- output is then
        # hidden states / logits, not a per-token loss, and must not be scaled.
        if labels is None or num_cg_labels is None or not self._reasoning_ce_1_over_k:
            return output
        scale = self._reasoning_ce_scale(labels, num_cg_labels, packed_seq_params)
        if isinstance(output, tuple):
            lm_output, reg_logits = output
            return lm_output * scale.to(lm_output.dtype).view_as(lm_output), reg_logits
        return output * scale.to(output.dtype).view_as(output)


class Qwen3VLAIPReasoningLoader(Qwen3VLLoader):
    model_cls = Qwen3VLAIPReasoningModel


def make_aip_reasoning_loader(base_loader_cls):
    """Wrap an arbitrary registered mcore loader so it builds the reasoning AIP model
    (same mechanics as ``qwen3_vl_aip.make_aip_loader``)."""
    if getattr(base_loader_cls, 'model_cls', None) is Qwen3VLAIPReasoningModel:
        return base_loader_cls
    return type(f'{base_loader_cls.__name__}AIPReasoning', (base_loader_cls, ),
                {'model_cls': Qwen3VLAIPReasoningModel})
