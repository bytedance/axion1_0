# Copyright (c) ModelScope Contributors. All rights reserved.
"""Gemma4 AIP / BT-SFT extension.

Re-bases the qwen ``Qwen3VLAIPModel`` AIP-head / BT-SFT logic onto gemma4's own
``Gemma4GPTModel`` purely through Python MRO, so gemma4 KEEPS its family-specific
``language_model_cls`` (``Gemma4TextGPTModel`` -- dual sliding/full rotary, final-logit
softcapping, MoE block, KV-sharing) while reusing the entire AIP/BT-SFT implementation.
``qwen3_vl_aip.py`` is intentionally left untouched.

Why MRO composition works (verified):
    ``class Gemma4AIPModel(Qwen3VLAIPModel, Gemma4GPTModel)`` linearizes to
    ``Gemma4AIPModel -> Qwen3VLAIPModel -> Gemma4GPTModel -> MultimodalGPTModel -> ...``
    (both bases share ``MultimodalGPTModel``). So:
      * ``language_model_cls`` resolves to ``Gemma4TextGPTModel`` (from Gemma4GPTModel,
        which precedes MultimodalGPTModel in the MRO), NOT the plain ``GPTModel``;
      * ``super().__init__`` / ``super().forward`` inside ``Qwen3VLAIPModel`` dispatch to
        ``Gemma4GPTModel`` -- i.e. the head is built on top of gemma4's real model.

The AIP regression head and the BT-SFT verbalized scoring both read the decoder's
``final_layernorm`` output via a forward hook (pre-softcap) and do their own LM-head
matmul, so no gemma4-specific pooling is required.

Softcapping (BT-SFT): gemma4 applies final-logit softcapping (``final_logit_softcapping=30``)
in ``Gemma4TextGPTModel._forward_output_layer``. For MEGATRON TRAINING we disable it by
default (``Gemma4AIPModel.__init__`` nulls the cap; ``GEMMA4_TRAIN_SOFTCAP=1`` re-enables),
because the BT-SFT score/loss paths already read raw pre-softcap logits and the tanh only
adds full-vocab activation memory. vLLM INFERENCE still softcaps (it reads the HF config, not
this runtime attribute); the verbalized ``<score_k>`` expected score is thus on a slightly
different scale at train vs. infer (softcap is monotonic -> argmax bucket preserved; only the
soft expectation is compressed). The AIP regression-head path never touches the LM head. If
exact train/infer parity is needed, keep softcapping on and override ``_bt_sft_score`` to
apply ``tanh(logits/cap)*cap`` before ``expected_score_from_logits``.
"""
import os
import re

from mcore_bridge.model.mm_gpts.gemma4 import Gemma4GPTModel

from swift.utils import get_logger

from .qwen3_vl_aip import Qwen3VLAIPModel

logger = get_logger()


def _patch_megatron_gemma4_expert_keys() -> None:
    """Work around a megatron-core fsdp_dtensor checkpoint bug for gemma4 MoE.

    gemma4 names its MoE module ``experts_mlp`` (each decoder layer has a dense ``mlp``
    plus a routed ``experts_mlp``). megatron-core's expert-key parser requires a literal
    ``.mlp.experts`` (regex ``^.*\\.mlp\\.experts\\.linear_fc\\d\\.weight(\\d+)``), which
    fails to match ``experts_mlp.experts...`` even though the loose substring guard passes
    -- so checkpoint save asserts ``Failed to parse expert index from key``. Relax the
    parser to match the expert suffix regardless of the ``mlp`` / ``experts_mlp`` prefix;
    the downstream re-index helpers are prefix-agnostic (substring guard + ``weight{idx}``
    replace), so fixing this single parser is sufficient. This is a dependency bug
    affecting base gemma4 MoE too, not AIP-specific.
    """
    try:
        import megatron.core.transformer.fsdp_dtensor_checkpoint as _m
    except Exception:
        return
    if getattr(_m, '_gemma4_expert_key_patched', False):
        return

    def get_expert_index_from_key(key):
        if 'experts.linear_fc1.weight' in key or 'experts.linear_fc2.weight' in key:
            m = re.search(r'experts\.linear_fc\d\.weight(\d+)', key)
            assert m, f'Failed to parse expert index from key: {key}'
            return int(m.group(1))
        elif 'experts.local_experts' in key:
            m = re.search(r'experts\.local_experts\.(\d+)', key)
            assert m, f'Failed to parse expert index from key: {key}'
            return int(m.group(1))
        return None

    _m.get_expert_index_from_key = get_expert_index_from_key
    _m._gemma4_expert_key_patched = True
    logger.info('[gemma4_aip] patched megatron fsdp_dtensor get_expert_index_from_key for experts_mlp keys.')


_patch_megatron_gemma4_expert_keys()


class Gemma4AIPModel(Qwen3VLAIPModel, Gemma4GPTModel):
    """``Gemma4GPTModel`` + the AIP regression head / BT-SFT verbalized scoring.

    All head/scoring logic is inherited from :class:`Qwen3VLAIPModel`; the MRO puts
    ``Gemma4GPTModel`` immediately after it so the model keeps ``Gemma4TextGPTModel``
    (see module docstring).

    Training-time softcap disable: by default we drop gemma4's final-logit softcapping
    (``30*tanh(logits/30)``) for MEGATRON TRAINING by nulling ``final_logit_softcapping``
    on the inner ``Gemma4TextGPTModel`` (its ``_forward_output_layer`` then skips the tanh).
    Why it is safe + why it helps:
      * The BT-SFT sparse LM-head path already bypasses ``_forward_output_layer`` and trains
        on raw logits, and ``_bt_sft_score`` reads pre-softcap logits -- so the score path is
        already softcap-free. Nulling the cap only affects the DENSE LM-CE fallback, making
        the two paths consistent.
      * Softcapping materialises 3 extra full-vocab ``[s, b, 262144]`` tensors (``/cap``,
        ``tanh``, ``*cap``), all retained for backward -- a large activation cost at gemma4's
        262k vocab. Skipping it frees that memory whenever the dense path runs.
    Export / vLLM inference are UNAFFECTED: they read ``final_logit_softcapping`` from the HF
    config, not this runtime attribute. Set ``GEMMA4_TRAIN_SOFTCAP=1`` to keep softcapping on
    during training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('GEMMA4_TRAIN_SOFTCAP', '0') == '1':
            return
        disabled = 0
        for module in self.modules():
            if getattr(module, 'final_logit_softcapping', None) is not None:
                module.final_logit_softcapping = None
                disabled += 1
        if disabled:
            logger.info(
                f'[gemma4_aip] disabled final_logit_softcapping on {disabled} module(s) for '
                f'training (frees full-vocab tanh activations in the dense LM-CE path; the BT-SFT '
                f'sparse path already skips it). Export/vLLM inference still softcap via the HF '
                f'config. Set GEMMA4_TRAIN_SOFTCAP=1 to keep it.')


def make_gemma4_aip_loader(base_loader_cls):
    """Subclass gemma4's registered mcore loader, overriding only ``model_cls`` with
    :class:`Gemma4AIPModel`.

    Subclassing (rather than replacing) the loader preserves gemma4's own
    ``build_model`` / layer-spec wiring (sliding vs full attention specs, MoE) and only
    swaps in the AIP model class. Mirrors ``qwen3_vl_aip.make_aip_loader`` but pins the
    gemma4 model class, so ``qwen3_vl_aip.py`` stays untouched.
    """
    if getattr(base_loader_cls, 'model_cls', None) is Gemma4AIPModel:
        return base_loader_cls
    return type(f'{base_loader_cls.__name__}AIP', (base_loader_cls, ), {'model_cls': Gemma4AIPModel})
