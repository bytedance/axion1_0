# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fused-kernel forward for Gemma4 (``*ForConditionalGeneration``).

Gemma4 has a 262k vocabulary; materializing the ``[seq_len, vocab]`` logits tensor
(required by the default forward + ``entropy_from_logits``) OOMs on long multimodal
sequences (e.g. 128 video frames -> ~41k prompt tokens). This module provides a fused
forward that computes per-token log-probs (and entropy) chunk-by-chunk via
``FusedLinearForPPO`` without ever materializing the full logits tensor, in either the
forward or the backward pass.

Two Gemma4-specific details vs. the generic ``dense_common`` fused forward:
  * ``self.model`` (a ``Gemma4Model``) accepts ``pixel_values`` directly and merges the
    vision features internally, so we thread the multimodal kwargs straight through
    instead of dropping them (the generic path drops them -> text-only -> broken VLM).
  * Gemma4 applies ``final_logit_softcapping`` (30.0) to the logits before the softmax.
    ``FusedLinearForPPO`` replicates it inside the chunked kernel so the fused log-probs
    match the non-fused model forward.
"""

from typing import Optional

import torch

from verl.models.transformers.dense_common import CausalLMOutputForPPO


def _gemma4_run_model(self, input_ids, temperature, shift_labels, labels, kwargs):
    """Run the inner Gemma4Model (vision merged internally) and return hidden states + rolled labels."""
    # `temperature`/`shift_labels` are consumed here; `return_dict` is forced True. Everything
    # else in kwargs (attention_mask, position_ids, pixel_values, mm_token_type_ids, ...) is
    # passed through to Gemma4Model, exactly as Gemma4ForConditionalGeneration.forward does.
    kwargs.pop("return_dict", None)
    kwargs.pop("logits_to_keep", None)
    outputs = self.model(input_ids=input_ids, return_dict=True, **kwargs)
    hidden_states = outputs.last_hidden_state

    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("gemma4 fused forward requires either labels or input_ids.")
    return outputs, hidden_states, rolled_labels


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **kwargs,
):
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs, hidden_states, rolled_labels = _gemma4_run_model(
        self, input_ids, temperature, shift_labels, labels, kwargs
    )

    softcap = self.config.get_text_config().final_logit_softcapping
    # Larger chunk than the 512 default: fewer Python-level chunk iterations over the 262k
    # vocab keeps the per-rank lm_head step fast (reduces cross-rank load imbalance that can
    # trip the NCCL watchdog on long batches). ~2048*262k*4B ≈ 2GB/chunk, well within headroom.
    fused_linear_for_ppo = FusedLinearForPPO(chunk_size=2048)
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
        final_logit_softcapping=softcap,
    )
    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=getattr(outputs, "hidden_states", None),
    )


# Gemma4's final_logit_softcapping is not supported by the Triton linear-cross-entropy
# kernel, so the Triton backend falls back to the (softcap-correct) torch backend.
forward_with_triton_backend = forward_with_torch_backend
