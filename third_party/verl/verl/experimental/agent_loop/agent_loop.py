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
"""
Agent framework for multi-turn rollout and agentic reinforcement learning.
- AgentLoopBase: coroutine based abstract base class for agent loop.
  - SingleTurnAgentLoop: single turn agent loop.
  - ToolAgentLoop: ReAct agent loop with tool calling, with user defined tools.
- AgentLoopWorker: worker class for running agent loop coroutines in parallel.
- AgentLoopManager: manager class for running agent loop workers in parallel.

AgentLoopManager is one specific agent-framework implementation in verl,
and is designed to be fully replaceable by other agent frameworks such as:
- NVIDIA Nemo-Gym
- AWS Bedrock AgentCore
- SWE-agent
- ...
"""

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.utils.phase_trace import mark as _pt_mark
from verl.utils import mm_ref
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.tools.tool_registry import load_all_tools
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import auto_await, get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
)
from verl.utils.skip import SkipManager
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
    normalize_token_ids,
)
from verl.utils.tokenizer.chat_template import apply_chat_template, initialize_system_prompt, initialize_turn_separator
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder
from verl.workers.config import (
    HFModelConfig,
    RolloutConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


def _expand_non_tensor_batch_for_agent_loop_n(
    non_tensor_batch: dict[str, Any],
    *,
    agent_loop_n: int,
) -> dict[str, Any]:
    """Interleave-repeat non_tensor_batch to match flattened agent-loop outputs."""
    if agent_loop_n <= 1:
        return non_tensor_batch
    expanded: dict[str, Any] = {}
    for key, val in non_tensor_batch.items():
        if key == "__do_sample__":
            continue
        expanded[key] = np.repeat(val, agent_loop_n, axis=0)
    return expanded


def _flatten_agent_loop_outputs(outputs: list, *, agent_loop_n: int, batch_size: int) -> list:
    flat: list = []
    for out in outputs:
        if isinstance(out, list):
            if len(out) != agent_loop_n:
                raise ValueError(
                    f"agent_loop_n={agent_loop_n} but agent returned {len(out)} outputs"
                )
            flat.extend(out)
        else:
            if agent_loop_n != 1:
                raise ValueError(
                    f"agent_loop_n={agent_loop_n} but agent returned a single output; "
                    "only single_turn_agent supports agent_loop_n>1"
                )
            flat.append(out)
    if len(flat) != batch_size * agent_loop_n:
        raise ValueError(
            f"expected {batch_size * agent_loop_n} trajectories, got {len(flat)}"
        )
    return flat


# Pixel tensors dominate Ray/pickle payload; vision forward already casts to
# model.visual.dtype (bf16 under typical training). Compress for transport only.
_TRANSPORT_BF16_KEYS = frozenset({"pixel_values", "pixel_values_videos"})


def cast_multi_modal_inputs_for_transport(multi_modal_inputs: dict[str, Any]) -> dict[str, Any]:
    """Cast bulky float pixel tensors to bf16; leave grid/index tensors unchanged."""
    if not multi_modal_inputs:
        return multi_modal_inputs
    out: dict[str, Any] = {}
    for key, value in multi_modal_inputs.items():
        if (
            key in _TRANSPORT_BF16_KEYS
            and isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dtype != torch.bfloat16
        ):
            out[key] = value.to(dtype=torch.bfloat16)
        else:
            out[key] = value
    return out


def _stamp_gen_metrics(output, wall: float, preprocess: float, postprocess: float) -> None:
    try:
        output.metrics.wall = float(wall)
        output.metrics.preprocess = float(preprocess)
        output.metrics.postprocess = float(postprocess)
    except Exception:  # metrics may be a plain dict for custom loops
        pass


def _env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in {"", "0", "false", "no"}


_MM_SOURCE_TYPES = ("image", "video", "audio")


def mm_source_key(messages: list[dict]) -> tuple | None:
    """Stable key for the multimodal sources referenced by a prompt.

    Only string sources (paths / urls) are keyable; in-memory objects (PIL images,
    tensors) return ``None`` so they are never shared across prompts by mistake.
    """
    parts = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            mm_type = item.get("type")
            if mm_type not in _MM_SOURCE_TYPES:
                continue
            src = item.get(mm_type, item.get(f"{mm_type}_url"))
            if not isinstance(src, str):
                return None
            extra = tuple(
                sorted(
                    (k, v)
                    for k, v in item.items()
                    if k not in {mm_type, f"{mm_type}_url", "type"} and isinstance(v, (str, int, float, bool))
                )
            )
            parts.append((mm_type, src, extra))
    return tuple(parts) if parts else None


class MultiModalStepCache:
    """Per-AgentLoopWorker, per-generate_sequences cache of multimodal work.

    GRPO sends the ``n`` rollouts of one prompt to the same worker, and a multi-turn
    agent additionally re-sends the same video to the inference server on every
    turn. Keyed by :func:`mm_source_key`, this cache lets those rollouts share:

    * ``data``   - decoded frames / images (``process_multi_modal_info`` runs once);
    * ``refs``   - one ``ray.put`` ObjectRef per modality for generate RPCs;
    * ``inputs`` - the HF processor output (``pixel_values*`` / ``*_grid_thw``), so
      the trajectories of a prompt carry the *same* tensor objects and pickle
      memoizes them once per RPC downstream.

    Entries are only reused when the trajectory's multimodal data is identical to
    the cached prompt data (element identity check), so tool-added images fall
    back to the uncached path. Cleared at the end of every ``generate_sequences``.
    """

    def __init__(self):
        self.data: dict[tuple, asyncio.Future] = {}
        self.refs: dict[tuple, dict[str, Any]] = {}
        self.inputs: dict[Any, dict] = {}
        self.hits = 0
        self.misses = 0

    async def get_or_create(self, key: tuple, factory):
        fut = self.data.get(key)
        if fut is not None:
            self.hits += 1
            return await fut
        self.misses += 1
        fut = asyncio.get_running_loop().create_future()
        self.data[key] = fut
        try:
            result = await factory()
        except BaseException as e:  # propagate to waiters, drop the entry
            self.data.pop(key, None)
            if not fut.done():
                fut.set_exception(e)
            raise
        fut.set_result(result)
        return result

    def canonical(self, key: tuple) -> dict | None:
        fut = self.data.get(key)
        if fut is None or not fut.done() or fut.exception() is not None:
            return None
        return fut.result()

    def matches(self, key: tuple | None, multi_modal_data: dict | None) -> bool:
        """True iff ``multi_modal_data`` holds exactly the cached objects for ``key``."""
        if key is None or not multi_modal_data:
            return False
        canonical = self.canonical(key)
        if canonical is None or set(canonical) != set(multi_modal_data):
            return False
        for mod, cached in canonical.items():
            cur = multi_modal_data[mod]
            if isinstance(cached, list):
                if not isinstance(cur, list) or len(cur) != len(cached):
                    return False
                if any(a is not b for a, b in zip(cur, cached, strict=True)):
                    return False
            elif cur is not cached:
                return False
        return True

    def generate_refs(self, key: tuple | None, multi_modal_data: dict | None) -> dict[str, Any] | None:
        """ObjectRefs for generate RPCs, created once per key; None if not shareable."""
        if not self.matches(key, multi_modal_data):
            return None
        refs = self.refs.get(key)
        if refs is None:
            refs = {mod: ray.put(val) for mod, val in self.canonical(key).items()}
            self.refs[key] = refs
        return refs

    def clear(self):
        self.data.clear()
        self.refs.clear()
        self.inputs.clear()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.lower() not in {"", "0", "false", "no"}


def _preview_text(value: str, max_chars: int = 400) -> str:
    env_max_chars = os.getenv("VERL_DEBUG_PREVIEW_MAX_CHARS")
    if env_max_chars is not None:
        max_chars = int(env_max_chars)
    value = value.replace("\n", "\\n")
    if max_chars <= 0:
        return value
    return value if len(value) <= max_chars else value[:max_chars] + "...<truncated>"


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""
    # gen-phase breakdown outside model generation (seconds)
    preprocess: float = 0.0   # process_multi_modal_info: video decode / frame fetch
    postprocess: float = 0.0  # HF processor + padding + score in _agent_loop_postprocess
    wall: float = 0.0         # agent_loop.run wall time (generate + tools + preprocess)

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    """Processor/backend kwargs that must stay aligned across rollout and training paths."""

    def as_dict(self) -> dict[str, Any]:
        """Convert agent loop output to a dictionary."""
        output = self.model_dump(exclude_unset=True)

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        response_logprobs = output.pop("response_logprobs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        # rm_scores: reward score for each token
        reward_score = output.pop("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        teacher_ids, teacher_logprobs = (
            output["extra_fields"].pop("teacher_ids", None),
            output["extra_fields"].pop("teacher_logprobs", None),
        )
        if teacher_ids is not None:
            output["teacher_ids"] = teacher_ids
        if teacher_logprobs is not None:
            output["teacher_logprobs"] = teacher_logprobs
        return output


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    teacher_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities from teacher model for prompt/response tokens."""
    teacher_ids: Optional[torch.Tensor] = None
    """Padded token ids corresponding to the teacher log probabilities."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class ToolListWrap:
    """Wraps a tool list so ``hydra.utils.instantiate`` doesn't recursively
    resolve its elements (which would demote them to ``DictConfig``)."""

    def __init__(self, tools: list):
        self.tools = tools


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments.

    Args:
        trainer_config (DictConfig): whole config for main entrypoint.
        server_manager (LLMServerClient): OpenAI compatible LLM server manager.
        tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        processor (AutoProcessor): Processor for process messages.
        dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
        data_config (DictConfigWrap): Dataset config.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: LLMServerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        self.config = trainer_config.config
        self.rollout_config = self.config.actor_rollout_ref.rollout
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config.config
        self.apply_chat_template_kwargs = dict(self.data_config.get("apply_chat_template_kwargs", {}) or {})
        # VERL_CHAT_TEMPLATE_FILE: render every rollout prompt/delta with this jinja template
        # instead of the checkpoint's own — rollout/serving prompt parity (e.g. the community
        # enhanced template whose tool-instruction block measurably raises tool appetite).
        # Threads through apply_chat_template_kwargs so ALL render sites (system prompt, turn
        # separator, tool deltas, CT builder) use it uniformly.
        _tpl_file = os.environ.get("VERL_CHAT_TEMPLATE_FILE")
        if _tpl_file and "chat_template" not in self.apply_chat_template_kwargs:
            with open(_tpl_file) as _f:
                self.apply_chat_template_kwargs["chat_template"] = _f.read()
        self.mm_processor_kwargs = self.data_config.get("mm_processor_kwargs", {})
        # Per-step multimodal cache owned by AgentLoopWorker (None when disabled).
        self.mm_cache: MultiModalStepCache | None = kwargs.get("mm_cache")
        self.mm_source_key: tuple | None = None
        self.continuous_token_builder = None
        self.enable_continuous_token = False
        self.continuous_token_multimodal = False
        continuous_token_config = self.data_config.continuous_token
        if continuous_token_config.enable:
            model_config = self.config.actor_rollout_ref.model
            try:
                self.continuous_token_builder = create_continuous_token_builder(
                    self.tokenizer,
                    model_family=continuous_token_config.model_family,
                    model_path=model_config.path,
                    tokenizer_name_or_path=model_config.tokenizer_path,
                    chat_template_kwargs=self.apply_chat_template_kwargs,
                    # None for text-only models; multimodal models get a builder
                    # that processor-expands media placeholders, or a ValueError
                    # when their family has none (legacy fallback below).
                    processor=self.processor,
                )
                self.enable_continuous_token = True
                self.continuous_token_multimodal = self.processor is not None
            except ValueError as e:
                logger.warning("Continuous Token disabled (%s); falling back to the legacy path.", e)
        if self.enable_continuous_token:
            # Continuous Token doesn't use the legacy removable system prompt.
            self.system_prompt = None
            # Continuous Token re-renders non-assistant turns from the full message list, so it does
            # not need the incremental turn separator.
            self.turn_separator = []
        else:
            processing_class = self.processor if self.processor is not None else self.tokenizer
            self.system_prompt = initialize_system_prompt(processing_class, **self.apply_chat_template_kwargs)
            # Turn separator dropped when the model stops at the assistant close token; restored at
            # turn boundaries in ``ToolAgentLoop._handle_processing_tools_state``.
            self.turn_separator = initialize_turn_separator(processing_class, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Backward-compatible wrapper for multi-modal extraction."""
        return await self.process_multi_modal_info(messages)

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        t0 = time.perf_counter()
        try:
            return await self._process_multi_modal_info_maybe_cached(messages)
        finally:
            self.preprocess_seconds = getattr(self, "preprocess_seconds", 0.0) + (time.perf_counter() - t0)

    async def _process_multi_modal_info_maybe_cached(self, messages: list[dict]) -> dict:
        if self.processor is not None and self.mm_cache is not None:
            key = mm_source_key(messages)
            if key is None:
                # Why the per-step mm cache never engages on this workload (2026-09-18): the
                # skip counters said mm_source_key was None on every call, and the shape of the
                # message content is the only thing that can cause it.
                _shapes = []
                for _m in list(messages)[:3]:
                    _c = _m.get("content") if isinstance(_m, dict) else None
                    _types = [
                        (i.get("type"), type(i.get(i.get("type"), "")).__name__)
                        for i in (_c if isinstance(_c, (list, tuple)) else [])
                        if isinstance(i, dict)
                    ]
                    _shapes.append((type(_m).__name__, type(_c).__name__, _types[:3]))
                _pt_mark("agentloop/mm_key_none", shapes=repr(_shapes)[:300])
            if key is not None:
                self.mm_source_key = key
                data = await self.mm_cache.get_or_create(
                    key, lambda: self._process_multi_modal_info_uncached(messages)
                )
                # Shallow-copy lists: agent loops may append tool images in place.
                return {mod: (list(val) if isinstance(val, list) else val) for mod, val in data.items()}
        return await self._process_multi_modal_info_uncached(messages)

    async def _process_multi_modal_info_uncached(self, messages: list[dict]) -> dict:
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    def mm_generate_payload(self, images, videos, audios) -> tuple[Any, Any, Any]:
        """Multimodal args for ``server_manager.generate``.

        When the data is the cached prompt data, return per-modality ObjectRefs
        (one ``ray.put`` per prompt per step) so the n rollouts x turns generate
        RPCs do not each re-serialize the frames. Otherwise pass concrete objects.
        """
        if self.mm_cache is None:
            return images, videos, audios
        current = {}
        if images is not None:
            current["images"] = images
        if videos is not None:
            current["videos"] = videos
        if audios is not None:
            current["audios"] = audios
        refs = self.mm_cache.generate_refs(self.mm_source_key, current)
        if refs is None:
            return images, videos, audios
        # Images stay concrete: the LLM client may inspect them (block-collapse path); videos
        # and audios dominate payload size and are passed straight through to the server.
        video_payload = refs.get("videos", videos)
        if video_payload is not videos:
            # By-reference transport: the server resolves this once instead of Ray
            # deserializing the clip again on every turn (verl/utils/mm_ref.py).
            video_payload = mm_ref.wrap(video_payload, n_items=len(videos) if videos else 0)
        return images, video_payload, refs.get("audios", audios)

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list = None,
        videos: list = None,
        audios: list = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[int]:
        """Build the initial prompt token ids with Continuous Token."""
        has_mm = images is not None or videos is not None or audios is not None
        if has_mm and not self.continuous_token_multimodal:
            raise ValueError(
                "Continuous Token received multimodal inputs but the active builder is text-only."
            )
        if self.continuous_token_multimodal:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.continuous_token_builder.build_initial_tokens(
                    messages,
                    tools=tools,
                    images=images,
                    videos=videos,
                    audios=audios,
                    mm_processor_kwargs=mm_processor_kwargs
                    if mm_processor_kwargs is not None
                    else self._get_mm_processor_kwargs(audios),
                ),
            )
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.continuous_token_builder.build_initial_tokens(messages, tools=tools),
            )
        if has_mm:
            # Mirror the legacy multimodal path: placeholder tokens must stay 1:1
            # with multi_modal_inputs features, so an over-long prompt cannot be
            # sliced — fail loudly instead.
            prompt_length = self.rollout_config.prompt_length
            if len(prompt_ids) > prompt_length:
                raise ValueError(
                    f"Multimodal prompt of {len(prompt_ids)} tokens exceeds "
                    f"rollout.prompt_length={prompt_length} and cannot be truncated."
                )
            return prompt_ids
        return self._cap_text_prompt_length(prompt_ids)

    async def ct_merge_non_assistant_msg(
        self,
        previous_messages: list[dict],
        updated_messages: list[dict],
        runtime_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        tools: list[dict] = None,
        images: list = None,
        videos: list = None,
        audios: list = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Merge appended non-assistant messages into runtime tokens and metadata."""
        mm_kwargs: dict[str, Any] = {}
        if images is not None or videos is not None or audios is not None:
            if not self.continuous_token_multimodal:
                raise ValueError(
                    "Continuous Token received multimodal tool outputs but the active builder is text-only."
                )
            mm_kwargs = {
                "images": images,
                "videos": videos,
                "audios": audios,
                "mm_processor_kwargs": mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            }
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_non_assistant_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
                **mm_kwargs,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result, response_mask, response_logprobs
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    async def ct_merge_assistant_token(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        assistant_logprobs: Optional[list[float]] = None,
    ):
        """Merge assistant-generated tokens and align response metadata."""
        merge_result = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.merge_assistant_tokens(
                runtime_token_ids,
                assistant_token_ids,
            ),
        )
        aligned_response_mask, aligned_response_logprobs = self.continuous_token_builder.align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )
        return merge_result, aligned_response_mask, aligned_response_logprobs

    def _cap_text_prompt_length(self, prompt_ids: list[int]) -> list[int]:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            logger.warning(
                "Prompt of %d tokens exceeds rollout.prompt_length=%d; left-truncating.",
                len(prompt_ids),
                prompt_length,
            )
            return prompt_ids[-prompt_length:]
        return prompt_ids

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        audios: list[Any] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                videos=videos,
                audio=audios,
                mm_processor_kwargs=mm_processor_kwargs
                if mm_processor_kwargs is not None
                else self._get_mm_processor_kwargs(audios),
            )
            prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt and self.system_prompt:
            # Strip ONLY when the render actually begins with the measured system prefix. The
            # blind slice assumed every render carries it, but tool-message deltas go through
            # apply_chat_template's dummy-user fallback, which already removed the prefix —
            # a template with a nonzero default system (stock qwen3.5 measures 42 tokens) then
            # had 42 tokens cut from the OBSERVATION itself, eating <|image_pad|> runs and
            # desynchronizing pad tokens from image grids (proven offline: 360 pads vs 396
            # grid-tokens on a 2-image observation).
            n = len(self.system_prompt)
            if prompt_ids[:n] == self.system_prompt:
                prompt_ids = prompt_ids[n:]

        # Mirror the response-side ``response_ids[:response_length]`` cap on the prompt side:
        # every prompt produced by the agent loop must fit in ``rollout.prompt_length`` so that
        # ``_pad_token_ids`` (and downstream ``torch.cat``) can rely on uniform shapes.
        # Multimodal prompts cannot be sliced here because placeholder tokens must remain
        # aligned 1:1 with ``multi_modal_inputs`` features, so we fail loudly instead.
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            if images or videos or audios:
                raise ValueError(
                    f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                    f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                    f"sequences corrupts vision/audio feature alignment, so this is treated "
                    f"as a configuration error. Reduce the multimodal input size "
                    f"(e.g. ``total_pixels`` / ``max_pixels`` / fps / number of frames) or "
                    f"increase ``rollout.prompt_length``."
                )
            prompt_ids = self._cap_text_prompt_length(prompt_ids)

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        rollout_config, model_config = config.actor_rollout_ref.rollout, config.actor_rollout_ref.model
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)

        self.dataset_cls = get_dataset_class(config.data)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor
        self.mm_processor_kwargs = config.data.get("mm_processor_kwargs", {})

        # Online policy distillation
        self.distillation_enabled = is_distillation_enabled(config.distillation)
        if self.distillation_enabled:
            from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager

            self.teacher_key: str = config.distillation.teacher_key
            self.teacher_server_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )

        # Load tools once per worker; each trajectory just reuses self.tools.
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        function_tool_path = self.rollout_config.multi_turn.function_tool_path
        self.tools = load_all_tools(
            tool_config_path=resolve_config_path(tool_config_path) if tool_config_path else None,
            function_tool_path=resolve_config_path(function_tool_path) if function_tool_path else None,
        )

        # Load custom agent loop implementations from config path
        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

        trace_config = self.rollout_config.trace
        RolloutTraceConfig.init(
            self.rollout_config.trace.project_name,
            self.rollout_config.trace.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    def _get_mm_processor_kwargs(self, audio_data: Optional[list[Any]] = None) -> dict[str, Any]:
        """Return multimodal processor kwargs with audio sampling-rate defaults."""
        mm_processor_kwargs = dict(self.mm_processor_kwargs or {})
        if audio_data is not None and "sampling_rate" not in mm_processor_kwargs:
            sampling_rate = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                mm_processor_kwargs["sampling_rate"] = int(sampling_rate)
        return mm_processor_kwargs

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.rollout_config
        validate = batch.meta_info.get("validate", False)
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
            params["top_p"] = 1.0
            params["top_k"] = -1
            params["temperature"] = 0

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        agent_loop_n = int(batch.meta_info.get("agent_loop_n", 1))

        # Per-step multimodal cache shared by every trajectory this worker runs (see MultiModalStepCache).
        self._mm_step_cache = MultiModalStepCache() if _env_flag("VERL_AGENT_LOOP_MM_CACHE", "1") else None

        # NOTE: __do_sample__ is an internal per-sample override used by REMAX combined rollout.
        # Do not forward it to concrete agent loops, which may reject unknown kwargs.
        per_sample_do_sample = batch.non_tensor_batch.get("__do_sample__")
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            kwargs["agent_loop_n"] = agent_loop_n
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        try:
            outputs = await asyncio.gather(*tasks)
        finally:
            if self._mm_step_cache is not None:
                logger.info(
                    "agent_loop mm cache: %d hits, %d misses, %d processor outputs",
                    self._mm_step_cache.hits,
                    self._mm_step_cache.misses,
                    len(self._mm_step_cache.inputs),
                )
                self._mm_step_cache.clear()
                self._mm_step_cache = None
        outputs = _flatten_agent_loop_outputs(outputs, agent_loop_n=agent_loop_n, batch_size=len(batch))

        input_non_tensor_batch = batch.non_tensor_batch
        if agent_loop_n > 1:
            input_non_tensor_batch = _expand_non_tensor_batch_for_agent_loop_n(
                batch.non_tensor_batch, agent_loop_n=agent_loop_n
            )

        output = self._postprocess(
            outputs, input_non_tensor_batch=input_non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput | list[_InternalAgentLoopOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
                mm_cache=getattr(self, "_mm_step_cache", None),
            )
            t_run = time.perf_counter()
            output = await agent_loop.run(sampling_params, **kwargs)
            t_run = time.perf_counter() - t_run
            t_pre = getattr(agent_loop, "preprocess_seconds", 0.0)
            step_cache = getattr(self, "_mm_step_cache", None)
            mm_key = agent_loop.mm_source_key if step_cache is not None else None
            if isinstance(output, list):
                cache: dict = step_cache.inputs if step_cache is not None else {}
                results = []
                for rollout_i, one in enumerate(output):
                    with rollout_trace_attr(
                        step=trajectory["step"],
                        sample_index=trajectory["sample_index"],
                        rollout_n=rollout_i,
                        validate=trajectory["validate"],
                        name="agent_loop",
                        trace=trace,
                    ):
                        t_post = time.perf_counter()
                        res = await self._agent_loop_postprocess(
                            one, trajectory["validate"], _mm_inputs_cache=cache, _mm_source_key=mm_key, **kwargs
                        )
                        _stamp_gen_metrics(res, t_run, t_pre, time.perf_counter() - t_post)
                        results.append(res)
                return results
            t_post = time.perf_counter()
            res = await self._agent_loop_postprocess(
                output,
                trajectory["validate"],
                _mm_inputs_cache=step_cache.inputs if step_cache is not None else None,
                _mm_source_key=mm_key,
                **kwargs,
            )
            _stamp_gen_metrics(res, t_run, t_pre, time.perf_counter() - t_post)
            return res

    def _check_mm_token_invariant(self, output, input_ids, multi_modal_inputs) -> None:
        """Per-trajectory invariant: image-pad tokens in the final ids must equal the token
        budget of the image grids (sum t*h*w / merge^2). A violation here is what later kills
        the actor forward ('Image features and image tokens do not match', abl_A1_cliphigh
        step 4); catching it at postprocess captures the WHOLE trajectory — decoded stream,
        image sizes, grids — instead of a rank-level count. Telemetry only: logs + dumps,
        never raises (the forward's own check remains the enforcement point)."""
        try:
            img_id = get_processor_token_id(self.processor, "image") if self.processor is not None else None
            if img_id is None:
                return
            grid = multi_modal_inputs.get("image_grid_thw")
            n_tokens = int((input_ids == img_id).sum().item())
            n_feat = int(sum(t * h * w for t, h, w in grid.tolist()) // 4) if grid is not None else 0
            if n_tokens == n_feat:
                return
            import os as _os
            import time as _time
            images = (output.multi_modal_data or {}).get("images") or []
            sizes = [getattr(im, "size", None) for im in images]
            prov = getattr(self, "_mm_last_provenance", None)
            print(f"[mm-invariant] MISMATCH pads={n_tokens} grid-tokens={n_feat} "
                  f"n_images={len(images)} sizes={sizes} grids={None if grid is None else grid.tolist()} "
                  f"mm-cache-provenance={prov}", flush=True)
            dbg = _os.environ.get("VERL_MROPE_DEBUG_DIR", "/tmp/verl_mrope_debug")
            _os.makedirs(dbg, exist_ok=True)
            path = _os.path.join(dbg, f"mminv_{_os.getpid()}_{int(_time.time() * 1000)}.pt")
            torch.save({
                "input_ids": input_ids.cpu(),
                "image_grid_thw": grid.cpu() if grid is not None else None,
                "image_sizes": sizes,
                "n_pad_tokens": n_tokens,
                "n_grid_tokens": n_feat,
                "decoded": self.tokenizer.decode(input_ids[0], skip_special_tokens=False)[-20000:],
                "num_turns": getattr(output, "num_turns", None),
                "response_len": len(getattr(output, "response_ids", []) or []),
                "prompt_len": len(getattr(output, "prompt_ids", []) or []),
            }, path)
            print(f"[mm-invariant] specimen dumped: {path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[mm-invariant] check failed: {e!r}", flush=True)

    def _pad_token_ids(
        self,
        tokens: list[int],
        *,
        max_length: int,
        padding_side: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        """Right/left pad a flat list of token ids to a ``(1, max_length)`` tensor."""
        # tokenizer.pad() with empty input returns dict with list values
        # instead of tensors, which breaks downstream .dim() calls.
        if not tokens:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            result = {"input_ids": torch.full((1, max_length), pad_id, dtype=torch.long)}
            if return_attention_mask:
                result["attention_mask"] = torch.zeros((1, max_length), dtype=torch.long)
            return result
        self.tokenizer.padding_side = padding_side
        padded = self.tokenizer.pad(
            {"input_ids": tokens},
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=return_attention_mask,
        )
        if padded["input_ids"].dim() == 1:
            padded["input_ids"] = padded["input_ids"].unsqueeze(0)
            if return_attention_mask:
                padded["attention_mask"] = padded["attention_mask"].unsqueeze(0)
        return padded

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        _mm_inputs_cache = kwargs.pop("_mm_inputs_cache", None)
        _mm_source_key = kwargs.pop("_mm_source_key", None)
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # deprecated vLLM SPMD rollout implementation.
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # input_ids: concatenation of prompt + response
        # Mask:
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [0,0,0,0,1,1,1,1]
        # - response_attention_mask: 0s for padding, 1s for tokens
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # - position_ids: sequential positions for tokens, starting at 0
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

        # TODO(wuxibin): remove padding and use tensordict.
        prompt_output = self._pad_token_ids(
            output.prompt_ids,
            max_length=self.rollout_config.prompt_length,
            padding_side="left",
            return_attention_mask=True,
        )

        response_output = self._pad_token_ids(
            output.response_ids,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=True,
        )

        response_mask_output = self._pad_token_ids(
            output.response_mask,
            max_length=self.rollout_config.response_length,
            padding_side="right",
            return_attention_mask=False,
        )

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            # Calculate start position: left padding means original prompt starts at the end
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            # Add boundary checks for robustness
            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(
            output, input_ids, _mm_inputs_cache=_mm_inputs_cache, _mm_source_key=_mm_source_key
        )
        self._check_mm_token_invariant(output, input_ids, multi_modal_inputs)
        position_ids = self._compute_position_ids(
            input_ids,
            attention_mask,
            multi_modal_inputs,
            output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(
                output.multi_modal_data.get("audios") if output.multi_modal_data else None
            ),
        )
        await self._compute_score(
            [output], kwargs=kwargs, _mm_inputs_cache=_mm_inputs_cache, _mm_source_key=_mm_source_key
        )
        await self._compute_teacher_logprobs(
            output,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )
        teacher_ids, teacher_logprobs = (
            output.extra_fields.pop("teacher_ids", None),
            output.extra_fields.pop("teacher_logprobs", None),
        )
        if teacher_ids is not None and teacher_logprobs is not None:
            # TODO(wuxibin): remove padding and use tensordict.
            from verl.experimental.teacher_loop.teacher_manager import _pad_teacher_outputs

            teacher_ids, teacher_logprobs = _pad_teacher_outputs(
                teacher_ids,
                teacher_logprobs,
                prompt_width=prompt_output["input_ids"].shape[1],
                response_width=response_output["input_ids"].shape[1],
                prompt_length=len(output.prompt_ids),
                response_length=len(output.response_ids),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            mm_processor_kwargs=output.mm_processor_kwargs,
            teacher_logprobs=teacher_logprobs,
            teacher_ids=teacher_ids,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(
        self,
        output,
        input_ids,
        *,
        _mm_inputs_cache: dict | None = None,
        _mm_source_key: tuple | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image, video and audio."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        # Cache ONLY under the verified prompt-source key: the trajectory carries exactly the
        # cached prompt media (no tool-added images), so processor output is identical for all
        # n rollouts of the uid — that sharing is the entire perf win. The former fallback key,
        # id(multi_modal_data), was a use-after-free: pydantic COPIES the dict into
        # _InternalAgentLoopOutput, the original dies when the raw AgentLoopOutput drops after
        # postprocess, CPython reuses the address for a later trajectory's dict, and that
        # trajectory then inherits a DEAD SIBLING'S pixel grids (root cause of the
        # abl_A1_cliphigh step-4 crash: a 2x4-crop trajectory, 8x72=576 pad tokens, served a
        # sibling's 4 full-frame grids, 4x220=880 — the exact forward-crash numbers).
        # Image-bearing trajectories now always compute fresh, uncached.
        step_cache = getattr(self, "_mm_step_cache", None)
        cache_key = None
        if step_cache is not None and step_cache.matches(_mm_source_key, multi_modal_data):
            cache_key = ("src", _mm_source_key)
        if _mm_inputs_cache is not None and cache_key is not None and cache_key in _mm_inputs_cache:
            # provenance for the mm-invariant checker: which cache path produced these inputs
            self._mm_last_provenance = ("hit", "src" if isinstance(cache_key, tuple) else "id")
            return dict(_mm_inputs_cache[cache_key])
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=output.mm_processor_kwargs
            if output.mm_processor_kwargs is not None
            else self._get_mm_processor_kwargs(audios),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        multi_modal_inputs = cast_multi_modal_inputs_for_transport(multi_modal_inputs)
        self._mm_last_provenance = ("computed", "src" if isinstance(cache_key, tuple) else ("id" if cache_key else "none"))
        if _mm_inputs_cache is not None and cache_key is not None:
            _mm_inputs_cache[cache_key] = multi_modal_inputs
            return dict(multi_modal_inputs)
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        # text-only OR non-M-RoPE multimodal (e.g. Gemma4) -> standard 1D positions
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        try:
            vision_position_ids, _ = self.processor.get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **multi_modal_kwargs,
            )
        except StopIteration:
            # Vision segments in the token stream outnumber the grid rows (get_rope_index's
            # iterator underflowed). Capture the evidence for offline diagnosis, then fall
            # back to 1D positions for THIS trajectory so one bad sample cannot kill the run
            # (2026-09-05: first 2-image tool observations to survive rollout died here).
            import os as _os
            import time as _time
            dbg_dir = _os.environ.get("VERL_MROPE_DEBUG_DIR", "/tmp/verl_mrope_debug")
            try:
                _os.makedirs(dbg_dir, exist_ok=True)
                path = _os.path.join(dbg_dir, f"stopiter_{_os.getpid()}_{int(_time.time() * 1000)}.pt")
                torch.save(
                    {
                        "input_ids": input_ids.cpu(),
                        "attention_mask": attention_mask.cpu(),
                        "image_grid_thw": multi_modal_kwargs.get("image_grid_thw"),
                        "video_grid_thw": multi_modal_kwargs.get("video_grid_thw"),
                        "mm_token_type_ids": multi_modal_kwargs.get("mm_token_type_ids"),
                        "mm_processor_kwargs": mm_processor_kwargs,
                        "text_tail": self.tokenizer.decode(
                            input_ids[0][attention_mask[0].bool()][-2048:], skip_special_tokens=False
                        ),
                    },
                    path,
                )
                print(f"[mrope-debug] get_rope_index StopIteration; dumped {path}", flush=True)
            except Exception as dump_err:  # noqa: BLE001
                print(f"[mrope-debug] StopIteration (dump failed: {dump_err!r})", flush=True)
            # Degenerate M-RoPE: text-style positions in all three vision dims, same
            # (1, 4, seq_len) shape as the normal path so batch collation stays uniform.
            flat = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
            return torch.cat((flat.unsqueeze(0), flat.unsqueeze(0).expand(1, 3, -1)), dim=1)
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids

    async def _compute_score(
        self,
        outputs: list[AgentLoopOutput],
        kwargs: dict,
        *,
        _mm_inputs_cache: dict | None = None,
        _mm_source_key: tuple | None = None,
    ) -> None:
        """Compute reward score for all outputs in a trajectory; assigns result to outputs[-1]."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        final_output = outputs[-1]
        if final_output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                all_prompts, all_responses, all_input_ids, all_attention_mask, all_position_ids = [], [], [], [], []
                for output in outputs:
                    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
                    responses = torch.tensor(output.response_ids, dtype=torch.int64)
                    input_ids = torch.cat([prompts, responses], dim=0)
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                    multi_modal_inputs = self._compute_multi_modal_inputs(
                        output, input_ids, _mm_inputs_cache=_mm_inputs_cache, _mm_source_key=_mm_source_key
                    )
                    position_ids = self._compute_position_ids(
                        input_ids.unsqueeze(0),
                        attention_mask.unsqueeze(0),
                        multi_modal_inputs,
                        output.mm_processor_kwargs
                        if output.mm_processor_kwargs is not None
                        else self._get_mm_processor_kwargs(
                            output.multi_modal_data.get("audios") if output.multi_modal_data else None
                        ),
                    ).squeeze(0)
                    all_prompts.append(prompts)
                    all_responses.append(responses)
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                    all_position_ids.append(position_ids)

                n = len(outputs)
                batch = TensorDict(
                    {
                        "prompts": torch.nn.utils.rnn.pad_sequence(all_prompts, batch_first=True, padding_value=0),
                        "responses": torch.nn.utils.rnn.pad_sequence(all_responses, batch_first=True, padding_value=0),
                        "attention_mask": torch.nn.utils.rnn.pad_sequence(
                            all_attention_mask, batch_first=True, padding_value=0
                        ),
                        "input_ids": torch.nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=0),
                        "position_ids": torch.nn.utils.rnn.pad_sequence(
                            all_position_ids, batch_first=True, padding_value=0
                        ),
                    },
                    batch_size=n,
                )
                non_tensor_batch = {
                    **{k: np.array([v] * n) for k, v in kwargs.items()},
                    "__num_turns__": np.array([o.num_turns for o in outputs]),
                    "tool_extra_fields": np.array([o.extra_fields for o in outputs], dtype=object),
                    "prompt_len": np.array([len(o.prompt_ids) for o in outputs]),
                    "response_len": np.array([len(o.response_ids) for o in outputs]),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                final_output.reward_score = result["reward_score"]
                final_output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            final_output.metrics.compute_score = timing["compute_score"]

    async def _compute_teacher_logprobs(
        self,
        output: AgentLoopOutput,
        prompt_ids: list[int],
        response_ids: list[int],
        validate: bool,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Compute teacher logprobs for single sample."""
        if self.distillation_enabled and not validate:
            routing_key = None
            if sample_kwargs is not None:
                routing_value = sample_kwargs.get(self.teacher_key)
                if routing_value is not None:
                    # Non-tensor batch values arrive as 0-d numpy objects / arrays; normalize to Python.
                    routing_key = routing_value.item() if hasattr(routing_value, "item") else routing_value
            teacher_ids, teacher_logprobs = await self.teacher_server_manager.compute_teacher_logprobs_single(
                sequence_ids=prompt_ids + response_ids,
                multi_modal_data=output.multi_modal_data,
                mm_processor_kwargs=output.mm_processor_kwargs,
                routing_key=routing_key,
            )
            output.extra_fields["teacher_ids"] = teacher_ids
            output.extra_fields["teacher_logprobs"] = teacher_logprobs

    def _postprocess(
        self,
        inputs: list[_InternalAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
        validate: bool = False,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)
        if inputs[0].teacher_logprobs is not None and inputs[0].teacher_ids is not None:
            optional_outputs["teacher_logprobs"] = torch.cat([input.teacher_logprobs for input in inputs], dim=0)
            optional_outputs["teacher_ids"] = torch.cat([input.teacher_ids for input in inputs], dim=0)
        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        # Keep a stable set of keys so downstream batch concat stays consistent across agent loops.
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server.
        teacher_client (dict[str, LLMServerClient]): Client for multiple teacher servers.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.llm_client = llm_client
        self.teacher_client = teacher_client
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    self.llm_client,
                    self.teacher_client,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Attach per-sample priority to the batch (like ``uid``) so each sample gets
        # a globally-unique priority that flows to vLLM request scheduling. Assigned
        # before chunking so chunks own disjoint ranges without per-worker offsets.
        if "priority" not in prompts.non_tensor_batch:
            prompts.non_tensor_batch["priority"] = np.arange(len(prompts), dtype=np.int64)

        t0 = time.perf_counter()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        t_workers = time.perf_counter() - t0
        t1 = time.perf_counter()
        output = DataProto.concat(outputs)
        t_concat = time.perf_counter() - t1

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        # manager-level wall: workers (dispatch + agent loops + transport back) vs concat
        timing["agent_loop/manager/workers_wall"] = t_workers
        timing["agent_loop/manager/concat"] = t_concat

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()
        for key in ("preprocess", "postprocess", "wall"):
            vals = np.array([float(metric.get(key, 0.0)) for chunk in metrics for metric in chunk])
            timing[f"agent_loop/{key}/mean"] = vals.mean()
            timing[f"agent_loop/{key}/max"] = vals.max()
            timing[f"agent_loop/{key}/sum"] = vals.sum()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        if "attention_mask" in output.batch:
            attention_mask = output.batch["attention_mask"][slowest]
            timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
            timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing
