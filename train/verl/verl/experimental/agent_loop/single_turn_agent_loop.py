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
import asyncio
import logging
import os
from typing import Any
from uuid import uuid4

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _put_multimodal_for_generate(images, videos, audios):
    """Store multimodal payloads once in the Ray object store.

    ``agent_loop_n`` fans out ``n`` ``generate.remote`` calls that share the same
    decoded frames. Passing raw Python objects would re-serialize them per RPC;
    ``ray.put`` once and pass ObjectRefs so plasma holds a single copy.
    """
    if images is not None:
        images = ray.put(images)
    if videos is not None:
        videos = ray.put(videos)
    if audios is not None:
        audios = ray.put(audios)
    return images, videos, audios


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput | list[AgentLoopOutput]:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        n = int(kwargs.get("agent_loop_n", 1))
        if n < 1:
            raise ValueError(f"agent_loop_n must be >= 1, got {n}")

        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # apply chat template / tokenize ONCE, reused across the n generate RPCs (feature 3).
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # ray.put multimodal once, before the fan-out generate RPCs (feature 5).
        image_data, video_data, audio_data = _put_multimodal_for_generate(images, videos, audios)

        async def _one_generate(idx: int) -> AgentLoopOutput:
            metrics = {}
            with simple_timer("generate_sequences", metrics):
                # distinct id per fan-out so determinism mode does not collide across n
                request_id = (
                    f"det-{priority}-{idx}"
                    if getattr(self.rollout_config, "full_determinism", False)
                    else uuid4().hex
                )
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    audio_data=audio_data,
                    video_data=video_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    priority=priority,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

            if use_continuous_token:
                merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                    prompt_ids,
                    output.token_ids,
                    [],
                    [] if output.log_probs else None,
                    assistant_logprobs=output.log_probs if output.log_probs else None,
                )
                response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
                out_prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
            else:
                response_ids = output.token_ids
                response_mask = [1] * len(output.token_ids)
                response_logprobs = output.log_probs
                out_prompt_ids = prompt_ids

            result = AgentLoopOutput(
                prompt_ids=out_prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
                routed_experts=(
                    output.routed_experts[: len(out_prompt_ids) + self.response_length]
                    if output.routed_experts is not None
                    else None
                ),
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                num_turns=2,
                metrics=metrics,
                extra_fields=dict(output.extra_fields),
            )
            # keeping the schema consistent with tool_agent_loop
            result.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            return result

        if n == 1:
            return await _one_generate(0)

        outputs = list(await asyncio.gather(*[_one_generate(i) for i in range(n)]))
        for output in outputs:
            object.__setattr__(output, "multi_modal_data", multi_modal_data)
        return outputs
