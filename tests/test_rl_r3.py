"""CPU contracts for exact rollout-token routing and Megatron replay layout."""
import ast
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / 'third_party/verl/verl'


def load_helper():
    path = VERL / 'utils/rollout_routing.py'
    assert path.is_file(), 'Shared route validation/merging helper is required'
    spec = importlib.util.spec_from_file_location('rl_route_test_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def production_def(path, name, namespace):
    """Execute real production definitions; replace only unavailable GPU/Ray imports."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_merge_routes_retains_original_prefix_and_compacts_numpy():
    helper = load_helper()
    first = np.full((4, 2, 2), 7, dtype=np.int32)
    later = np.full((8, 2, 2), 9, dtype=np.int32)
    original = helper.merge_routed_experts(None, first, 3, 2, required=True, max_experts=256)
    result = helper.merge_routed_experts(original, later, 6, 3, required=True, max_experts=256)
    assert result.dtype == torch.uint8
    assert torch.equal(result[:4], torch.full((4, 2, 2), 7, dtype=torch.uint8))
    assert torch.equal(result[4:], torch.full((4, 2, 2), 9, dtype=torch.uint8))


@pytest.mark.parametrize('routes', [None, np.zeros((3, 2, 2)), np.zeros((4, 2)),
                                    np.full((4, 2, 2), -1), np.full((4, 2, 2), 256)])
def test_required_route_payload_rejects_missing_misaligned_or_invalid_ids(routes):
    with pytest.raises((ValueError, TypeError), match='rout|expert'):
        load_helper().merge_routed_experts(None, routes, 3, 2, required=True, max_experts=256)


def test_session_coverage_ignores_unforwarded_tool_suffix_but_rejects_causal_gap():
    helper = load_helper()
    routes = torch.full((4, 2, 2), 7, dtype=torch.uint8)
    clipped, mask = helper.session_routing(routes, 3, [11, 12, 99, 98], [1, 1, 0, 0], 10, required=True)
    assert torch.equal(clipped, routes)
    assert mask.tolist() == [True] * 4 + [False] * 3
    with pytest.raises(ValueError, match='causal|coverage'):
        helper.session_routing(routes[:3], 3, [11, 12, 99], [1, 1, 0], 10, required=True)


def test_version_bounds_include_zero_and_span_sessions():
    helper = load_helper()
    versions = {}
    helper.update_version_bounds(versions, {'min_global_steps': 0, 'max_global_steps': 2}, required=True)
    helper.update_version_bounds(versions, {'global_steps': 4}, required=True)
    assert versions == {'min_global_steps': 0, 'max_global_steps': 4}
    with pytest.raises(ValueError, match='version'):
        helper.update_version_bounds({}, {}, required=True)


def test_modern_client_resumes_with_exact_tokens_routes_and_version_bounds(monkeypatch):
    helper = load_helper()
    outputs = [
        SimpleNamespace(token_ids=[], log_probs=[], routed_experts=None, num_preempted=0,
                        stop_reason='aborted', extra_fields={'global_steps': 0}),
        SimpleNamespace(token_ids=[10, 11], log_probs=[-.1, -.2], routed_experts=np.full((4, 2, 2), 3),
                        num_preempted=1, stop_reason='aborted', extra_fields={'global_steps': 0}),
        SimpleNamespace(token_ids=[12], log_probs=[-.3], routed_experts=np.full((5, 2, 2), 8),
                        num_preempted=0, stop_reason='completed', extra_fields={'global_steps': 2}),
    ]
    calls = []
    class Parent:
        async def generate(self, **kwargs):
            calls.append((list(kwargs['prompt_ids']), dict(kwargs['sampling_params'])))
            return outputs.pop(0)
    class Output(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(routed_experts=None, stop_reason=None, extra_fields={}, **kwargs)
    async def no_wait(_): pass
    namespace = dict(LLMServerClient=Parent, rollout_trace_op=lambda f: f, TokenOutput=Output,
                     normalize_token_ids=list, _COLLAPSE_IMAGE_BLOCK_IDS=None, torch=torch,
                     asyncio=SimpleNamespace(sleep=no_wait), Optional=__import__('typing').Optional,
                     Any=__import__('typing').Any, merge_routed_experts=helper.merge_routed_experts,
                     update_version_bounds=helper.update_version_bounds)
    cls = production_def(VERL / 'workers/rollout/llm_server.py', 'FullyAsyncLLMServerClient', namespace)
    client = cls()
    client.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(enable_rollout_routing_replay=True)),
                                    async_training=SimpleNamespace(partial_rollout=True, axion_recipe=True))
    params = {'max_tokens': 3}
    result = asyncio.run(client.generate(request_id='sticky', prompt_ids=[1, 2, 3], sampling_params=params))
    assert [call[0] for call in calls] == [[1, 2, 3], [1, 2, 3], [1, 2, 3, 10, 11]]
    assert [call[1]['max_tokens'] for call in calls] == [3, 3, 1]
    assert params == {'max_tokens': 3}
    assert result.token_ids == [10, 11, 12]
    assert result.log_probs == [-.1, -.2, -.3]
    assert result.routed_experts.dtype == torch.uint8
    assert result.routed_experts[:, 0, 0].tolist() == [3, 3, 3, 3, 8]
    assert result.extra_fields['min_global_steps'] == 0
    assert result.extra_fields['max_global_steps'] == 2


def test_bshd_routes_match_sequence_first_router_order_with_padding():
    helper = load_helper()
    # Model hidden rows flatten [sequence, batch]; unequal lengths and forced S=6.
    routes = torch.tensor([[[[10]], [[11]], [[12]], [[0]], [[0]], [[0]]],
                           [[[20]], [[21]], [[22]], [[23]], [[24]], [[0]]]], dtype=torch.uint8)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    for rank in (0, 1):
        def scatter(t): return t.chunk(2, dim=0)[rank]
        flat, covered = helper.bshd_router_rows(routes, mask, scatter)
        expected = [10, 20, 11, 21, 12, 22] if rank == 0 else [0, 23, 0, 24, 0, 0]
        assert flat[:, 0, 0].tolist() == expected
        assert covered.tolist() == ([True, True, True, True, False, True] if rank == 0 else [False, True, False, False, False, False])


def test_base_agent_consumes_token_ids_and_keeps_routes_and_versions_across_turns():
    helper = load_helper()
    path = ROOT / 'axion/training/rl/agent_loop/deepresearch_agent_loop.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepResearchAgentLoop')
    method = next((n for n in cls.body if getattr(n, 'name', '') == '_append_generation_output'), None)
    assert method is not None, 'Base agent needs one validated token-output consumption boundary'
    ns = dict(merge_routed_experts=helper.merge_routed_experts, update_version_bounds=helper.update_version_bounds,
              AgentData=object)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    actor = SimpleNamespace(config=SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(enable_rollout_routing_replay=True)),
        async_training=SimpleNamespace(axion_recipe=True)))
    data = SimpleNamespace(prompt_ids=[1, 2, 3], response_ids=[], response_mask=[], response_logprobs=[],
                           routed_experts=None, extra_fields={})
    append = ns['_append_generation_output']
    append(actor, data, SimpleNamespace(token_ids=[10, 11], log_probs=[-.1, -.2],
        routed_experts=np.full((4, 2, 2), 3), extra_fields={'min_global_steps': 0, 'max_global_steps': 1}))
    # Actual TITO tool append uses zero policy mask and leaves previous token IDs unchanged.
    data.prompt_ids.extend([90, 91]); data.response_ids.extend([90, 91])
    data.response_mask.extend([0, 0]); data.response_logprobs.extend([0., 0.])
    append(actor, data, SimpleNamespace(token_ids=[12], log_probs=[-.3],
        routed_experts=np.full((7, 2, 2), 8), extra_fields={'global_steps': 2}))
    assert data.prompt_ids == [1, 2, 3, 10, 11, 90, 91, 12]
    assert data.response_ids == [10, 11, 90, 91, 12]
    assert data.response_mask == [1, 1, 0, 0, 1]
    assert data.routed_experts[:, 0, 0].tolist() == [3, 3, 3, 3, 8, 8, 8]
    assert data.extra_fields['min_global_steps'] == 0 and data.extra_fields['max_global_steps'] == 2


def test_compact_padding_slices_route_sequence_axis():
    path = VERL / 'workers/utils/padding.py'
    ns = {'torch': torch, 'DataProto': object}
    for name in ('_slice_full_sequence', 'compact_left_right_padding'):
        production_def(path, name, ns)
    # Prompt cap 4, response cap 4; real prompt 2 and response 3.
    batch = SimpleNamespace(batch={
        'prompts': torch.tensor([[0, 0, 1, 2]]), 'responses': torch.tensor([[3, 4, 5, 0]]),
        'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]]),
        'input_ids': torch.tensor([[0, 0, 1, 2, 3, 4, 5, 0]]),
        'routed_experts': torch.arange(8, dtype=torch.uint8).reshape(1, 8, 1, 1),
        'routed_experts_mask': torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
    })
    ns['_RESPONSE_ONLY_KEYS'] = ('responses',)
    ns['compact_left_right_padding'](batch)
    assert batch.batch['routed_experts'].shape == (1, 5, 1, 1)
    assert batch.batch['routed_experts'][0, :, 0, 0].tolist() == [2, 3, 4, 5, 6]
    assert batch.batch['routed_experts_mask'].tolist() == [[True, True, True, True, False]]


def test_megatron_replay_uses_real_bshd_preprocessor_and_sequence_parallel_order():
    import math
    from typing import Optional
    helper = load_helper()
    mpu = SimpleNamespace(get_context_parallel_world_size=lambda: 1, get_context_parallel_rank=lambda: 0,
                          get_tensor_model_parallel_world_size=lambda: 2)
    ns = dict(torch=torch, math=math, Optional=Optional, mpu=mpu, is_npu_available=False)
    preprocess = production_def(VERL / 'models/mcore/util.py', 'preprocess_bshd_engine', ns)
    routes = torch.nested.nested_tensor([torch.tensor([10, 11, 12]).reshape(3, 1, 1),
                                        torch.tensor([20, 21, 22, 23, 24]).reshape(5, 1, 1)], layout=torch.jagged)
    coverage = torch.nested.nested_tensor([torch.tensor([1, 1, 0], dtype=torch.bool),
                                          torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)], layout=torch.jagged)
    captured = []
    router = SimpleNamespace(set_target_indices=lambda x, replay_mask: captured.append((x, replay_mask)))
    for rank in (0, 1):
        ns = dict(torch=torch, device_name='cpu', preprocess_bshd_engine=preprocess,
                  bshd_router_rows=helper.bshd_router_rows,
                  scatter_to_sequence_parallel_region=lambda x: x.chunk(2, dim=0)[rank],
                  get_current_rank_layer_info=lambda config, vp: {'start': 0, 'end': 1},
                  is_moe_layer=lambda config, index: True,
                  RouterReplayHelper=SimpleNamespace(get_micro_batch_router_list=lambda config, vp: [router]))
        fn = production_def(VERL / 'utils/megatron/router_replay_utils.py', 'set_router_replay_data', ns)
        fn(routes, None, SimpleNamespace(fp8=None, num_layers=1, sequence_parallel=True),
           replay_mask=coverage, data_format='bshd', forced_max_seqlen=6)
    assert captured[0][0][:, 0].tolist() == [10, 20, 11, 21, 12, 22]
    assert captured[1][0][:, 0].tolist() == [0, 23, 0, 24, 0, 0]
    assert captured[1][1].tolist() == [False, True, False, False, False, False]
    wrong_layers = torch.nested.nested_tensor([torch.ones(3, 2, 1, dtype=torch.uint8)], layout=torch.jagged)
    with pytest.raises(ValueError, match="layer"):
        fn(wrong_layers, None, SimpleNamespace(fp8=None, num_layers=1, sequence_parallel=True),
           data_format='bshd', forced_max_seqlen=6)


def test_replay_mask_ends_before_last_policy_token_despite_trailing_tool_tokens():
    ns = {'torch': torch}
    fn = production_def(VERL / 'utils/megatron/router_replay_utils.py', 'build_r3_replay_mask', ns)
    ids = torch.nested.nested_tensor([torch.arange(7), torch.arange(5)], layout=torch.jagged)
    policy_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    response_attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    mask = fn(ids, policy_mask, response_attention_mask=response_attention)
    assert mask[0].tolist() == [True] * 4 + [False] * 3
    assert not mask[1].any()


def test_vllm_server_versions_cover_weight_update_during_generation():
    import contextlib
    from typing import Any, Optional
    path = VERL / 'workers/rollout/vllm_rollout/vllm_async_server.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'vLLMHttpServer')
    method = next(n for n in cls.body if getattr(n, 'name', '') == 'generate')
    method.decorator_list = []
    class Config(SimpleNamespace):
        def get(self, key, default): return getattr(self, key, default)
    class Sampling(SimpleNamespace):
        def __init__(self, **kwargs): super().__init__(prompt_logprobs=None, **kwargs)
    async def resolve(value): return value
    server = SimpleNamespace(global_steps=0, replica_rank=0, _disaggregation_role='null', lora_as_adapter=False,
        _mm_ref_cache=SimpleNamespace(resolve=resolve), model_config=SimpleNamespace(processor=None),
        config=Config(max_model_len=20, response_length=5, prompt_length=5, full_determinism=False,
                      enable_rollout_routing_replay=True, mtp=None))
    async def generate(**kwargs):
        # Simulate set_global_steps racing completion delivery after a pause/abort.
        server.global_steps = 2
        yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5], logprobs=[{5: SimpleNamespace(logprob=-.2)}],
            routed_experts=np.ones((2, 1, 1), dtype=np.int32), finish_reason='abort')])
    server.engine = SimpleNamespace(generate=generate)
    ns = dict(Any=Any, Optional=Optional, TokenOutput=SimpleNamespace, RequestOutput=object,
              normalize_token_ids=list, SamplingParams=Sampling, TokensPrompt=lambda **kwargs: kwargs,
              qwen2_5_vl_dedup_image_tokens=lambda ids, processor: ids,
              extract_prompt_logprobs=lambda **kwargs: None,
              RLInsightLogger=SimpleNamespace(trace_state=lambda *args, **kwargs: contextlib.nullcontext()))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    output = asyncio.run(ns['generate'](server, [1, 2], {'max_tokens': 1, 'logprobs': True}, 'req'))
    assert output.token_ids == [5]
    assert output.extra_fields['min_global_steps'] == 0
    assert output.extra_fields['max_global_steps'] == 2


def test_worker_unpadding_keeps_coverage_aligned_with_routes():
    # FlashAttention's index_first_axis requires ndim >= 2. Isolate that CUDA
    # package boundary while exercising the production worker conversion.
    def index_first_axis(values, indices):
        assert values.ndim >= 2
        return values.index_select(0, indices)
    def unpad(values, attention):
        indices = attention.flatten().nonzero().flatten()
        lengths = attention.sum(-1)
        offsets = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)))
        return values.flatten(0, 1).index_select(0, indices), indices, offsets
    metadata = SimpleNamespace(assign_non_tensor_data=lambda data, key, value: data.__setitem__(key, value))
    ns = dict(torch=torch, TensorDict=dict, tu=metadata, index_first_axis=index_first_axis, unpad_input=unpad)
    convert = production_def(VERL / 'workers/utils/padding.py', 'left_right_2_no_padding', ns)
    data = dict(input_ids=torch.tensor([[0, 0, 11, 12, 41, 42, 43, 0]]),
                attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]]),
                response_mask=torch.tensor([[1, 0, 1, 0]]), position_ids=torch.arange(8).reshape(1, 8),
                routed_experts=torch.arange(8, dtype=torch.uint8).reshape(1, 8, 1, 1),
                routed_experts_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]], dtype=torch.bool))
    result = convert(data)
    assert result['input_ids'][0].tolist() == [11, 12, 41, 42, 43]
    assert result['routed_experts'][0][:, 0, 0].tolist() == [2, 3, 4, 5, 6]
    assert result['routed_experts_mask'][0].tolist() == [True, True, True, True, False]


@pytest.mark.parametrize('paper,legacy_total,expected', [
    (True, None, 16), (False, None, 100), (False, 25, 25),
])
def test_agent_turn_limit_uses_structured_role_limits_only_for_paper(paper, legacy_total, expected):
    import logging
    import os
    path = ROOT / 'axion/training/rl/agent_loop/deepresearch_agent_loop.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeepResearchAgentLoop')
    method = next(n for n in cls.body if getattr(n, 'name', '') == 'init_class')
    method.decorator_list = []
    namespace = dict(logger=logging.getLogger(__name__), os=os,
                     _env_bool=lambda name, default: default,
                     _env_int=lambda name, default: default,
                     initialize_system_prompt=lambda tokenizer, **kwargs: [])
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    turn_config = dict(max_assistant_turns=7, max_user_turns=9)
    if legacy_total is not None:
        turn_config['max_turns'] = legacy_total
    config = SimpleNamespace(
        async_training=SimpleNamespace(axion_recipe=paper), data={},
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            multi_turn=turn_config, prompt_length=12000, response_length=49152)))
    agent_class = SimpleNamespace(_class_initialized=False, _compute_turn_boundary_tokens=lambda: None)
    namespace['init_class'](agent_class, config, object(), None)
    assert agent_class.max_turns == expected
    assert agent_class.max_assistant_turns == 7
    assert agent_class.max_user_turns == 9
