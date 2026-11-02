"""Actual configured agent/dataset imports and exact session-start boundaries (CPU)."""
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 3

    def __init__(self):
        self.render_calls = 0

    def encode(self, text, **kwargs):
        return [ord(char) + 10 for char in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(token - 10) for token in ids if token >= 10)

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, **kwargs):
        self.render_calls += 1
        text = ''.join('<' + msg['role'] + '>' + msg['content'] + '</end>' for msg in messages)
        text += '<assistant>' if add_generation_prompt else ''
        return self.encode(text) if tokenize else text


def make_agent(monkeypatch, *, prompt_length=12000, tier1=False):
    # These are genuine modules, classes, configuration wrappers and Hydra factory.
    # No backend import or parent initializer is replaced.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    module = importlib.import_module('axion.training.rl.agent_loop.deepresearch_agent_loop')
    monkeypatch.setenv('MEMORY_TIER1', '1' if tier1 else '0')
    monkeypatch.setenv('FORCE_FINAL_ANSWER', '0')
    monkeypatch.setenv('SOFTEN_ANSWER_PROMPT', '0')
    monkeypatch.delenv('VERL_CHAT_TEMPLATE_FILE', raising=False)
    monkeypatch.setattr(module.DeepResearchAgentLoop, '_class_initialized', False)
    config = OmegaConf.create({
        'async_training': {'axion_recipe': True},
        'data': {'continuous_token': {'enable': False}, 'apply_chat_template_kwargs': {}},
        'actor_rollout_ref': {'rollout': {
            'multi_turn': {'max_assistant_turns': 7, 'max_user_turns': 9},
            'prompt_length': prompt_length, 'response_length': 49152,
            'max_model_len': 61152, 'enable_rollout_routing_replay': True}}})
    registry = OmegaConf.load(ROOT / 'axion/training/rl/config/agent_loop_config.yaml')
    entry = OmegaConf.to_container(registry[0])
    entry.pop('name')
    tokenizer = CharacterTokenizer()
    agent = instantiate(entry, trainer_config=DictConfigWrap(config),
        server_manager=SimpleNamespace(), tokenizer=tokenizer, processor=None,
        dataset_cls=RLHFDataset, data_config=DictConfigWrap(config.data),
        tools=ToolListWrap([]), mm_cache=None)
    return module, agent, tokenizer


def test_configured_factory_imports_and_accepts_backend_keyword(monkeypatch):
    async def check():
        module, agent, tokenizer = make_agent(monkeypatch)
        assert isinstance(agent, module.DeepResearchAgentLoop)
        assert agent.tokenizer is tokenizer
        assert agent.max_turns == 16
        assert agent.data_config.continuous_token.enable is False
    asyncio.run(check())


def test_initial_prompt_overflow_fails_before_any_generation_or_state_mutation(monkeypatch):
    async def check():
        module, agent, _ = make_agent(monkeypatch, prompt_length=100)
        data = module.AgentData([], {}, 'request', {}, 'question remains whole')
        with pytest.raises(ValueError, match='session prompt.*100'):
            await agent._handle_pending_state(data, {})
        assert data.prompt_ids == []
        assert data.messages == []
        assert data.original_user_content == 'question remains whole'
    asyncio.run(check())


@pytest.mark.parametrize('tier1', [False, True])
def test_oversized_memory_rollover_preserves_current_sample_without_empty_session(monkeypatch, tier1):
    async def check():
        module, agent, _ = make_agent(monkeypatch, prompt_length=100, tier1=tier1)
        data = module.AgentData([{'role': 'assistant', 'content': 'existing evidence'}], {}, 'request', {}, 'question')
        data.prompt_ids = [1, 2, 3, 4]
        data.response_ids = [3, 4]
        data.response_mask = [1, 1]
        data.response_logprobs = [-.2, -.3]
        data.routed_experts = torch.ones(3, 1, 1, dtype=torch.uint8)
        original_routes = data.routed_experts
        data.full_messages = list(data.messages)
        data.full_response = 'existing evidence'
        data.session_response = data.full_response
        data.assistant_turns = 1
        if tier1:
            monkeypatch.setattr(agent, '_tier1_compact', lambda messages: (
                [{'role': 'system', 'content': 'x' * 200}],
                {'chars_saved': 1, 'stubbed': 1, 'think_stripped': 0}))
        else:
            async def condense(*args):
                return json.dumps({'evidence': 'x' * 200})
            monkeypatch.setattr(module, 'call_condenser_async', condense)
        state = await agent._handle_memory_trigger_state(data, 'question')
        assert state == module.AgentState.TERMINATED
        assert data.session_id == 0 and data.completed_sessions == []
        assert data.prompt_ids == [1, 2, 3, 4] and data.response_ids == [3, 4]
        assert data.response_mask == [1, 1] and data.response_logprobs == [-.2, -.3]
        assert data.routed_experts is original_routes
        assert data.memory_state is None and data.pending_messages is None
        assert data.metrics['session_prompt_overflow'] == 1
        assert data.full_messages == [{'role': 'assistant', 'content': 'existing evidence'}]
    asyncio.run(check())


def test_accepted_memory_prefix_is_tokenized_once_and_cached_across_rotation(monkeypatch):
    async def check():
        module, agent, tokenizer = make_agent(monkeypatch)
        data = module.AgentData([{'role': 'assistant', 'content': 'evidence'}], {}, 'old-request', {}, 'question')
        data.prompt_ids = [1, 2, 3]
        data.response_ids = [3]
        data.response_mask = [1]
        data.response_logprobs = [-.2]
        data.routed_experts = torch.ones(2, 1, 1, dtype=torch.uint8)
        data.full_messages = list(data.messages)
        async def condense(*args):
            return json.dumps({'evidence': 'small'})
        monkeypatch.setattr(module, 'call_condenser_async', condense)
        before = tokenizer.render_calls
        assert await agent._handle_memory_trigger_state(data, 'question') == module.AgentState.PENDING
        assert data.session_id == 1 and len(data.completed_sessions) == 1
        assert data.completed_sessions[0].response_ids == [3]
        assert data.request_id != 'old-request'
        cached = list(data.pending_prompt_ids)
        assert await agent._handle_pending_state(data, {}) == module.AgentState.GENERATING
        assert data.prompt_ids == cached
        assert tokenizer.render_calls == before + 1
        assert data.pending_prompt_ids is None
        assert data.full_messages == [{'role': 'assistant', 'content': 'evidence'}]
    asyncio.run(check())


def test_configured_dataset_filters_actual_system_prompt_and_passes_exact_initial_ids(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    import datasets
    from verl.utils.dataset.rl_dataset import get_dataset_class
    from axion.training.rl.prompts import build_initial_messages, tokenize_session_prompt
    monkeypatch.setattr(datasets.config, 'HF_DATASETS_CACHE', tmp_path / 'cache')
    monkeypatch.setenv('SOFTEN_ANSWER_PROMPT', '0')
    monkeypatch.delenv('VERL_CHAT_TEMPLATE_FILE', raising=False)
    tokenizer = CharacterTokenizer()
    accepted_messages = build_initial_messages('ok')
    cap = len(tokenize_session_prompt(tokenizer, accepted_messages, {}))
    config = OmegaConf.load(ROOT / 'axion/training/rl/config/paper.yaml').data
    config.max_prompt_length = cap
    config.filter_overlong_prompts_workers = 1
    config.cache_dir = str(tmp_path / 'copy-cache')
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps({'prompt': [{'role': 'user', 'content': q}], 'data_source': 'axion'}) + '\n'
                              for q in ['ok', 'x' * cap]))
    dataset_cls = get_dataset_class(config)
    dataset = dataset_cls(str(source), tokenizer, config)
    assert len(dataset) == 1
    row = dataset[0]
    assert row['raw_prompt'][0]['content'] == 'ok'
    assert row['axion_initial_messages'] == accepted_messages
    assert row['axion_initial_prompt_ids'] == tokenize_session_prompt(tokenizer, accepted_messages, {})
    assert len(row['axion_initial_prompt_ids']) == cap
    async def check():
        module, agent, agent_tokenizer = make_agent(monkeypatch, prompt_length=cap)
        data = module.AgentData(row['raw_prompt'], {}, 'request', {}, 'ok')
        data.pending_messages = row['axion_initial_messages']
        data.pending_prompt_ids = row['axion_initial_prompt_ids']
        before = agent_tokenizer.render_calls
        assert await agent._handle_pending_state(data, {}) == module.AgentState.GENERATING
        assert data.prompt_ids == row['axion_initial_prompt_ids']
        assert agent_tokenizer.render_calls == before
    asyncio.run(check())


def test_full_run_consumes_numpy_cached_prefix_and_exposes_overflow_without_empty_session(monkeypatch):
    import numpy as np
    from axion.training.rl.prompts import build_initial_messages, tokenize_session_prompt
    async def check():
        module, agent, tokenizer = make_agent(monkeypatch)
        initial_messages = build_initial_messages('whole question')
        initial_ids = tokenize_session_prompt(tokenizer, initial_messages, {})
        monkeypatch.setattr(type(agent), "prompt_length", len(initial_ids))
        monkeypatch.setattr(type(agent), "response_length", 2)
        sampled = tokenizer.encode('ok')
        generated_calls = []
        async def generate(**kwargs):
            generated_calls.append(list(kwargs['prompt_ids']))
            return SimpleNamespace(token_ids=sampled, log_probs=[-.2, -.3],
                routed_experts=torch.ones(len(initial_ids) + 1, 1, 1, dtype=torch.uint8),
                extra_fields={'global_steps': 0})
        agent.server_manager.generate = generate
        async def condense(*args):
            return json.dumps({'evidence': 'over the initial prefix cap'})
        monkeypatch.setattr(module, 'call_condenser_async', condense)
        before = tokenizer.render_calls
        output = await agent.run({}, raw_prompt=np.array([{'role': 'user', 'content': 'whole question'}], dtype=object),
            axion_initial_messages=np.array(initial_messages, dtype=object),
            axion_initial_prompt_ids=np.array(initial_ids))
        assert generated_calls == [initial_ids]
        assert output.prompt_ids == initial_ids and output.response_ids == sampled
        assert output.response_mask == [1, 1]
        assert output.extra_fields['session_prompt_overflow'] == 1
        assert output.extra_fields['num_sessions'] == 1
        assert output.extra_fields['session_outputs'][0]['response_ids'] == sampled
        assert output.extra_fields['min_global_steps'] == 0
        assert output.extra_fields['full_messages'][-1] == {'role': 'assistant', 'content': 'ok'}
        # Only the rejected condenser prefix is rendered; the initial IDs are consumed verbatim.
        assert tokenizer.render_calls == before + 1
    asyncio.run(check())
