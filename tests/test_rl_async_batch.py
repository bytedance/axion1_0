"""Numerical regressions for paper GRPO across variable-length sessions."""
import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def api():
    path = ROOT / 'axion/training/rl/async_batch.py'
    assert path.is_file(), 'Paper async session batching has not been implemented'
    spec = importlib.util.spec_from_file_location('axion_async_batch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _session(sid, ids=(41, 42, 43), mask=(1, 0, 1)):
    return dict(session_id=sid, prompt_ids=[11, 12], response_ids=list(ids),
                response_mask=list(mask), response_logprobs=[-.1, 0., -.3][:len(ids)])


def _trajectory_batch():
    rewards = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.])
    sessions = [[_session(0)] for _ in range(8)]
    sessions[-1] = [_session(0), _session(1, (51, 52), (1, 1)), _session(2)]
    return rewards, sessions


def test_group_normalization_precedes_unequal_session_expansion(api):
    rewards, sessions = _trajectory_batch()
    advantages = api.normalize_trajectory_rewards(rewards)
    tensors, fields, count = api.expand_session_rows(
        sessions, rewards, advantages, {'uid': np.array(['g'] * 8, dtype=object)},
        prompt_length=4, response_length=5, pad_token_id=99, require_routes=False,
    )
    # Four zero rewards and four unit rewards => unbiased std sqrt(2/7).
    expected = .5 / ((2 / 7) ** .5 + 1e-6)
    assert count == 10
    assert tensors['trajectory_advantage'][:4].tolist() == pytest.approx([-expected] * 4)
    assert tensors['trajectory_advantage'][4:].tolist() == pytest.approx([expected] * 6)
    assert fields['trajectory_id'][-3:].tolist() == ['g:7'] * 3
    assert fields['session_id'][-3:].tolist() == [0, 1, 2]
    assert tensors['rm_scores'].sum(-1).tolist() == [0.] * 4 + [1.] * 6


def test_expansion_preserves_tito_and_relocates_session_reward(api):
    tensors, _, _ = api.expand_session_rows(
        [[_session(0), _session(1, (51, 52), (1, 1))]], torch.tensor([1.]), torch.tensor([.7]),
        {'uid': np.array(['g'], dtype=object)}, prompt_length=4, response_length=5,
        pad_token_id=99, require_routes=False,
    )
    assert tensors['prompts'].tolist() == [[99, 99, 11, 12]] * 2
    assert tensors['responses'].tolist() == [[41, 42, 43, 99, 99], [51, 52, 99, 99, 99]]
    assert tensors['response_mask'].tolist() == [[1, 0, 1, 0, 0], [1, 1, 0, 0, 0]]
    assert tensors['rm_scores'].tolist() == [[0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.]]
    assert tensors['rollout_log_probs'][0].tolist() == pytest.approx([-.1, 0., -.3, 0., 0.])


@pytest.mark.parametrize('oldest,current,want', [(0, 11, None), (0, 12, 'stale'), (1, 12, None)])
def test_staleness_uses_oldest_weight_and_keeps_boundary(api, oldest, current, want):
    reason = api.prompt_group_rejection(
        torch.tensor([0., 1.] * 4), ['g'] * 8, [oldest] * 8, [current] * 8, current,
    )
    assert reason == want


def test_degenerate_invalid_and_incomplete_groups(api):
    assert api.prompt_group_rejection(torch.ones(8), ['g'] * 8, [0] * 8, [0] * 8, 0) == 'degenerate'
    assert api.prompt_group_rejection(torch.tensor([float('nan')] + [0.] * 7), ['g'] * 8, [0] * 8, [0] * 8, 0) == 'invalid_reward'
    with pytest.raises(ValueError, match='eight'):
        api.prompt_group_rejection(torch.ones(7), ['g'] * 7, [0] * 7, [0] * 7, 0)
    with pytest.raises(ValueError, match='version'):
        api.prompt_group_rejection(torch.arange(8.), ['g'] * 8, [None] * 8, [0] * 8, 0)


def test_padding_and_microbatch_denominator_preserve_session_gradient(api):
    tensors, fields, count = api.expand_session_rows(
        [[_session(0), _session(1, (51, 52), (1, 1)), _session(2)]],
        torch.tensor([1.]), torch.tensor([.7]), {'uid': np.array(['g'], dtype=object)},
        prompt_length=4, response_length=5, pad_token_id=99, require_routes=False,
    )
    tensors, fields = api.pad_session_rows(tensors, fields, divisor=4)
    assert count == 3 and tensors['input_ids'].shape[0] == 4
    assert tensors['response_mask'][-1].sum() == 0
    assert tensors['rm_scores'][-1].sum() == 0
    assert tensors['trajectory_advantage'][-1] == 0
    assert fields['is_padding'].tolist() == [False, False, False, True]
    parameter = torch.tensor(2., requires_grad=True)
    # Per-session means are 2, 6, 10; dummy is ignored. DP average must
    # cancel its scaling irrespective of unequal microbatch boundaries.
    values = parameter * torch.tensor([[1.] * 5, [3.] * 5, [5.] * 5, [1000.] * 5])
    mask = tensors['response_mask']
    from verl.trainer.ppo.core_algos import agg_loss
    result = sum(agg_loss(values[i:j], mask[i:j], 'seq-mean-token-mean', global_batch_size=count, dp_size=2)
                 for i, j in [(0, 1), (1, 2), (2, 4)]) / 2
    result.backward()
    assert result.item() == pytest.approx(6.)
    assert parameter.grad.item() == pytest.approx(3.)


def test_queue_collector_replenishes_rejected_groups_and_never_returns_partial(api):
    async def run():
        queue = asyncio.Queue()
        for item in ['bad'] * 600 + list(range(24)):
            await queue.put((item, 0))
        await queue.put((None, 0))
        accepted, counts = await api.collect_prompt_groups(queue.get, lambda x: 'degenerate' if x == 'bad' else None)
        assert accepted == list(range(24))
        assert counts == {'degenerate': 600}
        assert (await api.collect_prompt_groups(queue.get, lambda x: None))[0] is None
        for item in range(23):
            await queue.put((item, 0))
        await queue.put((None, 0))
        assert (await api.collect_prompt_groups(queue.get, lambda x: None))[0] is None
    asyncio.run(run())


def test_route_coverage_and_alignment(api):
    session = _session(0)
    session['routed_experts'] = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=torch.uint8)
    session['routed_experts_mask'] = [True, True, True, True, False]
    tensors, _, _ = api.expand_session_rows(
        [[session]], torch.tensor([1.]), torch.tensor([.7]), {'uid': np.array(['g'], dtype=object)},
        prompt_length=4, response_length=5, pad_token_id=99, require_routes=True,
    )
    assert tensors['routed_experts_mask'][0].tolist() == [False, False, True, True, True, True, False, False, False]
    assert tensors['routed_experts'][0, 2:6, 0].tolist() == [[1, 2], [3, 4], [5, 6], [7, 8]]
    session['routed_experts_mask'][2] = False
    with pytest.raises(ValueError, match='routing'):
        api.expand_session_rows([[session]], torch.tensor([1.]), torch.tensor([.7]),
            {'uid': np.array(['g'], dtype=object)}, prompt_length=4, response_length=5,
            pad_token_id=99, require_routes=True)


def test_missing_logprobs_overlength_and_zero_policy_sessions_fail(api):
    for change, message in [({'response_logprobs': None}, 'log'),
                             ({'prompt_ids': [1, 2, 3, 4, 5]}, 'prompt'),
                             ({'response_mask': [0, 0, 0]}, 'policy')]:
        session = _session(0)
        session.update(change)
        with pytest.raises(ValueError, match=message):
            api.expand_session_rows([[session]], torch.tensor([1.]), torch.tensor([.7]),
                {'uid': np.array(['g'], dtype=object)}, prompt_length=4, response_length=5,
                pad_token_id=99, require_routes=False)


def test_actor_batch_size_counts_sessions_without_padding_in_denominator(api):
    assert api.actor_batch_parameters(12, 10) == {
        'mini_batch_size': 12, 'global_batch_size': 10, 'epochs': 1,
    }
    assert api.actor_batch_parameters(208, 201)['mini_batch_size'] == 208
    with pytest.raises(ValueError):
        api.actor_batch_parameters(10, 12)
