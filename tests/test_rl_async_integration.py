"""CPU integration at the real DataProto and vendor method boundaries.

GPU/Ray worker construction is deliberately not run. The isolated vendor methods
below are compiled unchanged from their source and exercised with real batches;
only the external RPC worker boundary is replaced with a CPU receiver.
"""
import ast
import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from axion.training.rl.async_batch import AxionAsyncBatchAdapter

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / 'third_party/verl/verl'


def _method(path, class_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    method.decorator_list = []
    code = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), 'exec')
    exec(code, namespace)
    return namespace[name]


def _config():
    return OmegaConf.create(dict(
        async_training=dict(axion_recipe=True, require_batches=1, trigger_parameter_sync_step=2, staleness_threshold=11),
        trainer=dict(resume_mode='disable', total_training_steps=4, critic_warmup=0, balance_batch=False),
        algorithm=dict(adv_estimator='grpo', use_kl_in_reward=False, rollout_correction=dict(bypass_mode=True)),
        actor_rollout_ref=dict(
            actor=dict(ppo_mini_batch_size=24, ppo_epochs=1, loss_agg_mode='seq-mean-token-mean',
                       use_dynamic_bsz=True, calculate_entropy=False, entropy_coeff=0.,
                       data_loader_seed=42, shuffle=True,
                       megatron=dict(context_parallel_size=1, override_transformer_config=dict(moe_aux_loss_coeff=0., moe_z_loss_coeff=0.))),
            rollout=dict(n=8, prompt_length=4, response_length=5, temperature=1., multi_turn=dict(enable=True)),
        ),
    ))


def _array(values):
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arr


def _group(uid, extra_sessions=False):
    from verl import DataProto
    def session(sid):
        return dict(session_id=sid, prompt_ids=[11, 12], response_ids=[41, 42, 43], response_mask=[1, 0, 1],
                    response_logprobs=[-.1, 0., -.2],
                    routed_experts=torch.ones(4, 1, 2, dtype=torch.uint8),
                    routed_experts_mask=[True, True, True, True, False])
    sessions = [[session(0)] for _ in range(8)]
    if extra_sessions:
        sessions[-1].extend([session(1), session(2)])
    scores = torch.zeros(8, 5)
    scores[4:, 2] = 1.
    return DataProto.from_dict(tensors={'rm_scores': scores}, non_tensors={
        'uid': _array([uid] * 8), 'session_outputs': _array(sessions),
        'min_global_steps': _array([0] * 8), 'max_global_steps': _array([0] * 8),
        'drop_from_training': np.zeros(8, dtype=bool),
    })


def test_real_dataproto_batch_carries_normalized_trajectories_to_masked_sessions():
    from verl import DataProto
    adapter = AxionAsyncBatchAdapter(_config())
    batches = [_group(f'g{i}', extra_sessions=i == 0) for i in range(24)]
    for batch in batches:
        assert adapter.classify(SimpleNamespace(full_batch=batch), current_version=0) is None
    trainer = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=99), actor_rollout_wg='actor', ref_policy_wg='ref',
        use_reference_policy=True, _get_dp_size=lambda group, role: 8 if group == 'actor' else 4,
        metrics={},
    )
    expanded = adapter.expand(DataProto.concat(batches), trainer)
    assert len(expanded) == 200
    assert expanded.meta_info['axion_real_sessions'] == 194
    assert expanded.non_tensor_batch['is_padding'].sum() == 6
    assert expanded.batch['rm_scores'].sum().item() == 98.
    trainer.reward_tensor = expanded.batch['rm_scores']
    trainer.reward_extra_infos_dict = {}
    expanded = adapter.compute_advantage(trainer, expanded)
    assert torch.all(expanded.batch['advantages'][expanded.batch['response_mask'] == 0] == 0)
    assert 'session_outputs' not in expanded.non_tensor_batch
    chosen = expanded.non_tensor_batch['trajectory_id'] == 'g0:7'
    assert expanded.batch['advantages'][chosen, 0].tolist() == pytest.approx([.9354126] * 3, abs=1e-5)
    assert expanded.batch['advantages'][-6:].sum().item() == 0.


def test_failed_judge_discards_entire_group_even_with_reward_variation():
    adapter = AxionAsyncBatchAdapter(_config())
    batch = _group('failed')
    batch.non_tensor_batch['drop_from_training'][0] = True
    assert adapter.classify(SimpleNamespace(full_batch=batch), 0) == 'reward_failure'
    assert 'trajectory_advantage' not in batch.batch


def test_resume_is_rejected_instead_of_skipping_uncheckpointed_trajectories():
    config = _config()
    config.trainer.resume_mode = 'auto'
    with pytest.raises(ValueError, match='fresh-start'):
        AxionAsyncBatchAdapter(config)
    config.trainer.resume_mode = 'disable'
    config.trainer.total_training_steps = 3
    with pytest.raises(ValueError, match='even'):
        AxionAsyncBatchAdapter(config)


def test_vendor_clock_publishes_only_after_each_two_optimizer_updates():
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_trainer.py',
                     'FullyAsyncTrainer', '_fit_update_local_step', {'datetime': datetime})
    trainer = SimpleNamespace(axion_batch_adapter=object(), optimizer_updates=0, global_steps=1,
                              local_trigger_step=1, trigger_parameter_sync_step=2, current_param_version=0)
    observed = []
    for _ in range(4):
        method(trainer)
        observed.append((trainer.optimizer_updates, trainer.local_trigger_step, trainer.current_param_version))
        trainer.global_steps += 1
    assert observed == [(1, 2, 0), (2, 1, 1), (3, 2, 1), (4, 1, 2)]


def test_vendor_backpressure_releases_rejected_groups_before_next_sync():
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
                     'FullyAsyncRollouter', '_should_pause_generation', {})
    class Queue:
        size = 0
        async def get_statistics(self):
            return {'queue_size': self.size}
    queue = Queue()
    state = SimpleNamespace(message_queue_client=queue, max_queue_size=576, paused=False,
                            axion_recipe=True, active_tasks=set(), staleness_samples=600, max_required_samples=576)
    async def run():
        assert await method(state) is False
        queue.size = 576
        assert await method(state) is True
        queue.size = 0
        state.axion_recipe = False
        assert await method(state) is True
    asyncio.run(run())


def test_vendor_actor_rpc_uses_one_session_minibatch_and_real_session_normalizer():
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.utils.py_functional import rename_dict
    adapter = AxionAsyncBatchAdapter(_config())
    raw = _group('g', extra_sessions=True)
    adapter.classify(SimpleNamespace(full_batch=raw), 0)
    trainer = SimpleNamespace(config=_config(), tokenizer=SimpleNamespace(pad_token_id=99),
        use_reference_policy=False, actor_rollout_wg=None, _get_dp_size=lambda group, role: 4, metrics={},
        distillation_config=None)
    batch = adapter.expand(raw, trainer)
    trainer.reward_tensor = batch.batch['rm_scores']
    trainer.reward_extra_infos_dict = {}
    adapter.compute_advantage(trainer, batch)
    batch.batch['old_log_probs'] = batch.batch['rollout_log_probs'].clone()
    observations = []
    class Receiver:
        def update_actor(self, data):
            observations.append((len(data), tu.get(data, 'mini_batch_size'), tu.get(data, 'global_batch_size'), tu.get(data, 'epochs')))
            # This is the external GPU/RPC boundary; the input preparation above
            # and TensorDict conversion below use the actual library code.
            return tu.get_tensordict({}, {'metrics': {'mfu': 0.}})
    trainer.actor_rollout_wg = Receiver()
    method = _method(VERL / 'trainer/ppo/ray_trainer.py', 'RayPPOTrainer', '_update_actor', dict(
        DataProto=DataProto, tu=tu,
        # GPU padding imports FlashAttention; isolate that hardware boundary.
        # This test verifies optimizer batching metadata, not CUDA unpadding.
        left_right_2_no_padding=lambda data: data,
        _strip_actor_mm_from_update=lambda data: False,
        is_distillation_enabled=lambda config: False, rename_dict=rename_dict,
    ))
    result = method(trainer, batch)
    assert observations == [(12, 12, 10, 1)]
    assert result.meta_info['metrics']['perf/mfu/actor'] == 0.


def test_attempt_budget_cycles_dataset_instead_of_epoch_one_shortfall():
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
                     'FullyAsyncRollouter', '_create_continuous_iterator', {})
    state = SimpleNamespace(axion_recipe=True, total_rollout_steps=7,
                            train_dataloader=['a', 'b'], config=SimpleNamespace(trainer=SimpleNamespace(total_epochs=1)))
    assert list(method(state)) == [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b'),
                                   (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]
    state.axion_recipe = False
    assert list(method(state)) == [(0, 'a'), (0, 'b')]


def test_checkpoint_gate_skips_unsynchronized_optimizer_state():
    from contextlib import nullcontext
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_trainer.py',
        'FullyAsyncTrainer', '_fit_save_checkpoint', dict(
            should_save_ckpt_esi=lambda **kwargs: False,
            marked_timer=lambda *args, **kwargs: nullcontext(),
        ))
    saved = []
    state = SimpleNamespace(axion_batch_adapter=object(), local_trigger_step=2,
        current_param_version=1, last_ckpt_version=0, timing_raw={}, max_steps_duration=0,
        config=SimpleNamespace(trainer=SimpleNamespace(esi_redundant_time=0, save_freq=5)),
        _save_checkpoint=lambda: saved.append('saved'))
    method(state, force=True)
    assert saved == []
    state.local_trigger_step = 1
    method(state, force=True)
    assert saved == ['saved'] and state.last_ckpt_version == 1


def test_stop_generation_cancels_pending_work_after_even_update_target():
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
                     'FullyAsyncRollouter', 'stop_generation', {'asyncio': asyncio})
    async def run():
        event = asyncio.Event()
        tasks = [asyncio.create_task(event.wait()) for _ in range(3)]
        state = SimpleNamespace(_stop_requested=False, running=True, _resume_event=asyncio.Event(),
            feed_task=tasks[0], processor_task=tasks[1], active_tasks={tasks[2]})
        await method(state)
        assert all(task.cancelled() for task in tasks)
        assert state._stop_requested and not state.running and state._resume_event.is_set()
    asyncio.run(run())


def test_static_actor_and_reference_microbatches_both_constrain_padding():
    config = _config()
    config.actor_rollout_ref.actor.use_dynamic_bsz = False
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 3
    config.actor_rollout_ref.ref = dict(log_prob_use_dynamic_bsz=False, log_prob_micro_batch_size_per_gpu=5)
    adapter = AxionAsyncBatchAdapter(config)
    batch = _group('g', extra_sessions=True)
    adapter.classify(SimpleNamespace(full_batch=batch), 0)
    trainer = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=99), actor_rollout_wg='actor',
        ref_policy_wg='ref', use_reference_policy=True,
        _get_dp_size=lambda group, role: 4 if group == 'actor' else 2, metrics={})
    expanded = adapter.expand(batch, trainer)
    # Actor rows must divide 4*3, reference rows 2*5: lcm=60.
    assert len(expanded) == 60
    assert expanded.meta_info['axion_real_sessions'] == 10
    assert expanded.batch['response_mask'][10:].sum() == 0


def test_disabled_validation_dataloader_does_not_recreate_empty_source():
    from typing import Optional
    from torch.utils.data import TensorDataset, SequentialSampler, Sampler, default_collate
    from torchdata.stateful_dataloader import StatefulDataLoader
    from omegaconf import open_dict
    config = _config()
    config.data = dict(train_files=['train.parquet'], val_files=[], gen_batch_size=1,
                       train_batch_size=0, dataloader_num_workers=0, val_batch_size=None)
    config.trainer.val_before_train = False
    config.trainer.test_freq = -1
    config.trainer.total_epochs = 1
    dataset = TensorDataset(torch.arange(3))
    def load_dataset(paths, *args, **kwargs):
        assert paths, 'Disabled validation must not read an empty validation source'
        return dataset
    method = _method(VERL / 'trainer/ppo/ray_trainer.py', 'RayPPOTrainer', '_create_dataloader', dict(
        Optional=Optional, Sampler=Sampler, create_rl_dataset=load_dataset,
        StatefulDataLoader=StatefulDataLoader, OmegaConf=OmegaConf, open_dict=open_dict,
    ))
    state = SimpleNamespace(config=config, tokenizer=None, processor=None)
    method(state, dataset, None, default_collate, SequentialSampler(dataset))
    assert state.val_dataset is None and state.val_dataloader is None
    assert [batch[0].item() for batch in state.train_dataloader] == [0, 1, 2]
    assert state.total_training_steps == 4


@pytest.mark.parametrize('val_files, before_train, test_freq, expected_sources', [
    ([], False, -1, [['train.parquet']]),
    (['val.parquet'], False, 5, [['train.parquet'], ['val.parquet']]),
])
def test_rollouter_constructs_only_requested_dataset_sources(val_files, before_train, test_freq, expected_sources):
    from torch.utils.data import TensorDataset
    config = _config()
    config.data = dict(train_files=['train.parquet'], val_files=val_files)
    config.trainer.val_before_train = before_train
    config.trainer.test_freq = test_freq
    sources = []
    def load_dataset(paths, *args, **kwargs):
        assert paths
        sources.append(list(paths))
        return TensorDataset(torch.arange(2))
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
        'FullyAsyncRollouter', '_create_rollout_datasets', {'create_rl_dataset': load_dataset})
    state = SimpleNamespace(config=config, tokenizer=None, processor=None)
    train, val = method(state)
    assert sources == expected_sources and len(train) == 2
    assert (val is None) == (not val_files)


@pytest.mark.parametrize('before_train,test_freq', [(True, -1), (False, 1)])
def test_explicit_validation_without_data_fails_before_dataset_work(before_train, test_freq):
    config = _config()
    config.data = dict(train_files=['train.parquet'], val_files=[])
    config.trainer.val_before_train = before_train
    config.trainer.test_freq = test_freq
    def forbidden_dataset_read(*args, **kwargs):
        raise AssertionError('Configuration validation must precede dataset work')
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
        'FullyAsyncRollouter', '_create_rollout_datasets', {'create_rl_dataset': forbidden_dataset_read})
    with pytest.raises(ValueError, match='validation'):
        method(SimpleNamespace(config=config, tokenizer=None, processor=None))


def test_direct_validation_rpc_rejects_missing_loader():
    method = _method(VERL / 'experimental/fully_async_policy/fully_async_rollouter.py',
        'FullyAsyncRollouter', 'do_validate', {})
    with pytest.raises(ValueError, match='validation'):
        method(SimpleNamespace(val_dataloader=None))
