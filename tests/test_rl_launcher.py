"""Paper recipe composition without importing a GPU training stack."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args, cwd, extra_env=None):
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
           'AXION_TEST_SECRET': 'must-not-appear-in-plan', **(extra_env or {})}
    return subprocess.run([sys.executable, '-m', 'axion.training.rl', *args],
                          cwd=cwd, env=env, capture_output=True, text=True)


def inputs(tmp_path):
    model = tmp_path / 'model'; model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5_moe', 'text_config': {
        'hidden_size': 3072, 'num_hidden_layers': 48, 'num_experts': 256, 'num_experts_per_tok': 8}}))
    data = tmp_path / 'selected.parquet'; data.write_bytes(b'PAR1-fixture')
    deployment = tmp_path / 'deployment.yaml'
    deployment.write_text(yaml.safe_dump({
        'context_threshold': 48000, 'max_assistant_turns': 50, 'max_turn_response_length': 49152,
        'max_tool_response_length': 12000,
        'judge': {'base_url': 'http://judge.example/v1', 'model': 'Qwen3.8-27B', 'api_key_env': 'AXION_TEST_SECRET'},
        'summarizer': {'base_url': 'http://summary.example/v1', 'model': 'Qwen3.5-9B', 'api_key_env': 'AXION_TEST_SECRET'},
        'condenser': {'base_url': 'http://memory.example/v1', 'model': 'Qwen3.5-9B', 'api_key_env': 'AXION_TEST_SECRET'},
        'search_url': 'http://search.example:8000', 'scholar_url': 'http://scholar.example:8001',
        'python_url': 'http://python.example:9000/run_code',
    }))
    return ['--model', str(model), '--train-data', str(data), '--output', str(tmp_path / 'output'),
            '--updates', '20', '--deployment', str(deployment), '--dry-run'], deployment


def test_rl_help_is_available_without_importing_training_frameworks(tmp_path):
    result = run_cli(['--help'], tmp_path)
    assert result.returncode == 0, result.stderr
    assert '--updates' in result.stdout and '--deployment' in result.stdout
    result = subprocess.run([sys.executable, '-c',
        'import sys; import axion.training.rl; assert not any(x in sys.modules for x in ("torch","ray","verl"))'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'},
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_paper_recipe_is_effective_and_secrets_are_absent_from_dry_run(tmp_path):
    args, _ = inputs(tmp_path)
    result = run_cli(args, tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'must-not-appear-in-plan' not in result.stdout + result.stderr
    config = json.loads(result.stdout)['config']
    actor, rollout = config['actor_rollout_ref']['actor'], config['actor_rollout_ref']['rollout']
    assert actor['optim']['lr'] == 2e-6
    assert actor['optim']['lr_warmup_steps'] == 5
    assert actor['optim']['betas'] == [0.9, 0.999]
    assert actor['optim']['weight_decay'] == 0.1
    assert actor['optim']['lr_decay_style'] == 'constant'
    assert actor['use_kl_loss'] and actor['kl_loss_coef'] == .001
    assert actor['clip_ratio_low'] == .2 and actor['clip_ratio_high'] == .28
    assert actor['loss_agg_mode'] == 'seq-mean-token-mean'
    assert actor['ppo_mini_batch_size'] == 24 and actor['ppo_epochs'] == 1
    assert actor['megatron']['router_replay']['mode'] == 'R3'
    assert rollout['enable_rollout_routing_replay']
    assert rollout['n'] == 8 and rollout['temperature'] == 1.0
    assert rollout['agent']['default_agent_loop'] == 'deepresearch_agent'
    assert config['data']['max_prompt_length'] == 12000
    assert config['data']['max_response_length'] == 49152
    assert config['data']['filter_overlong_prompts']
    assert config['data']['custom_cls'] == {'path': 'pkg://axion.training.rl.dataset', 'name': 'PaperRLHFDataset'}
    assert config['async_training']['axion_recipe']
    assert config['async_training']['trigger_parameter_sync_step'] == 2
    assert config['async_training']['staleness_threshold'] == 11
    assert config['trainer']['total_training_steps'] == actor['optim']['total_training_steps'] == 20
    assert config['trainer']['resume_mode'] == 'disable'
    assert config['custom_reward_function']['name'] == 'compute_paper_score'
    assert config['rollout']['nnodes'] == 5 and config['trainer']['nnodes'] == 2
    assert not (tmp_path / 'output').exists()


def test_unreported_deployment_choices_must_be_explicit(tmp_path):
    args, deployment = inputs(tmp_path)
    d = yaml.safe_load(deployment.read_text()); d['context_threshold'] = None
    deployment.write_text(yaml.safe_dump(d))
    result = run_cli(args, tmp_path)
    assert result.returncode != 0 and 'context_threshold' in result.stderr


@pytest.mark.parametrize('updates', ['0', '3'])
def test_stop_count_rejects_partial_sync_intervals(tmp_path, updates):
    args, _ = inputs(tmp_path); args[args.index('--updates') + 1] = updates
    result = run_cli(args, tmp_path)
    assert result.returncode != 0 and 'even' in result.stderr.lower()


def test_service_environment_uses_only_declared_models_and_endpoints(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('rl_launcher_test', ROOT / 'axion/training/rl/launcher.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    _, path = inputs(tmp_path)
    deployment = module.read_deployment(path)
    monkeypatch.setenv('AXION_TEST_SECRET', 'fixture-credential')
    env = module.worker_environment(deployment)
    assert env['FORCE_FINAL_ANSWER'] == env['SOFTEN_ANSWER_PROMPT'] == '0'
    assert env['MEMORY_MODEL_NAME'] == 'Qwen3.5-9B'
    assert env['MEMORY_AZURE_ENDPOINT'] == ''
    assert env['MEMORY_LOCAL_FALLBACK_MODEL_NAME'] == ''
    assert env['VISIT_SUMMARY_FALLBACK_API_BASE'] == ''
    assert env['EVAL_LLM_MODEL_NAME'] == 'Qwen3.8-27B'
    assert env['OPENAI_BASE_URL'] == 'http://judge.example/v1'
    assert env['SEARCH_NODES_CONF'] == env['SCHOLAR_NODES_CONF'] == ''
    assert env['S2_JUDGE_BASE_URL'] == 'http://judge.example/v1'
    assert env['S2_JUDGE_MODEL'] == 'Qwen3.8-27B'
    assert env['S2_JUDGE_API_KEY'] == 'fixture-credential'


def test_attempt_budget_cannot_be_smaller_than_required_accepted_groups(tmp_path):
    args, _ = inputs(tmp_path)
    result = run_cli([*args, '--rollout-attempts', '1'], tmp_path)
    assert result.returncode != 0 and 'attempt' in result.stderr.lower()


def test_composed_paper_settings_are_accepted_by_real_worker_configs(tmp_path):
    import importlib.util
    from types import SimpleNamespace
    from verl.utils.config import omega_conf_to_dataclass
    spec = importlib.util.spec_from_file_location('rl_launcher_schema_test', ROOT / 'axion/training/rl/launcher.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    args, path = inputs(tmp_path)
    cfg, _ = module.build_config(SimpleNamespace(updates=20, rollout_attempts=None,
        model=args[1], train_data=args[3], output=args[5], deployment=str(path)))
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    assert actor.kl_loss_coef == .001 and actor.megatron.router_replay.mode == 'R3'
    assert rollout.enable_rollout_routing_replay and rollout.multi_turn.max_assistant_turns == 50


def test_session_microbatches_use_fixed_rows_without_minibatch_wide_padding(tmp_path):
    args, _ = inputs(tmp_path)
    result = run_cli(args, tmp_path)
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)['config']['actor_rollout_ref']
    assert cfg['actor']['use_dynamic_bsz'] is False
    assert cfg['actor']['ppo_micro_batch_size_per_gpu'] == 1
    assert cfg['ref']['log_prob_use_dynamic_bsz'] is False
    for role in ('actor', 'ref'):
        assert cfg[role]['megatron']['pad_bshd_to_minibatch_max'] is False

    # Exercise the actual backend splitter without importing GPU engine plugins.
    import ast
    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    source = ROOT / 'third_party/verl/verl/workers/engine/utils.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_micro_batches')
    namespace = {'TensorDict': TensorDict, 'tu': tu}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    lengths = [60000] + [100] * 23
    ids = torch.nested.nested_tensor([torch.full((n,), i) for i, n in enumerate(lengths)], layout=torch.jagged)
    data = TensorDict({'input_ids': ids}, batch_size=[len(lengths)])
    tu.assign_non_tensor(data, use_dynamic_bsz=cfg['actor']['use_dynamic_bsz'],
                         micro_batch_size_per_gpu=cfg['actor']['ppo_micro_batch_size_per_gpu'])
    batches, indices = namespace['prepare_micro_batches'](data)
    assert len(batches) == len(lengths) and indices is None
    for i, (batch, length) in enumerate(zip(batches, lengths)):
        assert len(batch) == 1
        assert batch['input_ids'].unbind()[0].tolist() == [i] * length
