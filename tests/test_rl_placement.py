"""Launch validation against CPU-only deployment and Ray resource snapshots."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location('placement_launcher_test', ROOT / 'axion/training/rl/launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deployment():
    return {
        'context_threshold': 48000, 'max_assistant_turns': 50,
        'max_turn_response_length': 49152, 'max_tool_response_length': 12000,
        'judge': {'base_url': 'http://judge.example/v1', 'model': 'Qwen3.8-27B', 'api_key_env': 'AXION_TEST_KEY'},
        'summarizer': {'base_url': 'http://summary.example/v1', 'model': 'Qwen3.5-9B', 'api_key_env': 'AXION_TEST_KEY'},
        'condenser': {'base_url': 'http://memory.example/v1', 'model': 'Qwen3.5-9B', 'api_key_env': 'AXION_TEST_KEY'},
        'search_url': 'http://search.example:8000', 'scholar_url': 'http://scholar.example:8001',
        'python_url': 'http://python.example:9000/run_code',
    }


@pytest.mark.parametrize('field,value', [
    ('trainer_nodes', 0), ('rollout_nodes', False), ('gpus_per_node', 1.5),
    ('trainer_tensor_parallel_size', 'two'), ('trainer_expert_parallel_size', -1),
    ('rollout_tensor_parallel_size', 0), ('judge', None), ('judge', []), ('python_url', 9000),
])
def test_invalid_deployment_fails_before_worker_launch(tmp_path, field, value):
    cfg = deployment()
    cfg[field] = value
    path = tmp_path / 'deployment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=field):
        launcher().read_deployment(path)


@pytest.mark.parametrize('field,value', [
    ('api_key_env', 123), ('api_key_env', 'BAD KEY'), ('base_url', 'http://judge.example:99999/v1'),
    ('base_url', 'http://judge.example/v1#fragment'),
])
def test_invalid_service_fields_are_rejected(tmp_path, field, value):
    cfg = deployment()
    cfg['judge'][field] = value
    path = tmp_path / 'deployment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='judge'):
        launcher().read_deployment(path)


@pytest.mark.parametrize('overrides', [
    {'rollout_nodes': 1, 'rollout_tensor_parallel_size': 16},
    {'rollout_nodes': 1, 'rollout_tensor_parallel_size': 3},
    {'trainer_nodes': 1, 'gpus_per_node': 4},  # default EP8 cannot fit world size 4
    {'trainer_tensor_parallel_size': 3},
    {'trainer_nodes': 3, 'trainer_expert_parallel_size': 3},  # EP must divide 256 experts
])
def test_parallelism_rejects_incompatible_groups(tmp_path, overrides):
    cfg = deployment()
    cfg.update(overrides)
    path = tmp_path / 'deployment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='parallel|GPUs|experts'):
        launcher().read_deployment(path)


def config(*, trainer_nodes=1, rollout_nodes=1, gpus=4, rollout_tp=4):
    return SimpleNamespace(
        trainer=SimpleNamespace(nnodes=trainer_nodes, n_gpus_per_node=gpus),
        rollout=SimpleNamespace(nnodes=rollout_nodes, n_gpus_per_node=gpus),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(tensor_model_parallel_size=rollout_tp)))


@pytest.mark.parametrize('nodes', [
    {'only': {'GPU': 8, 'CPU': 64}},
    {'trainer': {'GPU': 4, 'CPU': 32}, 'rollout': {'GPU': 4, 'CPU': 32}},
    {'trainer': {'GPU': 4, 'CPU': 12}, 'rollout': {'GPU': 4, 'CPU': 9}, 'control': {'CPU': 23}},
])
def test_disjoint_pools_can_share_one_host_or_use_separate_hosts(nodes):
    launcher().validate_cluster_resources(config(), nodes)


@pytest.mark.parametrize('cfg,nodes', [
    (config(), {'only': {'GPU': 4, 'CPU': 64}}),  # missing rollout pool
    (config(gpus=8), {str(i): {'GPU': 4, 'CPU': 64} for i in range(4)}),  # STRICT_PACK fragmentation
    (config(), {'only': {'GPU': 8, 'CPU': 42}}),  # pools need 20 CPU + coordinators need 23
    (config(), {'trainer': {'GPU': 4, 'CPU': 12}, 'rollout': {'GPU': 4, 'CPU': 8}, 'head': {'CPU': 64}}),
    (config(), {'a': {'GPU': 7, 'CPU': 64}, 'b': {'GPU': 1, 'CPU': 64}}),
    (config(), {'gpu': {'GPU': 8, 'CPU': 20}, **{str(i): {'CPU': 9} for i in range(4)}}),
])
def test_impossible_capacity_or_placement_fails_before_model_loading(cfg, nodes):
    with pytest.raises(ValueError, match='Ray|GPU|CPU|placement'):
        launcher().validate_cluster_resources(cfg, nodes)


def test_paper_layout_fits_seven_eight_gpu_hosts():
    nodes = {str(i): {'GPU': 8, 'CPU': 64} for i in range(7)}
    launcher().validate_cluster_resources(config(trainer_nodes=2, rollout_nodes=5, gpus=8), nodes)


def test_cross_node_rollout_layout(tmp_path):
    cfg = deployment()
    cfg.update(gpus_per_node=8, trainer_nodes=1, rollout_nodes=2, rollout_tensor_parallel_size=16)
    path = tmp_path / 'deployment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    launcher().read_deployment(path)
    launcher().validate_cluster_resources(config(gpus=8, rollout_nodes=2, rollout_tp=16),
        {str(i): {'GPU': 8, 'CPU': 64} for i in range(3)})


def test_existing_collective_environment_is_forwarded(monkeypatch):
    monkeypatch.setenv('AXION_TEST_KEY', 'fixture-credential')
    values = {'NCCL_DEBUG': 'INFO', 'NCCL_SOCKET_IFNAME': 'eth0', 'NCCL_IB_HCA': 'mlx5_0',
              'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC': '900', 'CUDA_DEVICE_MAX_CONNECTIONS': '1'}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv('UNRELATED_PRIVATE_SETTING', 'do-not-forward')
    env = launcher().worker_environment(deployment())
    assert {key: env.get(key) for key in values} == values
    assert 'UNRELATED_PRIVATE_SETTING' not in env


def test_live_ray_preflight_rejects_before_output_or_training(tmp_path, monkeypatch, capsys):
    from axion.training.rl import launcher as command, runtime
    monkeypatch.setattr(command.os, 'environ', command.os.environ.copy())
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5_moe', 'text_config': {
        'hidden_size': 3072, 'num_hidden_layers': 48, 'num_experts': 256, 'num_experts_per_tok': 8}}))
    data = tmp_path / 'selected.parquet'
    data.write_bytes(b'PAR1-fixture')
    data.with_suffix('.selection.json').write_text(json.dumps({'attempts_per_task': 4,
        'criterion': 'at_least_one_correct_and_nonzero_reward_variance', 'selected_count': 1,
        'parquet_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}))
    cfg = deployment()
    cfg.update(trainer_nodes=1, rollout_nodes=1, gpus_per_node=4, trainer_expert_parallel_size=4)
    path = tmp_path / 'deployment.yaml'
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / 'output'
    monkeypatch.setenv('AXION_TEST_KEY', 'fixture-credential')
    monkeypatch.setattr(sys, 'argv', ['axion.training.rl', '--model', str(model), '--train-data', str(data),
        '--deployment', str(path), '--output', str(output), '--updates', '2'])
    monkeypatch.setattr(runtime, 'verify_runtime', lambda *args, **kwargs: {'gpu_validated': False})
    # Substitute only external runtime boundaries. Stale resources from a dead
    # node must not make the deployment look feasible.
    monkeypatch.setitem(sys.modules, 'ray', SimpleNamespace(
        init=lambda **kwargs: None,
        nodes=lambda: [{'NodeID': 'live', 'Alive': True}, {'NodeID': 'dead', 'Alive': False}],
        _private=SimpleNamespace(state=SimpleNamespace(available_resources_per_node=lambda: {
            'live': {'GPU': 4, 'CPU': 64}, 'dead': {'GPU': 100, 'CPU': 1000}}))))
    monkeypatch.setitem(sys.modules, 'verl.trainer.main_ppo', SimpleNamespace(get_ppo_ray_runtime_env=lambda config: {}))
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
    assert 'Ray cluster needs' in capsys.readouterr().err
    assert not output.exists()
