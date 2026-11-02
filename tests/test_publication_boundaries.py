"""CPU regressions for deployment inputs and optional backend adapters."""
import copy
import importlib.util
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / 'axion/training/rl'


def _load(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def preprocessors(monkeypatch):
    # Import the real preprocessing modules without loading GPU/model registries.
    # The model-template and remote-cache helpers are not used by these rows.
    for name in ('swift', 'swift.dataset', 'swift.dataset.preprocessor', 'modelscope',
                 'modelscope.hub', 'modelscope.hub.utils'):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    def unused(*args, **kwargs):
        raise AssertionError('unexpected model/storage helper call')
    monkeypatch.setitem(sys.modules, 'swift.template', SimpleNamespace(history_to_messages=unused))
    monkeypatch.setitem(sys.modules, 'swift.utils', SimpleNamespace(
        get_logger=logging.getLogger, is_dist=unused, is_master=unused, safe_ddp_context=unused))
    monkeypatch.setitem(sys.modules, 'modelscope.hub.utils.utils', SimpleNamespace(get_cache_dir=unused))
    directory = ROOT / 'third_party/swift/swift/dataset/preprocessor'
    core = _load('swift.dataset.preprocessor.core', directory / 'core.py', monkeypatch)
    adapter = _load('swift.dataset.preprocessor.aip', directory / 'aip.py', monkeypatch)
    return core, adapter


def test_standard_messages_keep_roles_content_loss_and_cache_version(preprocessors):
    core, adapter = preprocessors
    adapter.configure_joint_lm_training(False)
    auto = core.AutoPreprocessor()
    assert auto.preprocessor_version == '1'
    row = {'messages': [{'role': 'user', 'content': 'Calculate 2 + 3.'},
                        {'role': 'assistant', 'content': '5', 'loss': True}],
           'total_tokens': 12, 'visual_tokens': 0}
    selected = auto._get_preprocessor(SimpleNamespace(features=row))
    assert isinstance(selected, core.MessagesPreprocessor)
    assert selected.preprocess(copy.deepcopy(row)) == row


@pytest.mark.parametrize('features', [
    {'frame_tos_path': None}, {'messages': None, 'ts_score': None},
    {'conversations': None, 'ts_score_norm': None},
])
def test_unsupported_schema_fails_before_mapping_instead_of_dropping_labels(preprocessors, features):
    core, _ = preprocessors
    with pytest.raises(NotImplementedError, match='not included'):
        core.AutoPreprocessor()._get_preprocessor(SimpleNamespace(features=features))


def test_unsupported_joint_objective_cannot_silently_change_cache_or_loss(preprocessors):
    core, adapter = preprocessors
    with pytest.raises(NotImplementedError, match='not included'):
        adapter.configure_joint_lm_training(True)
    assert core.AutoPreprocessor().preprocessor_version == '1'


def _sandbox(monkeypatch, root):
    monkeypatch.setenv('SANDBOX_RO_ROOT', str(root))
    monkeypatch.delenv('SANDBOX_ALLOW', raising=False)
    return _load('sandbox_publication_test', RL / 'tools/local_sandbox_server.py', monkeypatch)


@pytest.mark.parametrize('root', ['', 'relative/path', '/nonexistent-axion-test-root'])
def test_sandbox_rejects_missing_or_invalid_storage_roots_before_serving(monkeypatch, root):
    with pytest.raises(ValueError, match='SANDBOX_RO_ROOT'):
        _sandbox(monkeypatch, root)


def test_sandbox_quotes_multiple_roots_and_retains_namespace_and_peer_limits(monkeypatch, tmp_path):
    first = tmp_path / "shared data's files"
    second = tmp_path / 'checkpoints'
    first.mkdir()
    second.mkdir()
    server = _sandbox(monkeypatch, f'{first},{second}')
    monkeypatch.setattr(server.shutil, 'which', lambda name: '/usr/bin/unshare')
    command = server._build_cmd('/tmp/main.py')
    assert command[:5] == ['unshare', '--user', '--map-root-user', '--mount', '--net']
    # Interpret generated shell arguments: paths containing spaces/quotes must
    # remain one mount operand, not become extra commands or separate operands.
    tokens = shlex.split(command[-1])
    for root in (first, second):
        assert str(root) in tokens
    assert tokens.count('remount,ro,bind') == 2
    assert server._client_allowed('127.0.0.1')
    assert server._client_allowed('::ffff:127.0.0.1')
    assert not server._client_allowed('192.0.2.1')
    monkeypatch.setenv('SANDBOX_ALLOW', '192.0.2.0/24')
    server.ALLOW_NETS = server._parse_allow()
    assert server._client_allowed('192.0.2.1')
    assert not server._client_allowed('198.51.100.1')
    monkeypatch.setenv('SANDBOX_ALLOW', '0.0.0.0/0')
    with pytest.raises(SystemExit):
        server._parse_allow()


def test_ray_environment_preserves_training_overrides_and_strategy_specific_gpu_settings(monkeypatch):
    from verl.trainer import constants_ppo as runtime
    monkeypatch.setenv('VERL_NCCL_TIMEOUT_SEC', '3600')
    monkeypatch.setenv('VERL_FULL_DETERMINISM', '1')
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.delenv('CUDA_DEVICE_MAX_CONNECTIONS', raising=False)
    monkeypatch.setattr(runtime, '_is_hopper_or_ampere', True)
    config = OmegaConf.create({'actor_rollout_ref': {'actor': {'strategy': 'megatron'}}})
    result = runtime.get_ppo_ray_runtime_env(config)['env_vars']
    assert result['VERL_NCCL_TIMEOUT_SEC'] == '3600'
    assert result['VERL_FULL_DETERMINISM'] == '1'
    assert result['WANDB_MODE'] == 'offline'
    assert result['CUDA_DEVICE_MAX_CONNECTIONS'] == '1'
    config.actor_rollout_ref.actor.strategy = 'fsdp2'
    assert 'CUDA_DEVICE_MAX_CONNECTIONS' not in runtime.get_ppo_ray_runtime_env(config)['env_vars']


@pytest.mark.parametrize('port,reserved,busy,message', [
    ('invalid', '', '', 'must be between'),
    ('65536', '', '', 'must be between'),
    ('18446744073709560716', '', '', 'must be between'),
    ('20001', '', '', 'reserved range'),
    ('9102', '9100,9102', '', 'reserved range'),
    ('09102', '09102', '', 'reserved range'),
    ('9102', '', '9102', 'already in use'),
])
def test_sandbox_launcher_rejects_conflicting_ports_without_starting_a_service(
        port, reserved, busy, message, tmp_path):
    # Isolate process/network boundaries; never touch existing host services.
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    commands = {
        'ps': 'exit 0', 'sleep': 'exit 0',
        'ss': f'echo "LISTEN 0 10 127.0.0.1:{busy or 1} 0.0.0.0:*"',
        'setsid': 'echo UNEXPECTED_SERVICE_START >&2; exit 1',
        'curl': 'exit 1',
    }
    for name, body in commands.items():
        command = binaries / name
        command.write_text('#!/bin/sh\n' + body + '\n')
        command.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               SERVER_PY=sys.executable, SANDBOX_RO_ROOT=str(tmp_path), SANDBOX_ALLOW='',
               SANDBOX_PORT=port, SANDBOX_RESERVED_PORTS=reserved,
               PYTHON_NODES_CONF=str(tmp_path / 'nodes.conf'))
    result = subprocess.run(['bash', str(RL / 'tools/start_local_sandbox.sh')],
                            env=env, cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert message in result.stderr
    assert 'UNEXPECTED_SERVICE_START' not in result.stderr
    assert not (tmp_path / 'nodes.conf').exists()
