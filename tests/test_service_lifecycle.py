"""Service lifecycle checks without starting services or contacting endpoints."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RL = ROOT / 'axion/training/rl'


def sandbox(monkeypatch, tmp_path):
    monkeypatch.setenv('SANDBOX_RO_ROOT', str(tmp_path))
    monkeypatch.setenv('SANDBOX_ISOLATE', '1')
    spec = importlib.util.spec_from_file_location('sandbox_lifecycle', RL / 'tools/local_sandbox_server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell(script, tmp_path, hooks, *args, **env):
    bashenv = tmp_path / 'hooks.sh'
    bashenv.write_text(hooks)
    return subprocess.run(['bash', str(RL / script), *args], cwd=tmp_path,
                          env={**os.environ, 'BASH_ENV': str(bashenv),
                               'SERVER_PY': sys.executable, 'SANDBOX_RO_ROOT': str(tmp_path), **env},
                          capture_output=True, text=True, timeout=15)


def test_requested_isolation_never_executes_without_unshare(monkeypatch, tmp_path):
    server = sandbox(monkeypatch, tmp_path)
    monkeypatch.setattr(server.shutil, 'which', lambda _: None)
    launched = []
    def popen(*args, **kwargs):
        launched.append(args)
        raise AssertionError('must refuse before starting an unisolated process')
    monkeypatch.setattr(server.subprocess, 'Popen', popen)
    result = server.run_python("print('unsafe')", 1, None)
    assert not launched
    assert result['status'] == 'Error'
    assert 'unshare' in result['stderr']


def test_sandbox_quotes_execution_paths(monkeypatch, tmp_path):
    server = sandbox(monkeypatch, tmp_path)
    monkeypatch.setattr(server.shutil, 'which', lambda _: '/usr/bin/unshare')
    server.PYTHON_BIN = "/tmp/python environment's/bin/python"
    script = "/tmp/work directory's/main.py"
    tokens = shlex.split(server._build_cmd(script)[-1])
    assert server.PYTHON_BIN in tokens
    assert script in tokens


def test_invalid_sandbox_launch_does_not_stop_other_instances(tmp_path):
    result = shell('tools/start_local_sandbox.sh', tmp_path, '''
ps() { echo '999991 python local_sandbox_server.py --port 9991'; }
kill() { echo INTERCEPTED_KILL; }
sleep() { :; }
''', SANDBOX_PORT='invalid')
    assert result.returncode != 0
    assert 'INTERCEPTED_KILL' not in result.stdout


def verification_hooks(tmp_path, scenario):
    # Execute generated probes with controlled OS/network errors; only HTTP is
    # replaced. This catches a probe accepting ENOENT as read-only or a refused
    # connection as isolation, rather than prescribing its source text.
    fake = tmp_path / 'fake_curl.py'
    fake.write_text('''import builtins, contextlib, errno, io, json, os, socket, sys, tempfile
args = sys.argv[1:]
code = json.loads(args[args.index('-d') + 1])['code']
scenario = os.environ['PROBE_SCENARIO']
if scenario == 'dead' and 'http://127.0.0.1:9991/run_code' in args:
    print('{}'); raise SystemExit
if 'sympy' in code:
    rr = dict(status='Finished', stdout='True', stderr='', return_code=0)
elif 'while True' in code:
    rr = dict(status='Error' if scenario == 'timeout' else 'TimeLimitExceeded', stdout='', stderr='', return_code=-9)
else:
    def denied(*a, **kw):
        number = errno.ENOENT if scenario == 'missing_root' else errno.EROFS
        raise OSError(number, os.strerror(number))
    def blocked(*a, **kw):
        number = errno.ECONNREFUSED if scenario == 'refused_network' else errno.ENETUNREACH
        raise OSError(number, os.strerror(number))
    builtins.open = denied
    tempfile.mkstemp = denied
    socket.create_connection = blocked
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output): exec(code, {})
        rr = dict(status='Finished', stdout=output.getvalue(), stderr='', return_code=0)
    except Exception as error:
        rr = dict(status='Error', stdout=output.getvalue(), stderr=str(error), return_code=1)
print(json.dumps(dict(status='Success', message='', compile_result=None, run_result=rr, files={})))
''')
    return f'''
curl() {{ {shlex.quote(sys.executable)} {shlex.quote(str(fake))} "$@"; }}
python3() {{
  if [[ "${{1:-}}" == '-' && "${{2:-}}" == http* ]]; then
    cat >/dev/null
    [[ "$PROBE_SCENARIO" != concurrency ]]
    return
  fi
  command python3 "$@"
}}
'''


@pytest.mark.parametrize('scenario', ['missing_root', 'refused_network', 'timeout', 'concurrency', 'dead'])
def test_verification_failure_has_nonzero_exit_without_changing_config(tmp_path, scenario):
    conf = tmp_path / 'nodes.conf'
    initial = 'endpoint=http://127.0.0.1:9991/run_code\n'
    conf.write_text(initial)
    result = shell('tools/verify_sandbox.sh', tmp_path, verification_hooks(tmp_path, scenario),
                   PYTHON_NODES_CONF=str(conf), PROBE_SCENARIO=scenario)
    assert result.returncode != 0, result.stdout + result.stderr
    assert conf.read_text() == initial
    if scenario == 'missing_root':
        assert '  storage read-only\n' not in result.stdout
    if scenario == 'refused_network':
        assert 'network cut off' not in result.stdout


def test_verified_sandbox_succeeds(tmp_path):
    conf = tmp_path / 'nodes.conf'
    conf.write_text('endpoint=http://127.0.0.1:9991/run_code\n')
    result = shell('tools/verify_sandbox.sh', tmp_path, verification_hooks(tmp_path, 'ok'),
                   PYTHON_NODES_CONF=str(conf), PROBE_SCENARIO='ok')
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'storage read-only' in result.stdout
    assert 'network cut off' in result.stdout


def test_prune_preserves_comments_and_other_endpoints(tmp_path):
    conf = tmp_path / 'nodes.conf'
    initial = ('# example http://127.0.0.1:9991/run_code\n'
               'endpoints=http://127.0.0.1:9991/run_code,http://127.0.0.1:9992/run_code\n')
    conf.write_text(initial)
    result = shell('tools/verify_sandbox.sh', tmp_path, verification_hooks(tmp_path, 'dead'), '--prune',
                   PYTHON_NODES_CONF=str(conf), PROBE_SCENARIO='dead')
    assert result.returncode != 0
    assert conf.read_text() == ('# example http://127.0.0.1:9991/run_code\n'
                                'endpoints=http://127.0.0.1:9992/run_code\n')


@pytest.mark.parametrize('scenario,successful', [('missing_root', False), ('ok', True)])
def test_startup_requires_real_storage_denial_and_cleans_up_failed_launch(tmp_path, scenario, successful):
    server_py = tmp_path / 'server python'
    server_py.write_text('#!' + sys.executable + '\n' + '''
import builtins, contextlib, errno, io, json, os, sys, tempfile, urllib.request
from types import SimpleNamespace
source=sys.stdin.read()
sys.argv=sys.argv[1:]
if 'PY_CONFIG' not in source and 'from local_sandbox_server import' not in source:
    def request(req, **kwargs):
        code=json.loads(req.data)['code']
        if 'sympy' in code:
            rr=dict(status='Finished', stdout='True', stderr='', return_code=0)
        elif 'scipy' in code:
            rr=dict(status='Finished', stdout='EXTRA_OK', stderr='', return_code=0)
        else:
            def denied(*a, **kw):
                number=errno.ENOENT if os.environ['PROBE_SCENARIO']=='missing_root' else errno.EROFS
                raise OSError(number, os.strerror(number))
            old_open, old_mkstemp = builtins.open, tempfile.mkstemp
            builtins.open, tempfile.mkstemp = denied, denied
            output=io.StringIO()
            try:
                with contextlib.redirect_stdout(output): exec(code, {})
                rr=dict(status='Finished', stdout=output.getvalue(), stderr='', return_code=0)
            except Exception as exc:
                rr=dict(status='Error', stdout=output.getvalue(), stderr=str(exc), return_code=1)
            finally:
                builtins.open, tempfile.mkstemp = old_open, old_mkstemp
        return io.BytesIO(json.dumps(dict(run_result=rr)).encode())
    urllib.request.build_opener=lambda *args: SimpleNamespace(open=request)
exec(compile(source, '<launcher>', 'exec'))
''')
    server_py.chmod(0o755)
    log = tmp_path / 'signals'
    conf = tmp_path / 'nodes.conf'
    hooks = f'''
ss() {{ :; }}
setsid() {{ :; }}
curl() {{ return 0; }}
sleep() {{ :; }}
kill() {{ printf '%s\\n' "$*" >> {shlex.quote(str(log))}; return 0; }}
'''
    # Override the interpreter used only at the process boundary; the launcher
    # and its Python smoke-check logic still execute unchanged.
    result = shell('tools/start_local_sandbox.sh', tmp_path, hooks,
                   SERVER_PY=str(server_py), PYTHON_NODES_CONF=str(conf), SANDBOX_PORT='9107',
                   SANDBOX_LOG=str(tmp_path / 'server.log'), PROBE_SCENARIO=scenario)
    assert (result.returncode == 0) is successful, result.stdout + result.stderr
    assert conf.exists() is successful
    assert log.exists() is not successful


def vllm_environment(tmp_path, **overrides):
    return dict(EVAL_VLLM_MODEL_PATH=str(tmp_path), EVAL_VLLM_SERVED_MODEL_NAME='expected',
                EVAL_VLLM_PORTS='9991', EVAL_VLLM_GPU_GROUPS='0',
                EVAL_VLLM_TENSOR_PARALLEL_SIZES='1', EVAL_VLLM_MAX_MODEL_LEN='1024',
                EVAL_VLLM_GPU_MEMORY_UTILIZATION='.8', EVAL_VLLM_WAIT_RETRIES='1',
                EVAL_VLLM_WAIT_INTERVAL_SECONDS='0', EVAL_VLLM_START_RETRIES='2', **overrides)


@pytest.mark.parametrize('mode', ['0', '1'])
def test_vllm_retries_rejected_readiness_without_global_process_kill(tmp_path, mode):
    launches = tmp_path / 'launches'
    result = shell('scripts/run_eval_vllm_service.sh', tmp_path, f'''
vllm() {{ echo started >> {shlex.quote(str(launches))}; }}
setsid() {{ "$@"; }}
kill() {{ return 0; }}
pkill() {{ echo UNRELATED_PROCESS_KILL; }}
sleep() {{ :; }}
wait() {{ return 0; }}
curl() {{ echo '{{"data":[{{"id":"wrong-model"}}]}}'; }}
''', **vllm_environment(tmp_path, EVAL_VLLM_PARALLEL_START=mode))
    assert result.returncode != 0
    assert 'All eval vLLM endpoints are ready' not in result.stdout
    assert launches.read_text().splitlines() == ['started', 'started']
    assert 'UNRELATED_PROCESS_KILL' not in result.stdout


@pytest.mark.parametrize('host,probe', [('192.0.2.10', 'http://192.0.2.10:9991/v1/models'),
                                     ('::', 'http://[::1]:9991/v1/models')])
def test_vllm_probes_bind_address_and_reports_early_clean_exit_as_failure(tmp_path, host, probe):
    requests = tmp_path / 'requests'
    result = shell('scripts/run_eval_vllm_service.sh', tmp_path, f'''
vllm() {{ :; }}
setsid() {{ "$@"; }}
kill() {{ return 0; }}
wait() {{ return 0; }}
curl() {{ printf '%s\\n' "$*" >> {shlex.quote(str(requests))}; echo '{{"data":[{{"id":"expected"}}]}}'; }}
''', **vllm_environment(tmp_path, EVAL_VLLM_HOST=host))
    assert probe in requests.read_text()
    assert 'All eval vLLM endpoints are ready' in result.stdout
    assert result.returncode != 0
