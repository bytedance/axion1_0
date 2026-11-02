import importlib.util
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(os.environ.get('AXION_LAYOUT_REPO', Path(__file__).resolve().parents[1]))


def backend_module():
    spec = importlib.util.spec_from_file_location('axion_backend_test', ROOT / 'axion/training/sft/backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_namespace_is_lightweight_and_module_entrypoint_works(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c',
        'import sys; import axion.training; import axion.training.sft; '
        'assert not any(x in sys.modules for x in ("torch", "swift", "verl"))'],
        cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, '-m', 'axion.training.sft', '--help'],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'stage1' in result.stdout and 'stage2' in result.stdout and 'export' in result.stdout


def test_installed_swift_is_verified_by_content_not_checkout_location(tmp_path, monkeypatch):
    module = backend_module()
    installed = tmp_path / 'site-packages' / 'swift'
    installed.mkdir(parents=True)
    (installed / '__init__.py').write_text('# fixture\n')
    (installed / 'version.py').write_text('VERSION = "fixture"\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({name: hashlib.sha256((installed / name).read_bytes()).hexdigest()
                                   for name in ('__init__.py', 'version.py')}))
    monkeypatch.setattr(module, 'MANIFEST', manifest)
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: SimpleNamespace(origin=str(installed / '__init__.py')))
    assert module.resolve_swift_package() == installed
    (installed / 'version.py').write_text('VERSION = "different-source"\n')
    with pytest.raises(RuntimeError, match='version.py'):
        module.resolve_swift_package()
    (installed / 'version.py').unlink()
    with pytest.raises(RuntimeError, match='version.py'):
        module.resolve_swift_package()


def test_source_dry_run_can_locate_backend_without_installing_it(monkeypatch):
    module = backend_module()
    monkeypatch.setattr(module.importlib.util, 'find_spec', lambda name: None)
    assert module.resolve_swift_package(verify=False) == ROOT / 'third_party/swift/swift'
    with pytest.raises(RuntimeError, match='Install'):
        module.resolve_swift_package()
