#!/usr/bin/env python3
"""Install or verify the exact GDN CP backport in a user-managed SFT environment."""
import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement

HERE = Path(__file__).resolve().parent
PATCH = HERE / 'patches/mcore_v0.17.1_gdn_cp_pr2642.diff'
PATCH_SHA256 = '3aa5a6266dd543fa50dd86e370bee25b38737801873736bacaef9ed5789d6cf0'
FILES = {
    'megatron/core/ssm/gated_delta_net.py': (
        '1ebe56125438984e3a0e0b4d5a89ea7ab45366e4a0fd757021b54523f3647875',
        'b456afde1aefc12cea991d3db7e5b9cf6d444dbaa56fc9496cb5812cce6eaee0'),
    'megatron/core/transformer/transformer_config.py': (
        'f05bfdabd18188bfb0c33b4fda9f205798addc8ea001ee8e7dde6c5528ce0802',
        '235c54ce7cd705789fffb587c13e9b42e4a9e8dd65f8e5d2c85fb21548cd3da6'),
}
VERSIONS = {'megatron-core': '0.17.1', 'mcore-bridge': '1.4.3', 'transformers': '5.10.4'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_mcore_patch(package_root):
    """Return whether both files are patched; reject any unrecognized source bytes."""
    states = []
    for name, (original, patched) in FILES.items():
        path = Path(package_root) / name
        if not path.is_file() or sha256(path) not in (original, patched):
            raise RuntimeError(f'unrecognized Megatron-core source: {name}; use the recorded 0.17.1 release')
        states.append(sha256(path) == patched)
    return all(states)


def apply_mcore_patch(package_root):
    """Patch only recognized files; already patched files are left unchanged."""
    package_root = Path(package_root)
    if check_mcore_patch(package_root):
        return
    if sha256(PATCH) != PATCH_SHA256:
        raise RuntimeError('The bundled GDN CP patch checksum does not match its provenance.')
    if not shutil.which('patch'):
        raise RuntimeError('Install the patch command before preparing the SFT environment.')
    sections = ['diff --git ' + part for part in PATCH.read_text().split('diff --git ')[1:]]
    with tempfile.TemporaryDirectory(prefix='axion-gdn-patch-') as directory:
        staged = Path(directory)
        changed = []
        for name, (original, patched) in FILES.items():
            source = package_root / name
            if sha256(source) == patched:
                continue
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            section = next(part for part in sections if part.startswith(f'diff --git a/{name} b/{name}\n'))
            result = subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-d', str(staged)],
                                    input=section, text=True, capture_output=True)
            if result.returncode or sha256(target) != patched:
                raise RuntimeError(f'GDN CP patch verification failed for {name}: {result.stdout} {result.stderr}')
            ast.parse(target.read_text(), filename=name)
            changed.append(name)
        # Validate every file before modifying either installed source file.
        for name in changed:
            target = package_root / name
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.axion-', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write((staged / name).read_bytes())
                temporary.chmod(target.stat().st_mode & 0o777)
                os.replace(temporary, target)
            except PermissionError as exc:
                raise RuntimeError('The SFT environment must be writable by its owner; no sudo is used.') from exc
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    if not check_mcore_patch(package_root):
        raise RuntimeError('GDN CP preparation did not patch both required files.')


def verify_model_dependencies():
    """Check the vendored Qwen3.5 loader requirements, not an inferred image lock."""
    installed = {}
    for value in ('qwen-vl-utils>=0.0.14', 'decord'):
        requirement = Requirement(value)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'Missing model-loader dependency {value}; install requirements-runtime.txt.') from exc
        if requirement.specifier and not requirement.specifier.contains(version):
            raise RuntimeError(f'Model loader requires {value}, found {version}.')
        installed[requirement.name] = version
    for name in ('qwen_vl_utils.vision_process', 'decord'):
        try:
            importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f'Model-loader import failed for {name}: {exc}') from exc
    return installed


def verify_runtime(apply=False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the recorded Python 3.12 SFT environment.')
    for package, expected in VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f'Missing {package}=={expected}; see train/sft/README.md.') from exc
        if installed != expected:
            raise RuntimeError(f'Expected {package}=={expected}, found {installed}.')
    verify_model_dependencies()
    distribution = importlib.metadata.distribution('megatron-core')
    package_root = Path(distribution.locate_file(''))
    if apply:
        apply_mcore_patch(package_root)
    if not check_mcore_patch(package_root):
        raise RuntimeError('GDN CP backport is missing; run python train/sft/runtime.py --apply first.')
    checkpointing = package_root / 'megatron/training/checkpointing.py'
    if not checkpointing.is_file() or 'def preprocess_fsdp_dtensor_state_dict(' not in checkpointing.read_text():
        raise RuntimeError('The installed Megatron release lacks the required FSDP checkpoint function.')
    required_imports = {
        'megatron.core.ssm.gated_delta_net': ['tensor_a2a_cp2hp', 'tensor_a2a_hp2cp', 'get_parameter_local_cp'],
        'megatron.training.checkpointing': ['preprocess_fsdp_dtensor_state_dict'],
        'mcore_bridge.model.modules.gated_delta_net': ['GatedDeltaNet'],
        'fla.modules.convolution': ['causal_conv1d'],
        'fla.ops.gated_delta_rule': ['chunk_gated_delta_rule'],
        'binpacking': [],
    }
    for name, symbols in required_imports.items():
        try:
            module = importlib.import_module(name)
            if any(not callable(getattr(module, symbol, None)) for symbol in symbols):
                raise ImportError('a required callable is unavailable')
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f'SFT runtime import failed for {name}: {exc}') from exc
    return {'python': '3.12', **VERSIONS, 'gdn_cp_patch_sha256': PATCH_SHA256}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='patch the current user-managed SFT environment')
    args = parser.parse_args()
    try:
        print(json.dumps(verify_runtime(apply=args.apply), indent=2))
    except (OSError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
