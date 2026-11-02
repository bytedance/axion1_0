"""Locate and verify the separately installed Swift training backend."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'swift_files.json'


def resolve_swift_package(verify=True):
    spec = importlib.util.find_spec('swift')
    if spec is None or spec.origin is None:
        checkout = HERE.parents[2] / 'third_party/swift/swift'
        if not verify and (checkout / '__init__.py').is_file():
            return checkout
        raise RuntimeError('Install the recorded Swift backend with pip install --no-deps -e third_party/swift.')
    package = Path(spec.origin).resolve().parent
    if verify:
        expected = json.loads(MANIFEST.read_text())
        for name, digest in expected.items():
            path = package / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Installed Swift differs from the recorded source at {name}; '
                                   'install this repository\'s third_party/swift backend.')
    return package
