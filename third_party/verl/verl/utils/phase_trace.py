# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wall-clock phase spans for step-time / GPU-utilization profiling.

Off unless VERL_PHASE_TRACE=1 (then a no-op context manager). When on, every span appends one
JSON line {name, t0, t1, pid, ...attrs} to $VERL_PHASE_TRACE_DIR/<host>_<pid>.jsonl, so the
trainer, each vLLM server and each actor rank write their own file and nothing contends.
Timestamps are time.time() so they line up with an external `nvidia-smi` sampler.
"""

import json
import os
import socket
import threading
import time
from contextlib import contextmanager

_ENABLED = os.environ.get("VERL_PHASE_TRACE", "0") == "1"
_DIR = os.environ.get("VERL_PHASE_TRACE_DIR", "/tmp/verl_phase_trace")
_LOCK = threading.Lock()


def enabled() -> bool:
    return _ENABLED


def _write(record: dict) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"{socket.gethostname()}_{os.getpid()}.jsonl")
        with _LOCK, open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:  # profiling must never break training
        pass


@contextmanager
def span(name: str, **attrs):
    """Time a block. Works around `await` inside async functions too."""
    if not _ENABLED:
        yield
        return
    t0 = time.time()
    try:
        yield
    finally:
        _write({"name": name, "t0": t0, "t1": time.time(), "pid": os.getpid(), **attrs})


def mark(name: str, **attrs) -> None:
    """A zero-length event (e.g. when an RPC actually starts running on its actor)."""
    if _ENABLED:
        t = time.time()
        _write({"name": name, "t0": t, "t1": t, "pid": os.getpid(), **attrs})


def record(name: str, t0: float, t1: float, **attrs) -> None:
    """Emit a span whose start/end were measured by the caller."""
    if _ENABLED:
        _write({"name": name, "t0": t0, "t1": t1, "pid": os.getpid(), **attrs})
