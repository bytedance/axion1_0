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
"""Ship a multimodal payload to the rollout server by reference, resolved once per server.

Ray dereferences an ObjectRef passed as a top-level task argument, so each turn of a multi-turn
trajectory re-deserializes the same video: ~180 MB per 64-frame clip (float32), ~26 GB per step
across the rollout servers on the moderation run, visible as torch.load in the server's stacks.
An ObjectRef nested inside another object is NOT resolved by Ray, so wrapping it keeps a
reference on the wire; the server awaits it once and serves later turns from a small LRU.

Enabled by VERL_MM_REF_CACHE=1 (default off). Entries per server: VERL_MM_REF_CACHE_ENTRIES
(default 4; one 64-frame clip each, so ~0.7 GB of host memory per server at float32).
"""

import asyncio
import collections
import os
from dataclasses import dataclass
from typing import Any

from verl.utils.phase_trace import mark as _pt_mark


@dataclass
class MMRef:
    """An ObjectRef kept nested so Ray passes it by reference, plus its cache identity."""

    ref: Any
    key: str
    n_items: int = 0


def enabled() -> bool:
    return os.environ.get("VERL_MM_REF_CACHE", "0") == "1"


def wrap(ref: Any, n_items: int = 0) -> Any:
    """Wrap an ObjectRef for by-reference transport; returns it unchanged when disabled."""
    if not enabled() or ref is None or not hasattr(ref, "hex"):
        return ref
    return MMRef(ref=ref, key=ref.hex(), n_items=n_items)


def _hand_out(data: Any) -> Any:
    """Shallow-copy a list payload before handing it to a caller.

    Cached entries are shared by every turn of a trajectory (and by the n rollouts of a prompt),
    whereas before caching each request owned a freshly deserialized copy. A caller that appended
    to or reordered the list in place would corrupt later turns; copying the list makes that
    impossible. The media objects themselves stay shared and are treated as read-only, which is
    already true of the prompt payload today.
    """
    return list(data) if isinstance(data, list) else data


class MMRefCache:
    """Per-server LRU of resolved payloads, keyed by ObjectRef id (a Ray object is immutable,
    so a hit returns exactly the content that ref was created with -- never a tool-modified
    payload: the agent loop stops wrapping as soon as a tool adds images)."""

    def __init__(self, max_entries: int | None = None):
        if max_entries is None:
            max_entries = int(os.environ.get("VERL_MM_REF_CACHE_ENTRIES", "4"))
        self._max = max(1, max_entries)
        self._data: collections.OrderedDict[str, Any] = collections.OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0

    async def resolve(self, value: Any) -> Any:
        """Return the payload for ``value``; awaits the ref on a miss, never blocking the loop."""
        if not isinstance(value, MMRef):
            return value
        cached = self._data.get(value.key)
        if cached is not None:
            self._data.move_to_end(value.key)
            self.hits += 1
            _pt_mark("vllm/mm_ref", hit=1)
            return _hand_out(cached)
        # One awaiter fetches; concurrent turns of the same prompt wait on the same lock.
        lock = self._locks.setdefault(value.key, asyncio.Lock())
        async with lock:
            cached = self._data.get(value.key)
            if cached is not None:
                self._data.move_to_end(value.key)
                self.hits += 1
                return _hand_out(cached)
            data = await value.ref  # ObjectRefs are awaitable; ray.get() would stall the loop
            self.misses += 1
            _pt_mark("vllm/mm_ref", hit=0)
            self._data[value.key] = data
            self._data.move_to_end(value.key)
            while len(self._data) > self._max:
                evicted, _ = self._data.popitem(last=False)
                self._locks.pop(evicted, None)
            return _hand_out(data)
