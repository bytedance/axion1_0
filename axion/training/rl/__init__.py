# Copyright 2025 DeepResearch authors
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
# Axion: defer optional training imports so setup and dry runs stay lightweight.
"""Axion's paper-based reinforcement learning recipe."""
from importlib import import_module

_EXPORTS = {
    'DeepResearchAgentLoop': '.agent_loop',
    'call_condenser_async': '.memory', 'extract_events_from_messages': '.memory',
    'format_prev_state_section': '.memory', 'parse_memory_result': '.memory',
    'compute_score': '.reward', 'compute_score_sync': '.reward',
    'compute_score_wrapper': '.reward', 'compute_score_wrapper_sync': '.reward',
    'DeepResearchSearchTool': '.tools', 'DeepResearchScholarTool': '.tools',
    'DeepResearchPythonTool': '.tools', 'DeepResearchVisitTool': '.tools',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
