# Copyright 2026 The Axion Authors
# SPDX-License-Identifier: Apache-2.0
"""Fail-fast checks for the paper RL implementation, without runtime installation.

A passing capability check is not distributed GPU or numerical validation.
The historical final RL environment is not recoverable from these checks.
"""

import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from packaging.version import InvalidVersion, Version

_ARCHITECTURE = 'Qwen3_5MoeForConditionalGeneration'


def _runtime_versions():
    versions = {}
    for distribution in ('torch', 'vllm', 'transformers', 'transformer-engine', 'megatron-core',
                         'mbridge', 'megatron-bridge', 'ray', 'tensordict', 'triton'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _capabilities(vanilla_mbridge):
    capabilities, failures = {}, []

    def check(name, operation):
        try:
            if not operation():
                raise ValueError('required symbol or registration is absent')
        except Exception as error:
            capabilities[name] = False
            # Package import errors can contain local paths/environment values.
            failures.append(f'{name} unavailable ({type(error).__name__})')
        else:
            capabilities[name] = True

    def imported(module, symbol):
        return getattr(importlib.import_module(module), symbol)

    check('vllm_capture_flag', lambda: 'enable_return_routed_experts' in inspect.signature(
        imported('vllm.engine.arg_utils', 'AsyncEngineArgs')).parameters)
    check('vllm_route_output', lambda: 'routed_experts' in getattr(
        imported('vllm.outputs', 'CompletionOutput'), '__annotations__', {}))
    check('vllm_qwen35_moe_vl', lambda: _ARCHITECTURE in imported(
        'vllm.model_executor.models.registry', 'ModelRegistry').get_supported_archs())
    check('vllm_route_capture', lambda: callable(imported(
        'vllm.model_executor.layers.fused_moe.routed_experts_capturer', 'RoutedExpertsCapturer').create))
    check('megatron_router', lambda: callable(imported('megatron.core.transformer.moe.router', 'TopKRouter').routing))
    check('transformer_engine', lambda: callable(imported('transformer_engine.pytorch', 'Linear')))
    check('verl_router_patch', lambda: callable(imported('verl.utils.megatron.router_replay_patch', 'apply_router_replay_patch')))
    check('verl_bshd_replay', lambda: {'data_format', 'forced_max_seqlen'} <= set(inspect.signature(
        imported('verl.utils.megatron.router_replay_utils', 'set_router_replay_data')).parameters))
    check('verl_partial_client', lambda: callable(imported('verl.workers.rollout.llm_server', 'FullyAsyncLLMServerClient').generate))
    if vanilla_mbridge:
        check('mbridge_qwen35_moe_vl', lambda: callable(imported(
            'mbridge.models.qwen3_5.qwen3_5_vl_bridge', 'Qwen3_5MoeVlBridge')))
        check('mbridge_linear_attention', lambda: callable(imported(
            'megatron.core.models.gpt.experimental_attention_variant_module_specs',
            'get_transformer_block_with_experimental_attention_variant_spec')))
    else:
        # The formal recipe uses legacy mbridge. New-bridge users must at least
        # expose the same Qwen3.5 VL registration in the installed bridge.
        check('megatron_bridge_qwen35', lambda: bool(importlib.import_module('megatron.bridge.models.qwen.qwen3_5_provider')))
        check('megatron_bridge_auto', lambda: callable(imported('verl.models.mcore.bridge', 'AutoBridge').from_hf_pretrained))
    return capabilities, failures


def verify_runtime(model_path=None, *, vanilla_mbridge=True):
    """Return a safe capability report or refuse an incompatible runtime.

    Reads only checkpoint config.json, package metadata and required Python
    APIs. Does not load model weights, install packages or allocate GPU models.
    """
    versions = _runtime_versions()
    failures = []
    try:
        if Version(versions.get('vllm') or '0') < Version('0.22.0'):
            failures.append('vLLM >= 0.22.0 is required for Qwen3.5 hybrid-attention R3')
    except InvalidVersion:
        failures.append('Cannot verify the installed vLLM version; require >= 0.22.0')
    model_info = {'architecture': _ARCHITECTURE, 'model_config_checked': False}
    if model_path is not None:
        try:
            config = json.loads((Path(model_path) / 'config.json').read_text())
            text_config = config.get('text_config', config)
            if config.get('architectures') != [_ARCHITECTURE] or config.get('model_type') != 'qwen3_5_moe':
                failures.append(f'The paper recipe requires architecture {_ARCHITECTURE}')
            experts = text_config.get('num_experts')
            if not isinstance(experts, int) or not 1 <= experts <= 256:
                failures.append('R3 uint8 storage requires num_experts in [1, 256]')
            topk, layers = text_config.get('num_experts_per_tok'), text_config.get('num_hidden_layers')
            if not isinstance(topk, int) or not isinstance(layers, int) or topk <= 0 or layers <= 0:
                failures.append('Checkpoint config must declare positive MoE top-k and layer count')
            model_info.update(model_config_checked=True, num_experts=experts, topk=topk, layers=layers)
        except (OSError, ValueError, TypeError):
            failures.append('Cannot read a valid checkpoint config.json')
    capabilities, capability_failures = _capabilities(vanilla_mbridge)
    failures.extend(capability_failures)
    if failures:
        raise RuntimeError('RL runtime preflight failed: ' + '; '.join(failures))
    root = Path(__file__).resolve().parents[3]
    fingerprints = {}
    for relative in ('third_party/verl/verl/workers/rollout/llm_server.py',
                     'third_party/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py',
                     'third_party/verl/verl/utils/rollout_routing.py',
                     'third_party/verl/verl/utils/megatron/router_replay_utils.py'):
        source = root / relative
        if source.is_file():
            fingerprints[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return dict(versions=versions, capabilities=capabilities, **model_info,
                bridge='mbridge' if vanilla_mbridge else 'megatron-bridge',
                source_sha256=fingerprints, lm_head_ce_fusion=False, gpu_validated=False)
