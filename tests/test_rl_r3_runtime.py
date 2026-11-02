"""Preflight rejects absent/incompatible R3 APIs without installing packages."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_runtime():
    path = ROOT / 'axion/training/rl/runtime.py'
    assert path.is_file(), 'The paper recipe needs a fail-fast runtime capability check'
    spec = importlib.util.spec_from_file_location('rl_runtime_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(tmp_path, *, architecture='Qwen3_5MoeForConditionalGeneration', experts=256):
    (tmp_path / 'config.json').write_text(json.dumps({
        'architectures': [architecture], 'model_type': 'qwen3_5_moe',
        'text_config': {'num_experts': experts, 'num_experts_per_tok': 8, 'num_hidden_layers': 48}}))
    return tmp_path


def test_runtime_rejects_old_vllm_despite_available_symbols(tmp_path, monkeypatch):
    module = load_runtime()
    monkeypatch.setattr(module, '_runtime_versions', lambda: {'vllm': '0.19.0'})
    monkeypatch.setattr(module, '_capabilities', lambda vanilla_mbridge: ({'required_api': True}, []))
    with pytest.raises(RuntimeError, match='vLLM.*0.22'):
        module.verify_runtime(checkpoint(tmp_path))


def test_runtime_rejects_missing_capability_without_silently_disabling_r3(tmp_path, monkeypatch):
    module = load_runtime()
    monkeypatch.setattr(module, '_runtime_versions', lambda: {'vllm': '0.22.0'})
    monkeypatch.setattr(module, '_capabilities', lambda vanilla_mbridge: ({'routing': False}, ['routing API missing']))
    with pytest.raises(RuntimeError, match='routing API missing'):
        module.verify_runtime(checkpoint(tmp_path))


@pytest.mark.parametrize('architecture,experts', [('Qwen3_5MoeForCausalLM', 256),
                                                ('Qwen3_5MoeForConditionalGeneration', 512)])
def test_runtime_rejects_wrong_architecture_or_noncompact_routes(tmp_path, monkeypatch, architecture, experts):
    module = load_runtime()
    monkeypatch.setattr(module, '_runtime_versions', lambda: {'vllm': '0.22.0'})
    monkeypatch.setattr(module, '_capabilities', lambda vanilla_mbridge: ({'required_api': True}, []))
    with pytest.raises(RuntimeError, match='architecture|uint8'):
        module.verify_runtime(checkpoint(tmp_path, architecture=architecture, experts=experts))


def test_runtime_reports_capability_check_separately_from_gpu_validation(tmp_path, monkeypatch):
    module = load_runtime()
    monkeypatch.setattr(module, '_runtime_versions', lambda: {'vllm': '0.22.0', 'torch': '2.10.0'})
    monkeypatch.setattr(module, '_capabilities', lambda vanilla_mbridge: ({'required_api': True}, []))
    report = module.verify_runtime(checkpoint(tmp_path), vanilla_mbridge=True)
    assert report['versions']['vllm'] == '0.22.0'
    assert report['gpu_validated'] is False
    assert report['lm_head_ce_fusion'] is False
    assert report['architecture'] == 'Qwen3_5MoeForConditionalGeneration'
    assert str(tmp_path) not in json.dumps(report)


def test_capability_collector_checks_actual_symbols_at_import_boundary(monkeypatch):
    from types import SimpleNamespace
    module = load_runtime()
    class Arguments:
        def __init__(self, enable_return_routed_experts=False): pass
    class Completion:
        routed_experts: object
    def replay(a, data_format='thd', forced_max_seqlen=None): pass
    modules = {
        'vllm.engine.arg_utils': SimpleNamespace(AsyncEngineArgs=Arguments),
        'vllm.outputs': SimpleNamespace(CompletionOutput=Completion),
        'vllm.model_executor.models.registry': SimpleNamespace(ModelRegistry=SimpleNamespace(
            get_supported_archs=lambda: ['Qwen3_5MoeForConditionalGeneration'])),
        'vllm.model_executor.layers.fused_moe.routed_experts_capturer': SimpleNamespace(
            RoutedExpertsCapturer=SimpleNamespace(create=lambda: None)),
        'megatron.core.transformer.moe.router': SimpleNamespace(TopKRouter=SimpleNamespace(routing=lambda: None)),
        'transformer_engine.pytorch': SimpleNamespace(Linear=lambda: None),
        'verl.utils.megatron.router_replay_patch': SimpleNamespace(apply_router_replay_patch=lambda: None),
        'verl.utils.megatron.router_replay_utils': SimpleNamespace(set_router_replay_data=replay),
        'verl.workers.rollout.llm_server': SimpleNamespace(FullyAsyncLLMServerClient=SimpleNamespace(generate=lambda: None)),
        'mbridge.models.qwen3_5.qwen3_5_vl_bridge': SimpleNamespace(Qwen3_5MoeVlBridge=lambda: None),
        'megatron.core.models.gpt.experimental_attention_variant_module_specs': SimpleNamespace(
            get_transformer_block_with_experimental_attention_variant_spec=lambda: None),
    }
    monkeypatch.setattr(module.importlib, 'import_module', modules.__getitem__)
    capabilities, failures = module._capabilities(True)
    assert not failures and all(capabilities.values())
    modules['vllm.outputs'] = SimpleNamespace(CompletionOutput=type('MissingRoutes', (), {}))
    capabilities, failures = module._capabilities(True)
    assert not capabilities['vllm_route_output'] and len(failures) == 1
