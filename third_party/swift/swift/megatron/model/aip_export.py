# Copyright (c) ModelScope Contributors. All rights reserved.
"""Export/load AIP regression_head sidecar for HF + head dual-artifact deployment."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from mcore_bridge import unwrap_model

from swift.utils import get_logger, is_master

logger = get_logger()

REGRESSION_HEAD_FILENAME = 'regression_head.pt'
AIP_CONFIG_FILENAME = 'aip_config.json'
AIP_EXPORT_FORMAT_VERSION = 1


def _tensor_to_cpu(param: torch.Tensor) -> torch.Tensor:
    if hasattr(param, 'full_tensor'):
        param = param.full_tensor()
    return param.detach().contiguous().cpu()

AIP_CONFIG_KEYS = (
    'regression_dim',
    'regression_hidden_dim',
    'regression_loss_type',
    'decision_threshold',
    'decision_thresholds',
    'bt_queue_size',
    'reg_loss_weight',
    'lm_loss_weight',
    'bt_margin',
    'l2_lambda',
    'focal_beta',
    'focal_mode',
    'continuous_pos_weight',
    'threshold_soft_bce_scale',
    'threshold_soft_bce_tau',
)


def find_aip_model(mg_models: Union[nn.Module, List[nn.Module]]) -> Optional[nn.Module]:
    """Return the first Megatron module that carries an AIP regression head."""
    if not isinstance(mg_models, (list, tuple)):
        mg_models = [mg_models]
    for mg_model in mg_models:
        model = unwrap_model(mg_model)
        candidates = model if isinstance(model, (list, tuple)) else [model]
        for candidate in candidates:
            if hasattr(candidate, 'regression_head'):
                return candidate
    return None


def _collect_aip_config(args, model: nn.Module) -> Dict[str, Any]:
    hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', None)
    config: Dict[str, Any] = {
        'format_version': AIP_EXPORT_FORMAT_VERSION,
        'artifact_type': 'qwen3_vl_aip_regression_head',
        'hidden_size': hidden_size,
    }
    for key in AIP_CONFIG_KEYS:
        if hasattr(args, key):
            config[key] = getattr(args, key)
    return config


def build_regression_head_from_config(config: Dict[str, Any]) -> nn.Sequential:
    """Build the same regression head architecture used in Qwen3VLAIPModel."""
    hidden_size = config['hidden_size']
    hidden_dim = config.get('regression_hidden_dim', 512)
    regression_dim = config.get('regression_dim', 1)
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, regression_dim),
    )


def save_aip_regression_head(
    mg_models: Union[nn.Module, List[nn.Module]],
    output_dir: str,
    args,
    *,
    mcore_model: Optional[str] = None,
) -> Optional[str]:
    """Save regression_head.pt (+ aip_config.json) next to HF weights."""
    if not is_master():
        return None

    if getattr(args, 'bt_sft', False):
        # BT-SFT reads the verbalized score off the LM head at the <score_k>
        # position; there is no separate regression head to export. The score
        # tokens are already part of the saved (resized) vocab/embeddings.
        logger.info('[save_aip_regression_head] bt_sft=True: no regression head to save (verbalized scoring).')
        return None

    regression_dim = getattr(args, 'regression_dim', 0) or 0
    model = find_aip_model(mg_models)
    if model is None:
        if regression_dim > 0:
            raise ValueError(
                f'regression_dim={regression_dim} but no regression_head found on the loaded Megatron model.')
        return None

    os.makedirs(output_dir, exist_ok=True)
    config = _collect_aip_config(args, model)
    if mcore_model is not None:
        config['mcore_model'] = mcore_model

    state_dict = {k: _tensor_to_cpu(v) for k, v in model.regression_head.state_dict().items()}
    bundle = {
        'format_version': AIP_EXPORT_FORMAT_VERSION,
        'artifact_type': 'qwen3_vl_aip_regression_head',
        'config': config,
        'state_dict': state_dict,
    }
    head_path = os.path.join(output_dir, REGRESSION_HEAD_FILENAME)
    torch.save(bundle, head_path)

    config_path = os.path.join(output_dir, AIP_CONFIG_FILENAME)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.info(f'Saved AIP regression head to `{head_path}` and `{config_path}`.')
    return head_path


def load_aip_regression_head(
    artifact_dir: str,
    *,
    map_location: Union[str, torch.device] = 'cpu',
    strict: bool = True,
) -> tuple[nn.Sequential, Dict[str, Any]]:
    """Load regression head module and config from an export directory."""
    head_path = os.path.join(artifact_dir, REGRESSION_HEAD_FILENAME)
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f'AIP regression head not found: {head_path}')

    bundle = torch.load(head_path, map_location=map_location, weights_only=False)
    if isinstance(bundle, dict) and 'state_dict' in bundle:
        config = bundle.get('config') or {}
        state_dict = bundle['state_dict']
    else:
        raise ValueError(f'Invalid AIP regression head artifact: {head_path}')

    config_path = os.path.join(artifact_dir, AIP_CONFIG_FILENAME)
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        config = {**file_config, **config}

    if config.get('hidden_size') is None:
        raise ValueError('AIP config missing hidden_size; cannot rebuild regression_head.')

    head = build_regression_head_from_config(config)
    head.load_state_dict(state_dict, strict=strict)
    return head, config
