# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import fields
from typing import List

import torch.nn as nn
from mcore_bridge import ModelConfig
from mcore_bridge import get_mcore_model as _get_mcore_model
from mcore_bridge import hf_to_mcore_config
from megatron.core import mpu
from megatron.core.enums import ModelType
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger

logger = get_logger()


def _check_attention_backend(args, config):
    """Validate attention backend compatibility with configuration."""
    attention_backend = config.attention_backend.name
    if attention_backend == 'flash' and config.softmax_type == 'learnable':
        raise ValueError(f'Attention backend "{attention_backend}" does not support learnable softmax_type.')


def _check_padding_free(args, config):
    """Validate and adjust padding_free setting based on configuration constraints."""
    if not args.padding_free:
        return

    attention_backend = config.attention_backend.name
    message = None

    if config.experimental_attention_variant == 'dsa':
        message = 'DSA is not supported in padding-free mode'
    elif attention_backend == 'unfused':
        message = f'Attention backend "{attention_backend}" is not supported in padding-free mode'

    if message:
        logger.warning(f'{message}. Setting args.padding_free to False.')
        args.padding_free = False


def get_mcore_model_config(args, hf_config):
    kwargs = hf_to_mcore_config(hf_config)
    kwargs['mcore_model_type'] = args.megatron_model_meta.model_type
    kwargs['hf_config'] = hf_config
    for f in fields(ModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['params_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp4_param'] = args.fp4_param_gather
    kwargs['fp8_param'] = args.fp8_param_gather
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    num_moe_experts = kwargs.get('num_moe_experts', None)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1

    if args.router_replay_mode != 'disabled':
        kwargs['moe_enable_routing_replay'] = True
    if args.megatron_extra_kwargs:
        kwargs.update(args.megatron_extra_kwargs)
    if getattr(args, 'moe_flex_dispatcher_backend', None) is not None:
        kwargs['moe_flex_dispatcher_backend'] = args.moe_flex_dispatcher_backend
    config = ModelConfig(**kwargs)
    if getattr(args, 'regression_dim', 0) > 0:
        config.aip_train_args = args
    if is_torch_npu_available() and getattr(args, 'attention_backend', 'flash') != 'local':
        setattr(config, 'use_flash_attn', True)
    _check_attention_backend(args, config)
    _check_padding_free(args, config)
    return config


def _build_models_with_loader(config: ModelConfig, loader_cls) -> List[nn.Module]:
    loader = loader_cls(config)
    model_type = ModelType.encoder_or_decoder
    if mpu.get_pipeline_model_parallel_world_size() > 1 and config.virtual_pipeline_model_parallel_size is not None:
        models = []
        for i in range(config.virtual_pipeline_model_parallel_size):
            pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
            post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
            model = loader.build_model(pre_process, post_process, vp_stage=i)
            models.append(model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = loader.build_model(pre_process=pre_process, post_process=post_process)
        models = [model]
    for model in models:
        model.model_type = model_type
        model.prepare_inputs_for_generation = None
    return models


# mcore model types that support the AIP regression head / BT-SFT verbalized
# scoring. Each entry's registered loader is wrapped so the AIP model class is built
# on top of the model's own loader (preserving its layer-spec / transformer-block /
# GDN wiring). qwen3_vl / qwen3_5 share the MultimodalGPTModel base that
# Qwen3VLAIPModel subclasses; gemma4 keeps its own Gemma4GPTModel base (Gemma4AIPModel
# re-bases the qwen AIP/BT-SFT logic onto it via MRO -- see gemma4_aip.py).
AIP_SUPPORTED_MODEL_TYPES = ('qwen3_vl', 'qwen3_5', 'gemma4')


def _make_aip_loader(model_type, base_loader_cls, args=None):
    """Return the loader that builds the AIP model for ``model_type``. gemma4 uses its
    own Gemma4AIPModel (Gemma4GPTModel base); the qwen families share Qwen3VLAIPModel
    (or Qwen3VLAIPReasoningModel when --bt_sft_reasoning_ce is set)."""
    reasoning_ce = args is not None and getattr(args, 'bt_sft_reasoning_ce', False)
    if model_type == 'gemma4':
        if reasoning_ce:
            raise NotImplementedError('--bt_sft_reasoning_ce is only wired for the qwen mcore AIP '
                                      'models; gemma4 BT-SFT runs on the transformers backend.')
        from .gemma4_aip import make_gemma4_aip_loader
        return make_gemma4_aip_loader(base_loader_cls)
    if reasoning_ce:
        from .qwen3_vl_aip_reasoning import make_aip_reasoning_loader
        return make_aip_reasoning_loader(base_loader_cls)
    from .qwen3_vl_aip import make_aip_loader
    return make_aip_loader(base_loader_cls)


def get_mcore_model(args, hf_config):
    config = get_mcore_model_config(args, hf_config)
    model_type = config.model_meta.model_type
    if getattr(args, 'regression_dim', 0) > 0 and model_type in AIP_SUPPORTED_MODEL_TYPES:
        loader_cls = _make_aip_loader(model_type, config.model_meta.loader, args)
        models = _build_models_with_loader(config, loader_cls)
    else:
        models = _get_mcore_model(config)
    return models
