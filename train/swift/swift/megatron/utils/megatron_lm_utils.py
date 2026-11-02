# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for Axion publication (2026-11-02): reproduce the recorded SFT runtime overlay.
# Parts of the functions in this file are code borrowed from NVIDIA/Megatron-LM
import copy
import dataclasses
import megatron.core
import numpy as np
import os
import pickle
import random
import sys
import types
import torch
from argparse import Namespace
from contextlib import contextmanager
from datetime import timedelta
from mcore_bridge import set_random_seed, split_cp_inputs, unwrap_model
from megatron.core import dist_checkpointing, mpu, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (get_default_load_sharded_strategy,
                                                            get_default_save_sharded_strategy)
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from megatron.core.dist_checkpointing.strategies.fully_parallel import (FullyParallelLoadStrategyWrapper,
                                                                        FullyParallelSaveStrategyWrapper)
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy, TorchDistSaveShardedStrategy
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel as megatron_FSDP
from megatron.core.fusions.fused_bias_dropout import bias_dropout_add_fused_train
from megatron.core.fusions.fused_bias_gelu import bias_gelu
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_torch_version, is_te_min_version, is_torch_min_version
from packaging import version
from transformers.utils import is_torch_npu_available
from typing import Any, Dict, Optional

from swift.utils import check_json_format, get_logger, init_process_group, is_master, set_device
from .patcher import patch_merge_fn

logger = get_logger()

mcore_017 = version.parse(megatron.core.__version__) >= version.parse('0.17.0rc0')


@contextmanager
def _patch_megatron_timeout(distributed_timeout_minutes):
    origin_create_group = parallel_state.create_group

    def create_group(ranks=None, timeout=None, *_args, **kwargs):
        if timeout is None:
            timeout = timedelta(minutes=distributed_timeout_minutes)
        return origin_create_group(ranks, timeout, *_args, **kwargs)

    parallel_state.create_group = create_group
    try:
        yield
    finally:
        parallel_state.create_group = origin_create_group


def _initialize_mpu(args):
    """Initialize torch.distributed and core model parallel."""
    if not torch.distributed.is_initialized():
        set_device()
        init_process_group(args.ddp_backend, args.ddp_timeout)
    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    if mpu.model_parallel_is_initialized():
        logger.info('model parallel is already initialized')
    else:
        distributed_timeout_minutes = args.ddp_timeout // 60
        with _patch_megatron_timeout(distributed_timeout_minutes):
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=args.expert_model_parallel_size,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=distributed_timeout_minutes,
            )
        if is_master():
            logger.info(f'TP: {args.tensor_model_parallel_size}, PP: {args.pipeline_model_parallel_size}, '
                        f'VPP: {args.virtual_pipeline_model_parallel_size}, CP: {args.context_parallel_size}, '
                        f'EP: {args.expert_model_parallel_size}, ETP: {args.expert_tensor_parallel_size}')


def initialize_megatron(args):
    # Pytorch distributed.
    _initialize_mpu(args)

    # Random seeds for reproducibility.
    logger.info(f'Setting random seeds to {args.seed}.')
    set_random_seed(args.seed, args.data_parallel_random_init, args.te_rng_tracker)

    # Setup MoE aux loss scale value.
    if args.model_info.is_moe_model:
        from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))


def _get_rng_state():
    """Collect rng state across data parallel ranks."""
    rng_state = {
        'random_rng_state': random.getstate(),
        'np_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
        'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()
    }

    # data_parallel_random_init False
    rng_state_list = [rng_state]

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    rng_state_list = ShardedObject(
        'rng_state',
        rng_state_list, (pp_size, tp_size), (pp_rank, tp_rank),
        replica_id=mpu.get_data_parallel_rank(with_context_parallel=True))
    return rng_state_list


def _get_model_config(model):
    module = model
    while hasattr(module, 'module'):
        module = module.module
    return getattr(module, 'config', None)


def _fsdp_preprocess_args(args, model):
    config = _get_model_config(model)
    return Namespace(
        swiglu=getattr(config, 'swiglu', True) if config is not None else getattr(args, 'swiglu', True),
        num_experts=getattr(config, 'num_moe_experts', None) if config is not None else getattr(args, 'num_experts',
                                                                                                 None),
    )


def _has_float16_module(model):
    module = model
    seen = set()
    while module is not None and id(module) not in seen:
        seen.add(id(module))
        if isinstance(module, Float16Module):
            return True
        module = getattr(module, 'module', None)
    return False


def _fsdp_preprocess_model(model):
    """Adapt Megatron-FSDP checkpoint preprocess for wrapped/unwrapped models."""

    class _PreprocessModelProxy:

        def __init__(self, fsdp_model):
            self._fsdp_model = fsdp_model

        def get_parameter(self, target: str):
            if not target.startswith('module.'):
                return self._fsdp_model.get_parameter(target)
            rest = target[len('module.'):]
            last_error = None
            try:
                return self._fsdp_model.get_parameter(rest)
            except AttributeError as err:
                last_error = err
            for depth in range(1, 5):
                candidate = '.'.join(['module'] * depth) + '.' + rest
                try:
                    return self._fsdp_model.get_parameter(candidate)
                except AttributeError as err:
                    last_error = err
            if last_error is not None:
                raise last_error
            raise AttributeError(target)

        def __getattr__(self, name):
            return getattr(self._fsdp_model, name)

    return _PreprocessModelProxy(model)


def _uses_fsdp_dtensor_ckpt(args, checkpoint_dir: Optional[str] = None) -> bool:
    ckpt_format = getattr(args, 'ckpt_format', 'torch_dist')
    if checkpoint_dir is not None:
        is_torch_dcp = os.path.exists(os.path.join(checkpoint_dir, '.metadata'))
        if is_torch_dcp and (getattr(args, 'use_megatron_fsdp', False) or ckpt_format == 'fsdp_dtensor'):
            return True
        if dist_checkpointing.check_is_distributed_checkpoint(checkpoint_dir):
            return False
    return getattr(args, 'use_megatron_fsdp', False) and ckpt_format == 'fsdp_dtensor'


def _generate_state_dict(args,
                         models,
                         optimizer=None,
                         opt_param_scheduler=None,
                         rng_state=None,
                         iteration=None,
                         model_sd_kwargs=None,
                         optim_sd_kwargs=None):
    model_sd_kwargs = model_sd_kwargs or {}
    optim_sd_kwargs = optim_sd_kwargs or {}
    ckpt_format = getattr(args, 'ckpt_format', 'torch_dist')
    state_dict = {
        'args': Namespace(**check_json_format(vars(args))),
        'checkpoint_version': 3.0,
    }
    if iteration is not None:
        state_dict['iteration'] = iteration
    for i, m in enumerate(models):
        key = 'model'
        if len(models) > 1:
            key = f'model{i}'
        if ckpt_format == 'torch_dist':
            model_sd = models[i].sharded_state_dict(**model_sd_kwargs)
        else:
            model_sd = models[i].state_dict_for_save_checkpoint()
        state_dict[key] = model_sd

    if not args.no_save_optim:
        if optimizer is not None:
            if ckpt_format == 'fsdp_dtensor':
                if 'metadata' not in optim_sd_kwargs:
                    optim_sd_kwargs['metadata'] = {}
                optim_sd_kwargs['metadata'].update(model_sd_kwargs.get('metadata', {}))
            state_dict['optimizer'] = _optimizer_sharded_state_dict(optimizer, state_dict, optim_sd_kwargs)
        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

    if not args.no_save_rng and rng_state is not None:
        state_dict['rng_state'] = rng_state
    return state_dict


def _optimizer_sharded_state_dict(optimizer, state_dict, optim_sd_kwargs):
    if is_torch_npu_available():
        from swift.model.npu_patch.megatron_checkpoint import optimizer_sharded_state_dict
        return optimizer_sharded_state_dict(optimizer, state_dict, **optim_sd_kwargs)
    return optimizer.sharded_state_dict(state_dict, **optim_sd_kwargs)


def _load_optimizer_state_dict(optimizer, state_dict):
    if is_torch_npu_available():
        from swift.model.npu_patch.megatron_checkpoint import load_optimizer_state_dict
        load_optimizer_state_dict(optimizer, state_dict)
        return
    optimizer.load_state_dict(state_dict)


def _filter_adapter_state_dict(state_dict, peft_format: bool, adapter_name: str = 'default'):
    """
    When peft_format is True, keep only the PEFT format state_dict;
    when False, remove the PEFT format state_dict.

    This function ensures it is called when tuner_type != 'full'.
    """
    if 'model' in state_dict:
        n_models = 1
    else:
        n_models = 0
        while f'model{n_models}' in state_dict:
            n_models += 1

    for i in range(n_models):
        if i == 0 and n_models == 1:
            model_key = 'model'
        else:
            model_key = f'model{i}'
        new_state_dict = {}
        state_dict_model = state_dict[model_key]
        for k, v in state_dict_model.items():
            if peft_format:
                if '.lora_A.' in k or '.lora_B.' in k or '.modules_to_save.' in k:
                    new_state_dict[k] = v
            else:
                if '.lora_A.' in k or '.lora_B.' in k or 'original_module.' in k:
                    continue
                k = k.replace('base_layer.', '')
                k = k.replace(f'modules_to_save.{adapter_name}.', '')
                if hasattr(v, 'key'):
                    v.key = v.key.replace('base_layer.', '')
                    v.key = v.key.replace(f'modules_to_save.{adapter_name}.', '')
                new_state_dict[k] = v
        state_dict[model_key] = new_state_dict


_EXTRA_MODULE_FRAGMENTS = ('regression_head',)


def _is_resume_checkpoint(args, checkpoint_dir: str) -> bool:
    output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save', None)
    if not output_dir or not checkpoint_dir:
        return False
    try:
        checkpoint_dir = os.path.realpath(checkpoint_dir)
        output_dir = os.path.realpath(output_dir)
        return checkpoint_dir == output_dir or checkpoint_dir.startswith(output_dir + os.sep)
    except Exception:
        return False


def _filter_extra_modules_for_pretrained_load(args, sharded_state_dict, checkpoint_dir: str):
    """Skip task-specific heads when finetuning from a pretrained mcore checkpoint."""
    if not getattr(args, 'finetune', False):
        return
    if getattr(args, 'tuner_type', 'full') != 'full':
        return
    if _is_resume_checkpoint(args, checkpoint_dir):
        return
    model_keys = [k for k in sharded_state_dict if k.startswith('model')]
    for model_key in model_keys:
        model_state = sharded_state_dict[model_key]
        for key in [k for k in list(model_state) if any(f in k for f in _EXTRA_MODULE_FRAGMENTS)]:
            logger.info(f'Skipping extra-module key from pretrained checkpoint: {key}')
            del model_state[key]


def _install_legacy_swift_llm_aliases():
    """Allow old ms-swift checkpoint metadata to unpickle in the new package layout."""
    if 'swift.llm.model.register' in sys.modules:
        return
    from swift.model import Model, ModelArch, ModelGroup, ModelInfo, ModelKeys, ModelMeta, ModelType, MultiModelKeys

    llm_module = types.ModuleType('swift.llm')
    llm_model_module = types.ModuleType('swift.llm.model')
    llm_model_model_module = types.ModuleType('swift.llm.model.model')
    llm_model_qwen_module = types.ModuleType('swift.llm.model.model.qwen')
    llm_register_module = types.ModuleType('swift.llm.model.register')
    llm_utils_module = types.ModuleType('swift.llm.model.utils')
    llm_model_arch_module = types.ModuleType('swift.llm.model.model_arch')

    for module in (llm_module, llm_model_module, llm_model_model_module, llm_model_qwen_module,
                   llm_register_module, llm_utils_module, llm_model_arch_module):
        module.__dict__['__compat_alias__'] = True
    llm_module.__path__ = []
    llm_model_module.__path__ = []
    llm_model_model_module.__path__ = []

    llm_module.ModelType = ModelType
    llm_module.Model = Model
    llm_module.ModelGroup = ModelGroup
    llm_module.ModelMeta = ModelMeta
    llm_module.ModelInfo = ModelInfo
    llm_module.ModelKeys = ModelKeys
    llm_module.MultiModelKeys = MultiModelKeys
    llm_module.ModelArch = ModelArch

    llm_register_module.Model = Model
    llm_register_module.ModelGroup = ModelGroup
    llm_register_module.ModelMeta = ModelMeta
    llm_utils_module.ModelInfo = ModelInfo
    llm_model_arch_module.ModelKeys = ModelKeys
    llm_model_arch_module.MultiModelKeys = MultiModelKeys
    llm_model_arch_module.ModelArch = ModelArch
    def _legacy_unused_model_loader(*args, **kwargs):
        raise RuntimeError('Legacy swift.llm model loader placeholder should not be called during checkpoint load.')
    llm_model_qwen_module.get_model_tokenizer_qwen3_moe_vl = _legacy_unused_model_loader

    llm_model_model_module.qwen = llm_model_qwen_module
    llm_model_module.register = llm_register_module
    llm_model_module.utils = llm_utils_module
    llm_model_module.model_arch = llm_model_arch_module
    llm_model_module.model = llm_model_model_module
    llm_module.model = llm_model_module

    sys.modules['swift.llm'] = llm_module
    sys.modules['swift.llm.model'] = llm_model_module
    sys.modules['swift.llm.model.model'] = llm_model_model_module
    sys.modules['swift.llm.model.model.qwen'] = llm_model_qwen_module
    sys.modules['swift.llm.model.register'] = llm_register_module
    sys.modules['swift.llm.model.utils'] = llm_utils_module
    sys.modules['swift.llm.model.model_arch'] = llm_model_arch_module


def _install_legacy_swift_megatron_aliases():
    if 'swift.megatron.model.mm_gpt.qwen3_vl' in sys.modules:
        return

    mm_gpt_module = types.ModuleType('swift.megatron.model.mm_gpt')
    mm_gpt_utils_module = types.ModuleType('swift.megatron.model.mm_gpt.utils')
    mm_gpt_qwen3_vl_module = types.ModuleType('swift.megatron.model.mm_gpt.qwen3_vl')
    gpt_module = types.ModuleType('swift.megatron.model.gpt')
    gpt_config_module = types.ModuleType('swift.megatron.model.gpt.config')
    model_provider_module = types.ModuleType('swift.megatron.model.model_provider')

    for module in (mm_gpt_module, mm_gpt_utils_module, mm_gpt_qwen3_vl_module, gpt_module, gpt_config_module,
                   model_provider_module):
        module.__dict__['__compat_alias__'] = True
    mm_gpt_module.__path__ = []
    gpt_module.__path__ = []

    class MMGPTMegatronModelMeta:
        pass

    class Qwen3VLGPTModel:
        pass

    class Qwen3VL_Vit:
        pass

    def _legacy_unused(*args, **kwargs):
        raise RuntimeError('Legacy Megatron metadata placeholder should not be called during checkpoint load.')

    mm_gpt_utils_module.MMGPTMegatronModelMeta = MMGPTMegatronModelMeta
    mm_gpt_qwen3_vl_module.convert_mcore2hf_qwen3_vl = _legacy_unused
    mm_gpt_qwen3_vl_module.convert_hf2mcore_qwen3_vl = _legacy_unused
    mm_gpt_qwen3_vl_module.Qwen3VLGPTModel = Qwen3VLGPTModel
    mm_gpt_qwen3_vl_module.Qwen3VL_Vit = Qwen3VL_Vit
    gpt_config_module.convert_gpt_hf_config = _legacy_unused
    model_provider_module.model_provider = _legacy_unused

    mm_gpt_module.utils = mm_gpt_utils_module
    mm_gpt_module.qwen3_vl = mm_gpt_qwen3_vl_module
    gpt_module.config = gpt_config_module

    sys.modules['swift.megatron.model.mm_gpt'] = mm_gpt_module
    sys.modules['swift.megatron.model.mm_gpt.utils'] = mm_gpt_utils_module
    sys.modules['swift.megatron.model.mm_gpt.qwen3_vl'] = mm_gpt_qwen3_vl_module
    sys.modules['swift.megatron.model.gpt'] = gpt_module
    sys.modules['swift.megatron.model.gpt.config'] = gpt_config_module
    sys.modules['swift.megatron.model.model_provider'] = model_provider_module


def _load_common_state_dict(checkpoint_dir: str):
    from megatron.core.dist_checkpointing.strategies.common import load_common as native_load_common
    try:
        return native_load_common(checkpoint_dir)
    except pickle.UnpicklingError as e:
        message = str(e)
        if 'Weights only load failed' not in message:
            raise
        common_path = os.path.join(checkpoint_dir, 'common.pt')
        logger.warning(
            f'Falling back to torch.load(weights_only=False) for trusted Megatron checkpoint common state: '
            f'{common_path}')
        _install_legacy_swift_llm_aliases()
        _install_legacy_swift_megatron_aliases()
        return torch.load(common_path, map_location='cpu', weights_only=False)


def _patch_dist_checkpointing_load_common():
    load_fn = dist_checkpointing.load
    if load_fn.__globals__.get('_swift_trusted_common_load_patched', False):
        return

    def _trusted_load_common(checkpoint_dir):
        return _load_common_state_dict(str(checkpoint_dir))

    load_fn.__globals__['load_common'] = _trusted_load_common
    load_fn.__globals__['_swift_trusted_common_load_patched'] = True


def _preprocess_common_before_consistancy_check(common_state_dict):
    # Convert args key of type namespace to dictionary
    preprocessed_common_state_dict = copy.deepcopy(common_state_dict)
    preprocessed_common_state_dict['args'] = vars(preprocessed_common_state_dict['args'])
    # Remove rank and local rank from state dict if it exists, since they are expected to be different
    preprocessed_common_state_dict['args'].pop('local_rank', None)
    preprocessed_common_state_dict['args'].pop('rank', None)
    return preprocessed_common_state_dict


def get_sharded_sd_metadata(args):
    sharded_sd_metadata = {'singleton_local_shards': False, 'chained_optim_avoid_prefix': True}
    if getattr(args, 'use_megatron_fsdp', False):
        sharded_sd_metadata['distrib_optim_sharding_type'] = 'fsdp_dtensor'
        return sharded_sd_metadata
    force_pre_mcore_014 = not is_torch_min_version('2.6a0')
    if force_pre_mcore_014 and not args.dist_ckpt_save_pre_mcore_014:
        args.dist_ckpt_save_pre_mcore_014 = True
        logger.warning(f'PyTorch version {get_torch_version()} below 2.6 detected.'
                       f' Forcing dist_ckpt_save_pre_mcore_014 behavior.')

    if args.dist_ckpt_save_pre_mcore_014:
        sharded_sd_metadata['distrib_optim_sharding_type'] = 'fully_sharded_model_space'
    else:
        if args.dist_ckpt_optim_fully_reshardable:
            sharded_sd_metadata['distrib_optim_sharding_type'] = 'fully_reshardable'
            sharded_sd_metadata[
                'distrib_optim_fully_reshardable_mem_efficient'] = args.distrib_optim_fully_reshardable_mem_efficient
        else:
            sharded_sd_metadata['distrib_optim_sharding_type'] = 'dp_reshardable'
    return sharded_sd_metadata


def save_mcore_checkpoint(
    args,
    models,
    optimizer=None,
    opt_param_scheduler=None,
    iteration=1,
    output_dir: Optional[str] = None,
    peft_format: bool = False,
):
    if output_dir is None:
        output_dir = args.output_dir
    use_fsdp_dtensor = _uses_fsdp_dtensor_ckpt(args)
    wrapped_models = models
    if use_fsdp_dtensor and models:
        save_models = models
    else:
        save_models = unwrap_model(models)
    rng_state = _get_rng_state() if save_models else None
    checkpoint_dir = os.path.join(output_dir, f'iter_{iteration:07d}')
    sharded_sd_metadata = get_sharded_sd_metadata(args)
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_dict = _generate_state_dict(
        args,
        save_models,
        optimizer,
        opt_param_scheduler,
        rng_state,
        iteration=iteration,
        model_sd_kwargs={'metadata': sharded_sd_metadata},
        optim_sd_kwargs={'metadata': sharded_sd_metadata},
    )
    _filter_adapter_state_dict(state_dict, peft_format)
    kwargs = {'content_metadata': sharded_sd_metadata}
    async_save = args.async_save
    if not save_models:  # save GPU memory
        assert 'optimizer' not in state_dict
        async_save = False
        common_path = os.path.join(checkpoint_dir, 'common.pt')
        if is_master():
            state_dict.update(kwargs)
            torch.save(state_dict, common_path)
        async_save_request = None
    elif use_fsdp_dtensor:
        if async_save:
            raise NotImplementedError('async_save is not supported with Megatron-FSDP fsdp_dtensor checkpoints.')
        from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
        preprocess_args = _fsdp_preprocess_args(args, wrapped_models[0])
        preprocess_model = _fsdp_preprocess_model(wrapped_models[0])
        state_dict = preprocess_fsdp_dtensor_state_dict(preprocess_args, state_dict, preprocess_model)
        fs_storage_writer = torch.distributed.checkpoint.FileSystemWriter(  # afm_122b_exp patch: parallel NAS writes
            checkpoint_dir, thread_count=int(os.environ.get('AFM_DCP_SAVE_THREADS', '8')),
            sync_files=os.environ.get('AFM_DCP_SYNC_FILES', '1') == '1')
        torch.distributed.checkpoint.save(
            state_dict=state_dict,
            storage_writer=fs_storage_writer,
        )
        async_save_request = None
    else:
        if mcore_017:
            save_strategy = TorchDistSaveShardedStrategy()
        else:
            save_strategy = get_default_save_sharded_strategy()
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True),
        )
        async_save_request = dist_checkpointing.save(
            state_dict,
            checkpoint_dir,
            save_strategy,
            async_sharded_save=async_save,
            validate_access_integrity=True,
            preprocess_common_before_consistancy_check=_preprocess_common_before_consistancy_check,
            **kwargs)

    if not async_save:
        assert async_save_request is None
        # Wait so everyone is done (necessary)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    if is_master():

        tracker_path = os.path.join(output_dir, 'latest_checkpointed_iteration.txt')
        try:
            from megatron.core.msc_utils import open_file
        except ImportError:
            open_file = open
        with open_file(tracker_path, 'w') as f:
            f.write(str(iteration))

        def iter_finalize_fn():
            logger.info(f'Successfully saved Megatron model weights in `{output_dir}`.')

        if async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(iter_finalize_fn)
        else:
            iter_finalize_fn()

    if async_save:
        schedule_async_save(async_save_request)


# Singleton manager of async calls
# The default is `TemporalAsyncCaller`
_async_calls_queue = AsyncCallsQueue()


def init_persistent_async_worker():
    global _async_calls_queue
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    _async_calls_queue = AsyncCallsQueue(persistent=True)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(args, blocking: bool = False, terminate=False):
    """Finalizes active async save calls.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not args.async_save:
        return

    _async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

    if terminate:
        _async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks."""
    return _async_calls_queue.get_num_unfinalized_calls() == 0


def _load_iteration(tracker_path: str):
    if not os.path.exists(tracker_path):
        return 0
    with open(tracker_path, 'r') as f:
        iteration = int(f.read())
    # Get the max iteration retrieved across the ranks.
    if torch.distributed.is_initialized():
        iters_cuda = torch.tensor([iteration], dtype=torch.long, device='cuda')
        torch.distributed.all_reduce(iters_cuda, op=torch.distributed.ReduceOp.MAX)
        iteration = iters_cuda[0].item()
    return iteration


def dcp_direct_load_backbone(model, checkpoint_dir: str, prefix: str = 'model.module.'):
    """Load an fsdp_dtensor (logical-key torch-DCP) checkpoint into a PLAIN mcore model, single
    process (TP1/EP1/PP1). Bypasses the Megatron-FSDP wrap (which allocates fp32 master/grad
    buffers and reshards the param layout — slow, and OOMs a large model on few GPUs). The
    checkpoint stores params by logical mcore name; this:
      - DCP-loads every tensor to a full CPU tensor (FileSystemReader),
      - recombines SwiGLU `_w`(gate)/`_v`(up) splits via cat(dim=0) (LLM experts + vision MLP fc1),
      - strips the `model.module.` prefix,
      - load_state_dict(strict=False) into the model (per-expert `weight{N}` keys match the model).
    Returns (n_loaded, missing_keys, unexpected_keys).
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    reader = FileSystemReader(checkpoint_dir)
    sdm = reader.read_metadata().state_dict_metadata
    tkeys = [k for k in sdm if hasattr(sdm[k], 'size') and not k.startswith('optimizer')]
    target = {k: torch.empty(tuple(sdm[k].size), dtype=sdm[k].properties.dtype) for k in tkeys}
    logger.info(f'[dcp-direct] DCP-loading {len(target)} tensors from {checkpoint_dir} ...')
    dcp.load(target, storage_reader=reader)
    logger.info('[dcp-direct] DCP load complete; recombining swiglu splits + stripping prefix')
    model_sd = {}
    for k, t in target.items():
        if not k.startswith(prefix):
            continue  # e.g. 'args'
        mk = k[len(prefix):]
        if mk.endswith('_w'):  # SwiGLU gate; concat with matching _v (up) -> combined fc1 ([gate; up])
            v = target.get(k[:-2] + '_v')
            assert v is not None, f'missing _v pair for {k}'
            model_sd[mk[:-2]] = torch.cat([t, v], dim=0)
        elif mk.endswith('_v'):
            continue
        else:
            model_sd[mk] = t
    ret = model.load_state_dict(model_sd, strict=False)
    missing = list(getattr(ret, 'missing_keys', ret[0] if isinstance(ret, tuple) else []))
    unexpected = list(getattr(ret, 'unexpected_keys', ret[1] if isinstance(ret, tuple) else []))
    # _extra_state (TE) keys are expected to be missing/unexpected; surface anything else.
    real_missing = [k for k in missing if '_extra_state' not in k]
    real_unexpected = [k for k in unexpected if '_extra_state' not in k]
    logger.info(f'[dcp-direct] loaded {len(model_sd)} tensors. '
                f'missing(non-extra)={len(real_missing)} unexpected(non-extra)={len(real_unexpected)}')
    if real_missing:
        logger.warning(f'[dcp-direct] missing (sample): {real_missing[:10]}')
    if real_unexpected:
        logger.warning(f'[dcp-direct] unexpected (sample): {real_unexpected[:10]}')
    n = len(model_sd)
    del target, model_sd
    return n, real_missing, real_unexpected


def prepare_models_for_mcore_load(args, models, load_dir: str):
    """Wrap models to match the checkpoint structure before load."""
    tracker_path = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    iteration = _load_iteration(tracker_path)
    checkpoint_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    if _uses_fsdp_dtensor_ckpt(args, checkpoint_dir):
        return wrap_model(args, models)
    return models


def materialize_fsdp_dtensor_params(models) -> int:
    """Gather Megatron-FSDP DTensor params/buffers into plain full tensors, in-place.

    After loading an fsdp_dtensor checkpoint, parameters are DTensors sharded along
    the FSDP mesh; mcore_bridge export expects plain local tensors. All ranks must
    call this together as it performs collective communication.
    Returns the number of materialized tensors.
    """
    from torch.distributed.tensor import DTensor
    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        gather_uneven_dtensor_to_full_tensor, update_uneven_dtensor_chunk_metadata)

    if not isinstance(models, (list, tuple)):
        models = [models]

    # Collect targets in deterministic (module-tree) order so all ranks issue the
    # same collective sequence; dedupe tensors shared between modules.
    targets = []
    for model in models:
        for module in unwrap_model(model).modules():
            for name, param in list(module._parameters.items()):
                if param is not None and isinstance(param, DTensor):
                    targets.append((module, name, True, param.data))
            for name, buf in list(module._buffers.items()):
                if buf is not None and isinstance(buf, DTensor):
                    targets.append((module, name, False, buf))
    if not targets:
        return 0

    # Sanity check: every rank must process the same collective sequence.
    if torch.distributed.is_initialized():
        counts = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(counts, len(targets))
        if len(set(counts)) != 1:
            raise RuntimeError(f'Mismatched DTensor param counts across ranks: {counts}. '
                               'Cannot materialize Megatron-FSDP params safely.')

    # Phase 1: refresh chunk metadata unconditionally. The lazy hasattr check inside
    # gather_uneven_dtensor_to_full_tensor may diverge across ranks (metadata is only
    # attached to tensors touched by checkpoint preprocessing), desynchronizing the
    # collective sequence and deadlocking with later bridge collectives.
    refreshed = set()
    for _, _, _, dtensor in targets:
        key = id(dtensor._local_tensor)
        if key in refreshed:
            continue
        update_uneven_dtensor_chunk_metadata(dtensor)
        refreshed.add(key)

    # Phase 2: gather full tensors and swap them into the modules.
    gathered = {}
    for module, name, is_param, dtensor in targets:
        key = id(dtensor._local_tensor)
        if key not in gathered:
            full = gather_uneven_dtensor_to_full_tensor(dtensor)
            gathered[key] = full.to_local().detach().contiguous()
        if is_param:
            module._parameters[name] = torch.nn.Parameter(gathered[key], requires_grad=False)
        else:
            module._buffers[name] = gathered[key]

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f'Materialized {len(targets)} Megatron-FSDP DTensor tensors to full tensors.')
    return len(targets)


def load_mcore_checkpoint(args,
                          ddp_models: list,
                          optimizer=None,
                          opt_param_scheduler=None,
                          load_arg: str = 'mcore_model',
                          adapter_name: str = 'default'):
    if load_arg in {'mcore_adapter', 'mcore_ref_adapter'}:
        peft_format = True
    else:
        # 'mcore_model', 'mcore_ref_model'
        peft_format = False
    load_dir = getattr(args, load_arg)

    no_load_optim = args.no_load_optim
    no_load_rng = args.no_load_rng
    finetune = args.finetune
    if not peft_format and args.tuner_type != 'full':
        # When training with LoRA and loading the base model
        no_load_optim = True
        no_load_rng = True
        finetune = True
    tracker_path = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    iteration = _load_iteration(tracker_path)
    checkpoint_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
    use_fsdp_dtensor = _uses_fsdp_dtensor_ckpt(args, checkpoint_dir)
    if use_fsdp_dtensor:
        load_models = ddp_models
        common_state_dict = {}
        ckpt_tp_pp = (
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
        )
        sharded_sd_metadata = get_sharded_sd_metadata(args)
    else:
        load_models = unwrap_model(ddp_models)
        common_state_dict = _load_common_state_dict(checkpoint_dir)
        ckpt_tp_pp = (
            common_state_dict['args'].tensor_model_parallel_size,
            common_state_dict['args'].pipeline_model_parallel_size,
        )
        sharded_sd_metadata = common_state_dict.get('content_metadata')
    run_tp_pp = (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
    )
    mismatch_msg = f'(TP, PP) mismatch after resume ({run_tp_pp} vs {ckpt_tp_pp} from checkpoint)'
    ckpt_args = common_state_dict.get('args', args)
    # Determine if RNG state will be loaded
    if (ckpt_tp_pp == run_tp_pp and not finetune and not no_load_rng
            and not getattr(ckpt_args, 'no_save_rng', False)):
        gen_sd_rng_state = _get_rng_state()  # we can load the rng state
    else:
        gen_sd_rng_state = None
        if ckpt_tp_pp != run_tp_pp:
            logger.info(f'{mismatch_msg}: RNG state will be ignored')
    if (not finetune and not no_load_optim and not getattr(ckpt_args, 'no_save_optim', False)):
        gen_sd_optim = optimizer
        gen_sd_opt_param_scheduler = opt_param_scheduler

        if (args.use_distributed_optimizer and ckpt_tp_pp != run_tp_pp
                and (sharded_sd_metadata or {}).get('distrib_optim_sharding_type') not in {
                    'fully_reshardable',
                    'fully_sharded_model_space',
                    'fsdp_dtensor',
                }):
            raise RuntimeError(f'{mismatch_msg}: not supported for DistributedOptimizer')
    else:
        gen_sd_optim, gen_sd_opt_param_scheduler = None, None
    optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
    model_sd_kwargs = dict(metadata=sharded_sd_metadata)
    # TODO: check no_save_optim
    sharded_state_dict = _generate_state_dict(
        args,
        load_models,
        gen_sd_optim,
        gen_sd_opt_param_scheduler,
        gen_sd_rng_state,
        iteration=iteration,
        model_sd_kwargs=model_sd_kwargs,
        optim_sd_kwargs=optim_sd_kwargs)
    _filter_adapter_state_dict(sharded_state_dict, peft_format, adapter_name=adapter_name)
    _filter_extra_modules_for_pretrained_load(args, sharded_state_dict, checkpoint_dir)
    if use_fsdp_dtensor:
        from megatron.training.checkpointing import preprocess_fsdp_dtensor_state_dict
        preprocess_args = _fsdp_preprocess_args(args, ddp_models[0])
        preprocess_model = _fsdp_preprocess_model(ddp_models[0])
        sharded_state_dict = preprocess_fsdp_dtensor_state_dict(preprocess_args, sharded_state_dict, preprocess_model)
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(checkpoint_dir)
        torch.distributed.checkpoint.load_state_dict(
            state_dict=sharded_state_dict,
            storage_reader=fs_storage_reader,
        )
        state_dict = sharded_state_dict
    else:
        model_keys = [k for k in sharded_state_dict.keys() if k.startswith('model')]  # compat vpp
        for k in model_keys:
            patch_merge_fn(sharded_state_dict[k])
        if mcore_017:
            load_strategy = TorchDistLoadShardedStrategy()
        else:
            load_strategy = get_default_load_sharded_strategy(checkpoint_dir)

        load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                         mpu.get_data_parallel_group(with_context_parallel=True))
        _patch_dist_checkpointing_load_common()
        state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_dir, load_strategy)

    if finetune:
        iteration = 0
    if 'args' in state_dict and not finetune:
        args.consumed_train_samples = getattr(state_dict['args'], 'consumed_train_samples', 0)

    if len(ddp_models) == 1:
        ddp_models[0].load_state_dict(state_dict['model'], strict=False)
    else:
        for i, m in enumerate(ddp_models):
            if f'model{i}' not in state_dict:
                continue
            m.load_state_dict(state_dict[f'model{i}'])

    if not finetune and not no_load_optim:
        if optimizer is not None:
            _load_optimizer_state_dict(optimizer, state_dict['optimizer'])
        if opt_param_scheduler is not None:
            opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
    elif (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()

    if not finetune and not no_load_rng:
        if 'rng_state' in state_dict:
            rng_state = state_dict['rng_state']
            if use_fsdp_dtensor:
                tp_rank = mpu.get_tensor_model_parallel_rank()
                pp_rank = mpu.get_pipeline_model_parallel_rank()
                rank_key = f'({pp_rank}, {tp_rank})'
                if rank_key in rng_state:
                    rng_state = rng_state[rank_key]
                elif rng_state:
                    rng_state = next(iter(rng_state.values()))
            elif args.data_parallel_random_init:
                rng_state = rng_state[mpu.get_data_parallel_rank()]
            else:
                rng_state = rng_state[0]
            random.setstate(rng_state['random_rng_state'])
            np.random.set_state(rng_state['np_rng_state'])
            torch.set_rng_state(rng_state['torch_rng_state'])
            torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
            tensor_parallel.get_cuda_rng_tracker().set_states(rng_state['rng_tracker_states'])
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    logger.info(f'Successfully loaded Megatron model weights from: {load_dir}')
    return iteration


def wrap_model(args, models, wrap_with_ddp: bool = True):
    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for m in models:
        for param in m.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
        m.cuda(torch.cuda.current_device())
    # Fp16
    config = models[0].config
    if args.fp16 or args.bf16:
        models = [Float16Module(config, model_module) for model_module in models]

    # DDP / Megatron-FSDP
    if not wrap_with_ddp:
        return
    if args.use_megatron_fsdp:
        DP = megatron_FSDP
    else:
        DP = DDP
    kwargs = {}
    for f in dataclasses.fields(DistributedDataParallelConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
    kwargs['check_for_nan_in_grad'] = True
    if args.use_megatron_fsdp:
        # Align with Megatron-LM training.py: wire FSDP main buffer dtypes explicitly.
        kwargs['megatron_fsdp_main_grads_dtype'] = (
            torch.float32 if args.accumulate_allreduce_grads_in_fp32 else args.main_grads_dtype)
        if args.use_precision_aware_optimizer:
            # PAO master weights are owned by TE FusedAdam (decoupled_grad path in MCore >= 0.16).
            kwargs['megatron_fsdp_main_params_dtype'] = None
    ddp_config = DistributedDataParallelConfig(**kwargs)

    # In the Megatron FSDP and DDP use path, we need to initialize the bucket size.
    # If bucket_size is not provided as an input, use sane default.
    # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
    # ring-reduce implementations are large enough to remain bandwidth-bound rather than
    # latency-bound.
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True))
    # Set bucket_size to infinity if overlap_grad_reduce is False.
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    ddp_stream = torch.cuda.Stream()
    ddp_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(ddp_stream):
        models = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step,
            ) for (model_chunk_idx, model_chunk) in enumerate(models)
        ]
    torch.cuda.current_stream().wait_stream(ddp_stream)

    if args.use_megatron_fsdp and args.use_precision_aware_optimizer:
        for model_chunk in models:
            model_chunk.param_and_grad_buffer.use_precision_aware_optimizer = True

    # Broadcast params from data parallel src rank to other data parallel ranks.
    if args.data_parallel_random_init:
        for m in models:
            m.broadcast_params()

    return models


def get_optimizer_param_scheduler(args, optimizer):
    # Iteration-based training.
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    if getattr(args, 'use_megatron_fsdp', False) and args.overlap_param_gather:
        # MegatronFSDP registers its own forward pre-hooks for param prefetch.
        return False
    return args.use_distributed_optimizer and args.overlap_param_gather


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        if isinstance(model_chunk, DDP):
            model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        if isinstance(model_chunk, DDP):
            model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def initialize_tp_communicators(args, config):
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""
    from transformer_engine.pytorch import module as te_module
    input_shape = [
        (args.seq_length * args.micro_batch_size) // args.context_parallel_size,
        config.hidden_size,
    ]

    if is_te_min_version('2.7.0'):
        UserBufferQuantizationMode = te_module.base.UserBufferQuantizationMode
        quantization_modes = [UserBufferQuantizationMode.FP8 if args.fp8 else UserBufferQuantizationMode.NONE]
        if args.fp8 is not None and args.first_last_layers_bf16 and (args.num_layers_at_start_in_bf16 > 0
                                                                     or args.num_layers_at_end_in_bf16 > 0):
            quantization_modes.append(UserBufferQuantizationMode.NONE)
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            quantization_modes=quantization_modes,
        )
    elif is_te_min_version('1.9.0'):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=args.tensor_model_parallel_size,
            use_fp8=(args.fp8 is not None),
        )


def warmup_jit_function(config, args):
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    bias = torch.rand(config.ffn_hidden_size // config.tensor_model_parallel_size, dtype=dtype, device='cuda')
    input_tensor = torch.rand(
        (
            args.seq_length // config.context_parallel_size,
            args.micro_batch_size,
            config.ffn_hidden_size // config.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device='cuda',
    )
    # Warmup JIT fusions with the input_tensor grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input_tensor.requires_grad = bias_grad, input_grad
        for _ in range(5):
            if config.swiglu:
                output = bias_swiglu(input_tensor, bias)
            else:
                output = bias_gelu(bias, input_tensor)
    del bias, input_tensor, output

    # Warmup fused bias+dropout+add
    if config.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input_tensor = torch.rand(
        (seq_length // config.context_parallel_size, args.micro_batch_size, config.hidden_size),
        dtype=dtype,
        device='cuda',
    )
    residual = torch.rand(
        (seq_length // config.context_parallel_size, args.micro_batch_size, config.hidden_size),
        dtype=dtype,
        device='cuda',
    )
    bias = torch.rand((config.hidden_size), dtype=dtype, device='cuda').expand_as(residual)
    dropout_rate = 0.1
    # Warmup JIT fusions with the input_tensor grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        input_tensor.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train([input_tensor, bias], residual, dropout_rate)
    del bias, input_tensor, residual, output
    torch.cuda.empty_cache()


def get_batch_on_this_cp_rank(args, batch: Dict[str, Any]):
    """Slice batch input along sequence dimension into multiple chunks,
    which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        keys = ['labels', 'position_ids', 'loss_scale']
        if not args.is_multimodal:
            # Multimodal models will handle CP in input_embeds.
            keys.append('input_ids')

        packed_seq_params = batch.get('packed_seq_params')
        for key, val in batch.items():
            if key not in keys:
                continue
            if args.task_type == 'seq_cls' and key == 'labels':
                continue
            if val is not None:
                batch[key] = split_cp_inputs(val, getattr(packed_seq_params, 'cu_seqlens_q', None), -1)

    return batch
