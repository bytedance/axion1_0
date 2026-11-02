# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import importlib.metadata
import inspect
import logging
import os
import torch
import torch.distributed as dist
from contextlib import contextmanager
from copy import copy
from packaging import version
from tqdm import tqdm
from transformers.modeling_utils import custom_object_save
from transformers.utils import is_torch_npu_available
from transformers.utils.versions import require_version

from swift.model import get_model_processor, save_checkpoint
from swift.utils import (HfConfigFactory, disable_safe_ddp_context_use_barrier, get_logger, get_modules_to_not_convert,
                         get_multimodal_target_regex, is_master, split_list)

logger = get_logger()


def _patch__batched_p2p_ops():
    from megatron.core.pipeline_parallel import p2p_communication

    _batched_p2p_ops_origin = p2p_communication._batched_p2p_ops

    def _batched_p2p_ops(**kwargs):
        kwargs['group'] = None
        return _batched_p2p_ops_origin(**kwargs)

    p2p_communication._batched_p2p_ops = _batched_p2p_ops


def _patch_torch_FileSystemReader():
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.futures import Future
    _origin_read_data = FileSystemReader.read_data
    _origin__slice_file = FileSystemReader._slice_file
    READER_MAX_WORKERS = int(os.environ.get('MCORE_READER_MAX_WORKERS', '16'))

    @contextmanager
    def _patch__slice_file(prog_bar):

        def _slice_file(self, *args, **kwargs):
            prog_bar.update()
            return _origin__slice_file(self, *args, **kwargs)

        FileSystemReader._slice_file = _slice_file
        try:
            yield
        finally:
            FileSystemReader._slice_file = _origin__slice_file

    def read_data(self, plan, planner):

        def _worker(plan_shard):
            _origin_read_data(self, plan_shard, planner)

        prog_bar = tqdm(total=len(plan.items), dynamic_ncols=True, desc='Loading: ')
        plan_shards = split_list(plan.items, READER_MAX_WORKERS, contiguous=False)
        with _patch__slice_file(prog_bar):
            with concurrent.futures.ThreadPoolExecutor(max_workers=READER_MAX_WORKERS) as pool:
                futures = []
                for i in range(READER_MAX_WORKERS):
                    plan_shard = copy(plan)
                    plan_shard.items = plan_shards[i]
                    futures.append(pool.submit(_worker, plan_shard))
                concurrent.futures.wait(futures)
        prog_bar.close()
        fut: Future = Future()
        fut.set_result(None)
        return fut

    FileSystemReader.read_data = read_data


def _patch_validate_non_overlapping_shards_metadata():
    # too slow
    from torch.distributed._shard.sharded_tensor import api
    from torch.distributed._shard.sharding_spec import api as api2
    from torch.distributed.checkpoint import default_planner

    def validate_non_overlapping_shards_metadata(*args, **kwargs):
        pass

    api.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata
    api2.validate_non_overlapping_shards_metadata = validate_non_overlapping_shards_metadata

    def _validate_global_plan(*args, **kwargs):
        return True

    default_planner._validate_global_plan = _validate_global_plan


def _patch_unified_memory():
    if is_torch_npu_available():
        return

    mcore_015 = version.parse(importlib.metadata.version('megatron-core')) >= version.parse('0.15.0rc0')
    if not mcore_015:
        return
    from torch.utils import cpp_extension
    load_inline = cpp_extension.load_inline

    def _new_load_inline(*args, **kwargs):
        name = kwargs.get('name')
        if name == 'managed_alloc_runtime':
            raise RuntimeError
        return load_inline(*args, **kwargs)

    # not create unified memory mempool
    cpp_extension.load_inline = _new_load_inline
    try:
        from megatron.core.inference import unified_memory
    except Exception:
        pass
    finally:
        cpp_extension.load_inline = load_inline


def _patch_mcore_bridge():
    require_version('mcore-bridge>=1.3.0', 'please install mcore-bridge via `pip install mcore-bridge -U`')
    import mcore_bridge
    from mcore_bridge import GPTBridge
    logger.info(f'mcore_bridge.__version__: {mcore_bridge.__version__}')
    origin_save_weights = GPTBridge.save_weights

    def save_weights(
        self,
        mg_models,
        output_dir: str,
        peft_format: bool = False,
        max_shard_size: str = '5GB',
        args=None,
        processor=None,
    ) -> None:
        origin_save_weights(self, mg_models, output_dir, peft_format=peft_format, max_shard_size=max_shard_size)
        if processor is None or args is None:
            return
        hf_config = self.config.hf_config
        hf_config = copy(hf_config)
        if is_master() and not hasattr(self, 'hf_model'):
            if hasattr(self, 'get_hf_meta_model'):
                self.hf_model = self.get_hf_meta_model()
                self.hf_model.model_meta = processor.model_meta
                self.hf_model.model_info = processor.model_info
            else:
                with torch.device('meta'), disable_safe_ddp_context_use_barrier():
                    self.hf_model = get_model_processor(
                        args.model_dir, model_type=args.model_type, return_dummy_model=True)[0]

        if is_master():
            if peft_format:
                peft_config = copy(mg_models[0].peft_config[self._adapter_name])
                if self.config.task_type == 'seq_cls':
                    peft_config.task_type = 'SEQ_CLS'
                if self.is_multimodal and 'all-linear' in args.target_modules:
                    peft_config.target_modules = get_multimodal_target_regex(
                        self.hf_model,
                        freeze_llm=args.freeze_llm,
                        freeze_vit=args.freeze_vit,
                        freeze_aligner=args.freeze_aligner,
                        include_embedding='all-embedding' in args.target_modules,
                        exclude_router='all-router' not in args.target_modules)
                else:
                    assert not isinstance(peft_config.target_modules, str), (
                        'target_regex is not currently supported for LoRA conversion. Please set `--merge_lora true`.')
                    peft_config.target_modules = self._peft_target_modules
                peft_config.modules_to_save = self._peft_modules_to_save
                peft_config.save_pretrained(output_dir)
            else:
                config = self.config
                llm_config = HfConfigFactory.get_text_config(hf_config)
                if config.mtp_num_layers:
                    for key in ['num_nextn_predict_layers', 'mtp_num_hidden_layers']:
                        if hasattr(llm_config, key):
                            setattr(llm_config, key, config.mtp_num_layers)
                            break
                    else:
                        llm_config.num_nextn_predict_layers = config.mtp_num_layers
                if config.fp8 is not None and config.fp8_recipe == 'blockwise' and config.fp8_param:
                    if getattr(hf_config, 'quantization_config', None) is None:
                        from transformers.utils.quantization_config import FineGrainedFP8Config
                        modules_to_not_convert = get_modules_to_not_convert(self.hf_model)
                        if hasattr(self, '_fp8_skip_modules'):
                            modules_to_not_convert = (modules_to_not_convert or []) + list(self._fp8_skip_modules)
                        hf_config.quantization_config = FineGrainedFP8Config(
                            modules_to_not_convert=modules_to_not_convert)
                elif hasattr(hf_config, 'quantization_config'):
                    del hf_config.quantization_config
                hf_config.save_pretrained(output_dir)
                if getattr(self.hf_model, '_auto_class') is not None:
                    try:
                        custom_object_save(self.hf_model, output_dir, config=hf_config)
                    except FileNotFoundError as e:
                        logger.error(f'custom_object_save Error: {e}')
                save_checkpoint(
                    None,
                    processor,
                    output_dir,
                    model_dirs=[args.model_dir],
                    additional_saved_files=self.hf_model.model_meta.additional_saved_files)
            logger.info(f'Successfully saved `safetensors` model weights in `{output_dir}`.')
        dist.barrier()  # Ensure all weights are saved completely

    GPTBridge.save_weights = save_weights


def _patch_nsys_nvtx():
    if os.environ.get('MS_SWIFT_NSYS_NVTX', '0') != '1':
        return

    from swift.megatron.utils.nvtx_profiler import is_gui_mode, nvtx_label, patch_method_nvtx, wrap_nvtx

    def _forward_step_name(*_args, **kwargs):
        mb = kwargs.get('current_microbatch')
        suffix = f' mb={mb}' if mb is not None else ''
        return nvtx_label('forward_step', suffix=suffix)

    def _wrap_forward_step_mark(orig):
        from functools import wraps
        from swift.megatron.utils.nvtx_profiler import nvtx_mark

        @wraps(orig)
        def wrapper(*args, **kwargs):
            if is_gui_mode():
                mb = kwargs.get('current_microbatch')
                suffix = f' mb={mb}' if mb is not None else ''
                nvtx_mark(nvtx_label('forward_step', suffix=suffix))
            return orig(*args, **kwargs)

        return wrapper

    try:
        from megatron.core.pipeline_parallel import schedules
        if is_gui_mode():
            schedules.forward_step = _wrap_forward_step_mark(schedules.forward_step)
        else:
            patch_method_nvtx(schedules, 'forward_step', '', dynamic_name=_forward_step_name)
        schedules.backward_step = wrap_nvtx(schedules.backward_step, nvtx_label('backward_step'))
        if not is_gui_mode():
            schedules.forward_step_calc_loss = wrap_nvtx(
                schedules.forward_step_calc_loss, nvtx_label('loss_compute'))
    except Exception as e:
        logger.warning(f'Failed to patch forward/backward NVTX for nsys: {e}')

    try:
        from megatron.core.distributed.finalize_model_grads import finalize_model_grads
        import megatron.core.distributed.finalize_model_grads as finalize_model_grads_mod
        if not is_gui_mode():
            finalize_model_grads_mod.finalize_model_grads = wrap_nvtx(
                finalize_model_grads, nvtx_label('finalize_grads'))
    except Exception as e:
        logger.warning(f'Failed to patch finalize_model_grads NVTX for nsys: {e}')

    try:
        from megatron.core.transformer.moe.moe_layer import MoELayer

        if is_gui_mode():
            from swift.megatron.utils.nvtx_profiler import make_ep_phase_name

            # 3 GUI-visible levels (no [STEP]): [FB] > [EP] Lxx moe > route/preprocess/dispatch/compute/combine/post.
            patch_method_nvtx(MoELayer, 'forward', '', dynamic_name=make_ep_phase_name('moe'))
            patch_method_nvtx(MoELayer, 'route', '', dynamic_name=make_ep_phase_name('route'))
            patch_method_nvtx(MoELayer, 'preprocess', '', dynamic_name=make_ep_phase_name('preprocess'))
            patch_method_nvtx(MoELayer, 'dispatch', '', dynamic_name=make_ep_phase_name('dispatch'))
            patch_method_nvtx(
                MoELayer, 'routed_experts_compute', '', dynamic_name=make_ep_phase_name('compute'))
            patch_method_nvtx(MoELayer, 'combine', '', dynamic_name=make_ep_phase_name('combine'))
            patch_method_nvtx(MoELayer, 'postprocess', '', dynamic_name=make_ep_phase_name('postprocess'))
        else:
            from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
            for method, key in (
                ('dispatch_preprocess', 'moe/dispatch_preprocess'),
                ('token_dispatch', 'ep_dispatch'),
                ('dispatch_postprocess', 'moe/dispatch_postprocess'),
                ('combine_preprocess', 'moe/combine_preprocess'),
                ('token_combine', 'ep_combine'),
                ('combine_postprocess', 'moe/combine_postprocess'),
            ):
                patch_method_nvtx(MoEAlltoAllTokenDispatcher, method, nvtx_label(key))

            def _dtoh_nvtx_name(self, point, tokens_per_expert=None):
                return nvtx_label('moe/dtoh', suffix=f'/{point}')

            patch_method_nvtx(
                MoEAlltoAllTokenDispatcher,
                '_maybe_dtoh_and_synchronize',
                '',
                dynamic_name=_dtoh_nvtx_name,
            )
    except Exception as e:
        logger.warning(f'Failed to patch MoE layer NVTX for nsys: {e}')

    try:
        from megatron.core.transformer.attention import Attention
        from megatron.core.transformer.transformer_layer import TransformerLayer

        if is_gui_mode():
            from functools import wraps
            from megatron.core.transformer.identity_op import IdentityOp

            from swift.megatron.utils.nvtx_profiler import (
                layer_phase_label,
                make_layer_phase_name,
                wrap_module_forward_nvtx,
            )

            # Non-MoE per-layer blocks (level 2 under [FB]): attention + norms + SP proj.
            patch_method_nvtx(
                Attention, 'forward', '', dynamic_name=make_layer_phase_name('ATTN', ''))
            patch_method_nvtx(
                TransformerLayer,
                '_forward_pre_mlp_layernorm',
                '',
                dynamic_name=make_layer_phase_name('NORM', 'pre_mlp'),
            )

            _orig_transformer_layer_init = TransformerLayer.__init__

            @wraps(_orig_transformer_layer_init)
            def _transformer_layer_init(self, *args, **kwargs):
                _orig_transformer_layer_init(self, *args, **kwargs)
                if not isinstance(self.input_layernorm, IdentityOp):
                    wrap_module_forward_nvtx(
                        self.input_layernorm,
                        layer_phase_label('NORM', self.layer_number, 'input'),
                    )

            TransformerLayer.__init__ = _transformer_layer_init

            _orig_attention_init = Attention.__init__

            @wraps(_orig_attention_init)
            def _attention_init(self, *args, **kwargs):
                _orig_attention_init(self, *args, **kwargs)
                linear_proj = getattr(self, 'linear_proj', None)
                if linear_proj is not None and not isinstance(linear_proj, IdentityOp):
                    wrap_module_forward_nvtx(
                        linear_proj,
                        layer_phase_label('SP', self.layer_number, 'proj'),
                    )

            Attention.__init__ = _attention_init
    except Exception as e:
        logger.warning(f'Failed to patch TransformerLayer NVTX for nsys: {e}')

    try:
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        P2PCommunicator._communicate = wrap_nvtx(P2PCommunicator._communicate, nvtx_label('pp_p2p'))
    except Exception as e:
        logger.warning(f'Failed to patch PP P2P NVTX for nsys: {e}')


def init_megatron_env():
    os.environ.pop('VLLM_USE_MODELSCOPE', None)
    logging_level = logging.root.level
    _patch_unified_memory()
    _patch_mcore_bridge()
    _patch__batched_p2p_ops()
    _patch_nsys_nvtx()
    logging.root.setLevel(logging_level)  # revert logger level
    try:
        _patch_torch_FileSystemReader()
    except Exception:
        logger.warning('Failed to patch FileSystemReader.')
    try:
        _patch_validate_non_overlapping_shards_metadata()
    except Exception:
        logger.warning('Patch validate_non_overlapping_shards_metadata failed.')
        pass
    import megatron.core
    logger.info(f'megatron.core.__version__: {megatron.core.__version__}')
