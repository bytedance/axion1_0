# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import shutil
import torch.distributed as dist
from transformers.utils import strtobool
from typing import List, Optional, Union

from swift.megatron.arguments import MegatronExportArguments
from swift.megatron.convert import test_convert_precision
from swift.megatron.model.aip_export import save_aip_regression_head
from swift.megatron.model import get_mcore_model
from swift.megatron.utils import (load_mcore_checkpoint, materialize_fsdp_dtensor_params, prepare_mcore_model,
                                  prepare_models_for_mcore_load, save_mcore_checkpoint)
from swift.pipelines import SwiftPipeline, prepare_model_template
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger, is_master

logger = get_logger()


class MegatronExport(SwiftPipeline):
    args_class = MegatronExportArguments
    args: args_class

    def run(self):
        os.environ['DISABLE_MP_DDP'] = 'true'
        args = self.args
        if args.to_hf:
            self.convert_mcore2hf()
        elif args.to_mcore:
            self.convert_hf2mcore()

    def convert_mcore2hf(self) -> None:
        args = self.args
        args.experts_impl = 'eager'
        # DCP-direct export (MS_SWIFT_EXPORT_NO_FSDP=1): load the fsdp_dtensor checkpoint straight
        # into a plain single-process model (no Megatron-FSDP wrap) — much faster and fits large
        # models on few GPUs. Parallelism was already forced to 1 in MegatronArguments; here we set
        # the ckpt_format and (optionally) the base-model source for config/tokenizer.
        no_fsdp_export = os.environ.get('MS_SWIFT_EXPORT_NO_FSDP', '0').lower() in ('1', 'true')
        if no_fsdp_export:
            args.ckpt_format = 'torch_dist'
            _base = os.environ.get('BASE_MODEL')
            if _base:
                args.model = _base
                logger.info(f'[export] BASE_MODEL override: args.model={_base}')
            logger.info('[export] MS_SWIFT_EXPORT_NO_FSDP=1: single-process DCP-direct load (TP1/EP1/PP1).')
        download_model = args.model is not None
        _, template = prepare_model_template(args, load_model=False, download_model=download_model)
        self.processor = template.processor
        hf_config = self.processor.model_info.config
        mg_model = get_mcore_model(args, hf_config)[0]
        logger.info('Megatron model created successfully.')
        bridge = mg_model.config.bridge
        if args.mcore_model is not None and no_fsdp_export:
            import glob as _glob
            from swift.megatron.utils import dcp_direct_load_backbone
            from swift.megatron.utils.megatron_lm_utils import _load_iteration
            tracker = os.path.join(args.mcore_model, 'latest_checkpointed_iteration.txt')
            it = _load_iteration(tracker)
            ckpt_dir = os.path.join(args.mcore_model, f'iter_{it:07d}')
            if not os.path.exists(os.path.join(ckpt_dir, '.metadata')):
                # No tracker (or wrong iter): fall back to the last iter_* dir with a .metadata.
                iters = sorted(_glob.glob(os.path.join(args.mcore_model, 'iter_*')))
                iters = [d for d in iters if os.path.exists(os.path.join(d, '.metadata'))]
                if not iters:
                    raise FileNotFoundError(f'No iter_*/.metadata under {args.mcore_model}')
                ckpt_dir = iters[-1]
            logger.info(f'[export] DCP-direct loading from {ckpt_dir}')
            n, _missing, _unexpected = dcp_direct_load_backbone(mg_model, ckpt_dir)
            logger.info(f'[export] DCP-direct loaded {n} tensors into mcore model.')
        elif args.mcore_model is not None:
            load_models = prepare_models_for_mcore_load(args, [mg_model], args.mcore_model)
            load_mcore_checkpoint(args, load_models, load_arg='mcore_model')
            materialize_fsdp_dtensor_params([mg_model])
        elif args.model is not None:
            bridge.load_weights([mg_model], args.model_info.model_dir)
        else:
            raise ValueError('Please specify `--mcore_model` or `--model`.')
        if args.adapters or args.mcore_adapter is not None:
            peft_model = prepare_mcore_model(args, mg_model)
            if args.mcore_adapter is not None:
                load_mcore_checkpoint(args, [mg_model], load_arg='mcore_adapter')
            elif args.adapters:
                assert len(args.adapters) == 1, 'Currently only support one adapter'
                bridge.load_weights([mg_model], args.adapters[0], peft_format=True)
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Converting weights and saving the model...')
        save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
        bridge.save_weights(
            [mg_model],
            args.output_dir,
            peft_format=save_peft_format,
            args=args,
            processor=self.processor,
        )
        if is_master():
            if args.ckpt_dir:
                args_path = os.path.join(args.ckpt_dir, 'args.json')
                shutil.copy(args_path, os.path.join(args.output_dir, 'args.json'))
            elif args.mcore_model is not None:
                args_path = os.path.join(args.mcore_model, 'args.json')
                if os.path.isfile(args_path):
                    shutil.copy(args_path, os.path.join(args.output_dir, 'args.json'))
            else:
                args.save_args(args.output_dir)
        save_aip_regression_head(
            mg_model,
            args.output_dir,
            args,
            mcore_model=args.mcore_model,
        )
        if args.test_convert_precision:
            with disable_safe_ddp_context_use_barrier():
                if save_peft_format:
                    kwargs = {'adapters': [args.output_dir]}
                else:
                    kwargs = {'model': args.output_dir, 'torch_dtype': None, 'adapters': []}
                device_map = args.device_map or 'auto'
                hf_model, template = prepare_model_template(
                    args, device_map=device_map, **kwargs) if is_master() else (None, template)
            test_convert_precision(args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)
            dist.barrier()

    def convert_hf2mcore(self) -> None:
        args = self.args
        args.experts_impl = 'eager'
        download_model = args.model is not None
        _, template = prepare_model_template(args, load_model=False, download_model=download_model)
        self.processor = template.processor
        hf_config = self.processor.model_info.config
        mg_model = get_mcore_model(args, hf_config)[0]
        logger.info('Megatron model created successfully.')
        bridge = mg_model.config.bridge
        if args.model is not None:
            bridge.load_weights([mg_model], args.model_info.model_dir)
        elif args.mcore_model is not None:
            load_mcore_checkpoint(args, [mg_model], load_arg='mcore_model')
        else:
            raise ValueError('Please specify `--mcore_model` or `--model`.')
        dist.barrier()
        if args.adapters or args.mcore_adapter is not None:
            peft_model = prepare_mcore_model(args, mg_model)
            if args.adapters:
                assert len(args.adapters) == 1, 'Currently only support one adapter'
                bridge.load_weights([mg_model], args.adapters[0], peft_format=True)
            elif args.mcore_adapter is not None:
                load_mcore_checkpoint(args, [mg_model], load_arg='mcore_adapter')
            if args.merge_lora:
                logger.info('Merge LoRA...')
                mg_model = peft_model.merge_and_unload()
        logger.info('Successfully transferred HF model weights to MG model.')
        _test_convert_precision = strtobool(os.getenv('SWIFT_TEST_CONVERT_PRECISION', '0'))
        if not _test_convert_precision:
            args.save_args(args.output_dir)
            logger.info('Saving the model...')
            save_peft_format = args.tuner_type == 'lora' and not args.merge_lora
            save_mcore_checkpoint(args, [mg_model], peft_format=save_peft_format)
        # hf_model does not support loading args.mcore_adapter, so test_convert_precision cannot be performed
        support_convert_precision = args.mcore_adapter is None
        if args.test_convert_precision:
            if support_convert_precision:
                with disable_safe_ddp_context_use_barrier():
                    device_map = args.device_map or 'auto'
                    hf_model, template = prepare_model_template(
                        args, device_map=device_map) if is_master() else (None, template)
                test_convert_precision(args, hf_model, mg_model, template, test_convert_dtype=args.test_convert_dtype)
                dist.barrier()
            else:
                logger.warning('Skip test_convert_precision because `--mcore_adapter` is specified.')


def megatron_export_main(args: Optional[Union[List[str], MegatronExportArguments]] = None):
    return MegatronExport(args).main()
