# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
from datetime import datetime
from types import SimpleNamespace

import torch
from tqdm import tqdm

from swift.megatron.utils import reduce_max_stat_across_model_parallel_group
from swift.utils import JsonlWriter, format_time, get_logger, is_last_rank
from .base import MegatronCallback

logger = get_logger()


class PrintCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.training_bar = None
        self.eval_bar = None
        self.jsonl_writer = None
        self.is_write_rank = is_last_rank()
        self.last_log_step = 0
        self.last_log_time = None

    def on_train_begin(self):
        self.training_bar = tqdm(
            total=self.args.train_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Train: ')
        self.start_step = self.state.iteration
        self.training_bar.update(self.state.iteration)
        self.current_step = self.state.iteration
        self.start_time = time.time()
        self.last_log_step = self.state.iteration
        self.last_log_time = self.start_time
        logging_path = os.path.join(self.args.output_dir, 'logging.jsonl')
        logger.info(f'logging_path: {logging_path}')
        self.jsonl_writer = JsonlWriter(logging_path, enable_async=True, write_on_rank='last')

    def on_train_end(self):
        self.training_bar.close()
        self.training_bar = None

    def on_step_end(self):
        n_step = self.state.iteration - self.current_step
        self.current_step = self.state.iteration
        self.training_bar.update(n_step)

    def on_eval_begin(self):
        self.eval_bar = tqdm(
            total=self.args.eval_iters, dynamic_ncols=True, disable=not self.is_write_rank, desc='Evaluate: ')

    def on_eval_end(self):
        self.eval_bar.close()
        self.eval_bar = None

    def on_eval_step(self):
        self.eval_bar.update()

    @staticmethod
    def _format_metric_value(value):
        if isinstance(value, float):
            return f'{value:.6E}'
        return str(value)

    def _get_loss_scale(self):
        optimizer = getattr(self.trainer, 'optimizer', None)
        loss_scale = None
        if optimizer is not None:
            get_loss_scale = getattr(optimizer, 'get_loss_scale', None)
            if callable(get_loss_scale):
                loss_scale = get_loss_scale()
            else:
                loss_scaler = getattr(optimizer, 'loss_scaler', None)
                loss_scale = getattr(loss_scaler, 'loss_scale', None)
        if isinstance(loss_scale, torch.Tensor):
            loss_scale = loss_scale.detach().float().max().item()
        if loss_scale is None:
            loss_scale = 1.0
        return float(loss_scale)

    def _build_throughput_args(self):
        config = getattr(self.trainer, 'config', None)

        def _get(name, default=None):
            if config is not None and hasattr(config, name):
                return getattr(config, name)
            return getattr(self.args, name, default)

        return SimpleNamespace(
            world_size=getattr(self.args, 'world_size', None),
            global_batch_size=getattr(self.args, 'global_batch_size', None),
            group_query_attention=_get('num_query_groups') != _get('num_attention_heads'),
            num_query_groups=_get('num_query_groups'),
            num_attention_heads=_get('num_attention_heads'),
            num_experts=_get('num_moe_experts'),
            moe_layer_freq=_get('moe_layer_freq', 1),
            num_layers=_get('num_layers'),
            mtp_num_layers=_get('mtp_num_layers'),
            moe_router_topk=_get('moe_router_topk', 1),
            moe_ffn_hidden_size=_get('moe_ffn_hidden_size'),
            ffn_hidden_size=_get('ffn_hidden_size'),
            moe_latent_size=_get('moe_latent_size'),
            moe_shared_expert_intermediate_size=_get('moe_shared_expert_intermediate_size'),
            swiglu=bool(_get('swiglu', _get('gated_linear_unit', False))),
            multi_latent_attention=bool(_get('multi_latent_attention', False)),
            q_lora_rank=_get('q_lora_rank'),
            hidden_size=_get('hidden_size'),
            qk_head_dim=_get('qk_head_dim'),
            qk_pos_emb_head_dim=_get('qk_pos_emb_head_dim'),
            kv_channels=_get('kv_channels'),
            attention_output_gate=bool(_get('attention_output_gate', False)),
            experimental_attention_variant=_get('experimental_attention_variant'),
            linear_attention_freq=_get('linear_attention_freq'),
            linear_key_head_dim=_get('linear_key_head_dim'),
            linear_value_head_dim=_get('linear_value_head_dim'),
            linear_num_key_heads=_get('linear_num_key_heads'),
            linear_num_value_heads=_get('linear_num_value_heads'),
            linear_conv_kernel_dim=_get('linear_conv_kernel_dim'),
            seq_length=getattr(self.args, 'seq_length', None) or getattr(self.args, 'max_length', None),
            padded_vocab_size=_get('padded_vocab_size'),
            is_hybrid_model=bool(_get('is_hybrid_model', False)),
            hybrid_layer_pattern=_get('hybrid_layer_pattern'),
            mamba_state_dim=_get('mamba_state_dim', 128),
            mamba_head_dim=_get('mamba_head_dim', 64),
            mamba_num_groups=_get('mamba_num_groups', 8),
            mamba_num_heads=_get('mamba_num_heads', 128),
        )

    @staticmethod
    def _clone_megatron_args(args):
        try:
            return SimpleNamespace(**vars(args))
        except TypeError:
            return args

    @staticmethod
    def _get_world_size(args):
        world_size = getattr(args, 'world_size', None)
        if world_size is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        return max(int(world_size or 1), 1)

    def _compute_tflops(self, time_per_iteration, actual_seq_length=None, actual_samples_per_step=None):
        if not getattr(self.args, 'log_throughput', False) or time_per_iteration <= 0:
            return None
        try:
            from megatron.training.training import num_floating_point_operations
        except Exception as e:
            logger.warning_once(f'Cannot log throughput: failed to import Megatron FLOPs helper: {e}')
            return None
        try:
            try:
                from megatron.training import get_args
                megatron_args = self._clone_megatron_args(get_args())
            except Exception:
                megatron_args = self._build_throughput_args()
            if actual_seq_length is not None and actual_seq_length > 0:
                megatron_args.seq_length = actual_seq_length
            world_size = self._get_world_size(megatron_args)
            batch_size = (
                actual_samples_per_step
                if actual_samples_per_step is not None and actual_samples_per_step > 0
                else getattr(megatron_args, 'global_batch_size', self.args.global_batch_size)
            )
            flops = num_floating_point_operations(megatron_args, batch_size)
            return flops / (time_per_iteration * 1e12 * world_size)
        except Exception as e:
            logger.warning_once(f'Cannot log throughput: failed to compute TFLOPS/GPU: {e}')
            return None

    def _compute_tokens_per_second(self, time_per_iteration, actual_tokens_per_step=None):
        if not getattr(self.args, 'log_throughput', False) or time_per_iteration <= 0:
            return None
        if actual_tokens_per_step is None or actual_tokens_per_step <= 0:
            return None
        return actual_tokens_per_step / time_per_iteration / self._get_world_size(self.args)

    def _format_train_log(self, logs, elapsed, interval_time_per_iteration):
        args = self.args
        state = self.state
        line = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        line += ' iteration {:8d}/{:8d} |'.format(state.iteration, args.train_iters)
        line += ' consumed samples: {:12d} |'.format(state.consumed_train_samples)
        line += ' elapsed time per iteration (ms): {:.1f} |'.format(interval_time_per_iteration * 1000.0)
        line += f" memory(GiB): {logs['memory(GiB)']} |"
        line += f" elapsed time: {logs['elapsed_time']} | remaining time: {logs['remaining_time']} |"
        if logs.get('learning_rate') is not None:
            line += f" learning rate: {logs['learning_rate']:.6E} |"
        line += f' global batch size: {args.global_batch_size:5d} |'
        if logs.get('throughput') is not None:
            line += f" throughput per GPU (TFLOP/s/GPU): {logs['throughput']:.1f} |"
        if logs.get('tokens_per_second_per_gpu') is not None:
            line += f" token/s/GPU: {logs['tokens_per_second_per_gpu']:.1f} |"

        skipped_keys = {
            'iteration', 'elapsed_time', 'remaining_time', 'memory(GiB)', 'train_speed(s/it)', 'learning_rate',
            'grad_norm', 'loss_scale', 'params_norm', 'throughput', 'tokens_per_second_per_gpu',
            'actual_seq_length', 'actual_tokens_per_step', 'actual_samples_per_step'
        }
        for key, value in logs.items():
            if key in skipped_keys or key.endswith('_ms') or key.endswith('_pct'):
                continue
            display_key = 'lm loss' if key == 'loss' and 'reg_loss' in logs else key
            line += f' {display_key}: {self._format_metric_value(value)} |'

        line += f' loss scale: {self._get_loss_scale():.1f} |'
        if logs.get('grad_norm') is not None:
            line += f" grad norm: {logs['grad_norm']:.3f} |"
        if logs.get('params_norm') is not None:
            line += f" params norm: {logs['params_norm']:.3f} |"
        # The new training loop returns only successful optimizer-step state; keep
        # the legacy stdout fields for operational log parsers.
        line += ' number of skipped iterations: {:3d} |'.format(0)
        line += ' number of nan iterations: {:3d} |'.format(0)
        return line

    def on_log(self, logs):
        state = self.state
        args = self.args
        logs['iteration'] = f'{state.iteration}/{args.train_iters}'
        now = time.time()
        elapsed = now - self.start_time
        logs['elapsed_time'] = format_time(elapsed)
        n_steps = state.iteration - self.start_step
        train_speed = elapsed / n_steps if n_steps > 0 else 0.0
        logs['remaining_time'] = format_time((args.train_iters - state.iteration) * train_speed)
        memory = reduce_max_stat_across_model_parallel_group(torch.cuda.max_memory_reserved() / 1024**3)
        logs['memory(GiB)'] = round(memory, 2)
        logs['train_speed(s/it)'] = round(train_speed, 6)
        interval_steps = max(1, state.iteration - self.last_log_step)
        interval_elapsed = now - (self.last_log_time or self.start_time)
        interval_time_per_iteration = interval_elapsed / interval_steps
        throughput = self._compute_tflops(
            interval_time_per_iteration, logs.get('actual_seq_length'), logs.get('actual_samples_per_step'))
        if throughput is not None:
            logs['throughput'] = throughput
        tokens_per_second = self._compute_tokens_per_second(
            interval_time_per_iteration, logs.get('actual_tokens_per_step'))
        if tokens_per_second is not None:
            logs['tokens_per_second_per_gpu'] = tokens_per_second
        logs = {k: round(v, 8) if isinstance(v, float) else v for k, v in logs.items()}
        self.jsonl_writer.append(logs)
        if self.is_write_rank:
            if any(key.startswith('eval_') for key in logs):
                self.training_bar.write(str(logs))
            else:
                self.training_bar.write(self._format_train_log(logs, elapsed, interval_time_per_iteration))
            self.last_log_step = state.iteration
            self.last_log_time = now
