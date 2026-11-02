# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn
from collections import defaultdict
from functools import partial
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronTrainer(BaseMegatronTrainer):

    def prepare_model(self):
        super().prepare_model()
        # BT-SFT reads the score off the LM head at the <score_k> position, so it
        # needs the 11 score-token vocab ids. Megatron's global tokenizer is not
        # initialized in this repo, so resolve the ids from the swift template
        # tokenizer here (same pattern as reranker_trainer.prepare_model) and stash
        # them on each BT-SFT model for _bt_sft_score to slice the logits.
        if any(getattr(m, '_bt_sft', False) for m in self.unwrapped_models):
            from ..model.bt_score import (get_score_token_ids, get_cg_marker_ids, get_score_unk_id,
                                          get_per_cg_score_token_ids)
            tok = self.template.tokenizer
            for model in self.unwrapped_models:
                if not getattr(model, '_bt_sft', False):
                    continue
                nb = getattr(model, '_num_score_buckets', 11)
                num_cg = getattr(model, '_num_cg', 1)
                per_cg = getattr(model, '_per_cg_score_tokens', False) and num_cg > 1
                if per_cg:
                    # Per-CG: each CG owns <cg_c_score_0..N-1>; resolve the [num_cg, nb] id matrix.
                    # (The shared <score_k> tokens are not registered in this mode.)
                    pc_ids = get_per_cg_score_token_ids(tok, num_cg, nb)
                    model._per_cg_score_token_ids = torch.tensor(pc_ids, dtype=torch.long)
                    logger.info(f'[MegatronTrainer] per-CG score tokens {num_cg}x{nb}; '
                                f'cg0 ids={pc_ids[0]}')
                else:
                    ids = get_score_token_ids(tok, nb)
                    model._score_token_ids = torch.tensor(ids, dtype=torch.long)
                    logger.info(f'[MegatronTrainer] BT-SFT score_token_ids({nb})={ids}')
                # Multi-CG: resolve the <cg_c> markers + <score_unk> placeholder.
                if num_cg > 1:
                    cg_ids = get_cg_marker_ids(tok, num_cg)
                    model._cg_marker_ids = torch.tensor(cg_ids, dtype=torch.long)
                    model._score_unk_id = get_score_unk_id(tok)
                    logger.info(f'[MegatronTrainer] multi-CG markers({num_cg})={cg_ids} '
                                f'score_unk_id={model._score_unk_id}')

    def seq_cls_loss_func(self, output_tensor, *, labels: torch.Tensor, packed_seq_params=None, attention_mask=None):
        args = self.args
        if args.context_parallel_size > 1:
            raise ValueError('Currently `task_type="seq_cls"` does not support context parallelism.')
        logits = self.get_last_tokens(output_tensor, packed_seq_params, attention_mask)
        num_labels = args.num_labels
        acc = None
        if args.problem_type == 'regression':
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif args.problem_type == 'single_label_classification':
            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, num_labels)
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            acc = (logits.detach().argmax(dim=-1) == labels).float().mean()
        elif args.problem_type == 'multi_label_classification':
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            preds = logits.sigmoid() > 0.5
            acc = (labels == preds).all(dim=-1).float().mean()
        metric = {'loss': loss.detach().clone()}
        if acc is not None:
            metric['acc'] = acc
        metric = self._all_reduce_metric(metric)
        return loss, metric

    def loss_func(self,
                  output_tensor: torch.Tensor,
                  *,
                  labels: torch.Tensor,
                  loss_scale: Optional[torch.Tensor] = None,
                  channels: Optional[List[str]] = None,
                  packed_seq_params=None,
                  attention_mask: Optional[torch.Tensor] = None):
        args = self.args

        losses = output_tensor.float()
        loss_mask = labels != -100
        if args.enable_dft_loss:
            losses = losses * torch.exp(-losses.detach())
        if loss_scale is not None:
            losses = losses * loss_scale
        loss = torch.cat([torch.sum(losses * loss_mask).view(1), loss_mask.sum().view(1)])

        # Reduce loss for logging.
        reporting_loss = loss.detach().clone()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group(with_context_parallel=True))

        lm_loss = loss[0]
        lm_loss = lm_loss.clone()
        local_num_tokens = loss[1].detach().clone().to(torch.int)
        metrics = {'loss': reporting_loss}
        token_metric = self._actual_seq_length_metric(labels, packed_seq_params, attention_mask)
        if token_metric is not None:
            metrics['actual_seq_length'] = token_metric
            num_microbatches = max(int(getattr(args, 'num_microbatches', 1) or 1), 1)
            step_weight = token_metric.new_tensor(1.0 / num_microbatches)
            metrics['actual_tokens_per_step'] = torch.stack([token_metric[0], step_weight])
            metrics['actual_samples_per_step'] = torch.stack([token_metric[1], step_weight])
        if args.enable_channel_loss:
            metrics.update(self._compute_channel_loss(losses, loss_mask, channels, packed_seq_params))
        return (lm_loss, local_num_tokens, metrics)

    def _actual_seq_length_metric(self, labels, packed_seq_params=None, attention_mask=None):
        if labels is None:
            return None
        if packed_seq_params is not None and getattr(packed_seq_params, 'cu_seqlens_q', None) is not None:
            context_parallel_size = max(int(getattr(self.args, 'context_parallel_size', 1) or 1), 1)
            num_samples = int(getattr(packed_seq_params, 'num_samples', packed_seq_params.cu_seqlens_q.numel() - 1))
            if num_samples <= 0:
                return None
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1]
            token_count = (cu_seqlens[-1] - cu_seqlens[0]).detach().float() / context_parallel_size
            sample_count = token_count.new_tensor(float(num_samples) / context_parallel_size)
        elif attention_mask is not None:
            mask = attention_mask
            if mask.ndim > 2:
                mask = mask.reshape(mask.shape[0], -1)
            token_count = mask.to(dtype=torch.float32).sum()
            sample_count = token_count.new_tensor(float(mask.shape[0]))
        else:
            if labels.ndim == 1:
                sample_count_value = 1
            else:
                sample_count_value = labels.shape[0]
            token_count = labels.new_tensor(float(labels.numel()), dtype=torch.float32)
            sample_count = token_count.new_tensor(float(sample_count_value))
        metric = torch.stack([token_count, sample_count])
        torch.distributed.all_reduce(metric, group=mpu.get_data_parallel_group(with_context_parallel=True))
        return metric

    def aip_loss_func(
        self,
        lm_output: torch.Tensor,
        *,
        lm_labels: torch.Tensor,
        reg_logits: torch.Tensor,
        regression_labels: Optional[torch.Tensor],
        score_percentile: Optional[torch.Tensor],
        cg_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        loss_scale: Optional[torch.Tensor],
        channels,
        packed_seq_params,
    ):
        from swift.megatron.model.qwen3_vl_aip import compute_regression_loss

        args = self.args
        lm_loss, local_num_tokens, metric = self.loss_func(
            lm_output,
            labels=lm_labels,
            loss_scale=loss_scale,
            channels=channels,
            packed_seq_params=packed_seq_params,
            attention_mask=attention_mask,
        )
        lm_weight = getattr(args, 'lm_loss_weight', 1.0) if getattr(args, 'joint_lm_training', False) else 1.0
        if regression_labels is not None and reg_logits is not None:
            # The BT memory bank lives on the model that owns the score head: the
            # AIP regression head, or a BT-SFT model (no regression_head; the queue
            # is on _logit_queue). Match either so the queue is actually used.
            model_for_queue = next(
                (m for m in self.unwrapped_models
                 if hasattr(m, 'regression_head') or hasattr(m, '_logit_queue')), None)
            reg_loss, reg_metrics = compute_regression_loss(
                reg_logits,
                regression_labels,
                cg_ids,
                packed_seq_params,
                model=model_for_queue,
                attention_mask=attention_mask,
                score_percentile=score_percentile,
            )
            # BT-SFT bakes its own bt_loss_weight / huber_loss_weight into reg_loss inside
            # compute_regression_loss, so it must NOT be re-scaled by reg_loss_weight. Other
            # regression loss types keep the single reg_loss_weight knob.
            if getattr(model_for_queue, '_bt_sft', False):
                reg_weight = 1.0
            else:
                reg_weight = getattr(args, 'reg_loss_weight', 1.0)
            combined_loss = lm_weight * lm_loss + reg_weight * reg_loss
            metric['reg_loss'] = reg_loss.detach().clone()
            for k, v in reg_metrics.items():
                metric[k] = v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(
                    float(v), dtype=torch.float32, device=reg_loss.device)
        else:
            combined_loss = lm_weight * lm_loss
        return combined_loss, local_num_tokens, metric

    def _compute_channel_loss(self, losses, loss_mask, channels, packed_seq_params=None):
        args = self.args
        metrics = defaultdict(lambda: torch.tensor([0.0, 0.0], dtype=torch.float32, device=torch.cuda.current_device()))
        if args.padding_free:
            num_samples = packed_seq_params.num_samples
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            for i in range(cu_seqlens.shape[0] - 1):
                channel = None if channels is None else channels[i]
                slice_ = slice(cu_seqlens[i], cu_seqlens[i + 1])
                c_loss = losses[0, slice_][loss_mask[0, slice_]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]
        else:
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                c_loss = losses[i][loss_mask[i]]
                metrics[f'loss_{channel}'][0] += c_loss.detach().sum()
                metrics[f'loss_{channel}'][1] += c_loss.shape[0]

        # Synchronize keys to avoid getting stuck.
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_keys = [None] * torch.distributed.get_world_size(group=dp_cp_group)
        dist.all_gather_object(all_keys, list(metrics.keys()), group=dp_cp_group)
        new_metrics = {}
        for key in sorted(set().union(*all_keys)):
            new_metrics[key] = metrics[key]
        new_metrics = self._all_reduce_metric(new_metrics, torch.distributed.ReduceOp.SUM, group=dp_cp_group)
        return new_metrics

    def forward_step(self, data_iterator, model):
        from swift.megatron.utils.nvtx_profiler import is_gui_mode, nvtx_label, nvtx_range

        # Get the batch.
        vp_stage = model.module.module.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        loss_scale = data.pop('loss_scale', None)
        channels = data.pop('channel', None)
        labels = data.get('labels')
        regression_labels = data.pop('regression_labels', None)
        # BT-SFT Huber target (opt-in via huber_target='score_percentile'). Pop it here so it does
        # NOT reach model(**data) as an unexpected kwarg; forward it to the loss like regression_labels.
        score_percentile = data.pop('score_percentile', None)
        cg_ids = data.pop('cg_ids', None)
        # Reasoning BT-SFT: `num_cg_labels` stays in `data` so it reaches model.forward as a
        # kwarg (Qwen3VLAIPReasoningModel's 1/K reasoning CE weight). Drop it for model classes
        # whose forward doesn't accept it.
        if 'num_cg_labels' in data and not any(
                getattr(m, '_reasoning_ce_1_over_k', False) for m in self.unwrapped_models):
            data.pop('num_cg_labels')
        if self.args.task_type == 'seq_cls':
            data.pop('labels', None)
        if is_gui_mode():
            output_tensor = model(**data)
        else:
            with nvtx_range(nvtx_label('model_forward')):
                output_tensor = model(**data)
        packed_seq_params = data.get('packed_seq_params')
        attention_mask = data.get('attention_mask')
        if self.args.task_type == 'seq_cls':
            loss_func = partial(
                self.seq_cls_loss_func,
                labels=labels,
                packed_seq_params=packed_seq_params,
                attention_mask=attention_mask)
        elif isinstance(output_tensor, tuple):
            lm_output, reg_logits = output_tensor
            loss_func = partial(
                self.aip_loss_func,
                lm_labels=labels,
                reg_logits=reg_logits,
                regression_labels=regression_labels,
                score_percentile=score_percentile,
                cg_ids=cg_ids,
                attention_mask=attention_mask,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params)
            output_tensor = lm_output
        else:
            loss_func = partial(
                self.loss_func,
                labels=labels,
                loss_scale=loss_scale,
                channels=channels,
                packed_seq_params=packed_seq_params,
                attention_mask=attention_mask)
        return output_tensor, loss_func
