"""Paper GRPO batches: normalize trajectories, then train their TITO sessions.

Numerical helpers deliberately need only NumPy/PyTorch. DataProto and Ray are
imported at the adapter boundary so this contract can be checked on a CPU.
"""
from collections import Counter
import math
from numbers import Integral

import numpy as np
import torch


def validation_dataset_required(config):
    """An empty validation source is valid only when validation is disabled."""
    has_source = bool(config.data.get('val_files'))
    requested = config.trainer.get('val_before_train', True) or config.trainer.get('test_freq', -1) > 0
    if requested and not has_source:
        raise ValueError('Requested validation requires nonempty data.val_files')
    return has_source


def normalize_trajectory_rewards(rewards):
    """Use verl GRPO's unbiased standard deviation before session expansion."""
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError('GRPO needs at least two finite trajectory rewards')
    return (rewards - rewards.mean()) / (rewards.std() + 1e-6)


def prompt_group_rejection(rewards, uids, oldest_versions, newest_versions, current_version):
    """Return a rejection reason; malformed identity/version data is an error.

    Version age is measured from the oldest weights used anywhere in the
    trajectory. This conservatively bounds age when generation spans syncs.
    A stale member discards its entire group, preserving eight-rollout GRPO.
    """
    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 1 or len(rewards) != 8 or len(uids) != 8:
        raise ValueError('A prompt group must contain exactly eight trajectories')
    if len(set(uids)) != 1:
        raise ValueError('A prompt group must share one uid')
    if len(oldest_versions) != 8 or len(newest_versions) != 8:
        raise ValueError('Missing trajectory parameter version metadata')
    for oldest, newest in zip(oldest_versions, newest_versions, strict=True):
        if (not isinstance(oldest, Integral) or isinstance(oldest, bool)
                or not isinstance(newest, Integral) or isinstance(newest, bool)
                or not 0 <= oldest <= newest <= current_version):
            raise ValueError(f'Invalid trajectory parameter version: {oldest!r}, {newest!r}, current={current_version}')
    if any(current_version - int(v) > 11 for v in oldest_versions):
        return 'stale'
    if not torch.isfinite(rewards).all():
        return 'invalid_reward'
    if bool(torch.all(rewards == rewards[0])):
        return 'degenerate'
    return None


async def collect_prompt_groups(get_sample, classify, required=24):
    """Read past rejected groups; never turn an incomplete tail into an update."""
    accepted = []
    rejected = Counter()
    while len(accepted) < required:
        item = await get_sample()
        if item is None or item[0] is None:
            return None, dict(rejected)
        sample, _queue_size = item
        reason = classify(sample)
        if reason is None:
            accepted.append(sample)
        else:
            rejected[reason] += 1
    return accepted, dict(rejected)


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def expand_session_rows(session_outputs, rewards, advantages, non_tensors, *,
                        prompt_length, response_length, pad_token_id, require_routes=True):
    """Build session rows without rendering, decoding, or retokenizing text."""
    if not (len(session_outputs) == len(rewards) == len(advantages)):
        raise ValueError('Trajectory/session metadata length mismatch')
    if pad_token_id is None:
        raise ValueError('A tokenizer pad token is required')
    rows = []
    parent_indices = []
    session_ids = []
    trajectory_ids = []
    route_shape = None
    occurrences = Counter()
    for index, sessions in enumerate(session_outputs):
        if not sessions:
            raise ValueError('Every trajectory must expose its TITO session_outputs')
        uid = str(non_tensors['uid'][index])
        trajectory_id = f'{uid}:{occurrences[uid]}'
        occurrences[uid] += 1
        for session in sessions:
            prompt = torch.as_tensor(session['prompt_ids'], dtype=torch.long)
            response = torch.as_tensor(session['response_ids'], dtype=torch.long)
            mask = torch.as_tensor(session['response_mask'], dtype=torch.long)
            if prompt.ndim != 1 or not 0 < len(prompt) <= prompt_length:
                raise ValueError('Session prompt exceeds its budget or is empty')
            if response.ndim != 1 or not 0 < len(response) <= response_length:
                raise ValueError('Session response exceeds its budget or is empty')
            if mask.shape != response.shape or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError('Session policy mask must align with response tokens and be binary')
            active = torch.where(mask.bool())[0]
            if not len(active):
                raise ValueError('Session contains no policy-generated tokens')
            logprobs = session.get('response_logprobs')
            if logprobs is None:
                raise ValueError('Session rollout log probabilities are required')
            logprobs = torch.as_tensor(logprobs, dtype=torch.float32)
            if logprobs.shape != response.shape or not torch.isfinite(logprobs).all():
                raise ValueError('Session rollout log probabilities are missing or misaligned')
            padded_prompt = torch.full((prompt_length,), int(pad_token_id), dtype=torch.long)
            padded_response = torch.full((response_length,), int(pad_token_id), dtype=torch.long)
            padded_prompt[-len(prompt):] = prompt
            padded_response[:len(response)] = response
            policy_mask = torch.zeros(response_length, dtype=torch.long)
            policy_mask[:len(response)] = mask
            attention = torch.zeros(prompt_length + response_length, dtype=torch.long)
            offset = prompt_length - len(prompt)
            attention[offset:prompt_length + len(response)] = 1
            score = torch.zeros(response_length, dtype=torch.float32)
            score[int(active[-1])] = rewards[index]
            padded_logprobs = torch.zeros(response_length, dtype=torch.float32)
            padded_logprobs[:len(response)] = logprobs
            row = dict(prompts=padded_prompt, responses=padded_response,
                       response_mask=policy_mask, input_ids=torch.cat([padded_prompt, padded_response]),
                       attention_mask=attention, position_ids=(attention.cumsum(-1) - 1).clamp_min(0),
                       rollout_log_probs=padded_logprobs, rm_scores=score,
                       trajectory_advantage=torch.as_tensor(advantages[index], dtype=torch.float32))
            if require_routes or session.get('routed_experts') is not None:
                routes = session.get('routed_experts')
                coverage = session.get('routed_experts_mask')
                if routes is None or coverage is None:
                    raise ValueError('Session routing records and coverage mask are required')
                routes = torch.as_tensor(routes)
                coverage = torch.as_tensor(coverage, dtype=torch.bool)
                sequence_length = len(prompt) + len(response)
                # Predicting response index j uses input position len(prompt)+j-1.
                required_prefix = len(prompt) + int(active[-1])
                if (routes.ndim != 3 or routes.shape[0] > sequence_length
                        or coverage.shape != (sequence_length,)
                        or not coverage[:required_prefix].all()
                        or not coverage[:routes.shape[0]].all()
                        or coverage[routes.shape[0]:].any()):
                    raise ValueError('Session routing coverage does not cover causal policy predecessors')
                if routes.is_floating_point() or routes.dtype == torch.bool or (routes < 0).any() or (routes > 255).any():
                    raise ValueError('Session routing expert IDs must be integers in [0,255]')
                if route_shape is None:
                    route_shape = tuple(routes.shape[1:])
                if tuple(routes.shape[1:]) != route_shape:
                    raise ValueError('Session routing layer/top-k dimensions disagree')
                padded_routes = torch.zeros((prompt_length + response_length, *route_shape), dtype=torch.uint8)
                padded_routes[offset:offset + len(routes)] = routes.to(torch.uint8)
                padded_coverage = torch.zeros(prompt_length + response_length, dtype=torch.bool)
                padded_coverage[offset:offset + sequence_length] = coverage
                row['routed_experts'] = padded_routes
                row['routed_experts_mask'] = padded_coverage
            rows.append(row)
            parent_indices.append(index)
            session_ids.append(session['session_id'])
            trajectory_ids.append(trajectory_id)
    if not rows:
        raise ValueError('No trainable sessions')
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError('Routing metadata must be present consistently across sessions')
    tensors = {key: torch.stack([row[key] for row in rows]) for key in keys}
    # Large session payloads are no longer needed after expansion. Keep parent
    # diagnostic fields, but never duplicate each trajectory's entire session list.
    fields = {key: _object_array([values[i] for i in parent_indices])
              for key, values in non_tensors.items() if key != 'session_outputs'}
    fields.update(trajectory_id=_object_array(trajectory_ids), session_id=_object_array(session_ids),
                  is_padding=np.zeros(len(rows), dtype=bool))
    return tensors, fields, len(rows)


def pad_session_rows(tensors, fields, divisor):
    """Pad dispatch rows with loss-masked copies; keep their token buffers valid."""
    if divisor < 1:
        raise ValueError('Session padding divisor must be positive')
    size = len(tensors['input_ids'])
    extra = (-size) % divisor
    if not extra:
        return tensors, fields
    shortest = int(tensors['attention_mask'].sum(-1).argmin())
    indices = torch.tensor(list(range(size)) + [shortest] * extra)
    tensors = {key: value[indices].clone() for key, value in tensors.items()}
    fields = {key: value[indices.numpy()].copy() for key, value in fields.items()}
    for key in ('response_mask', 'rm_scores', 'trajectory_advantage'):
        tensors[key][size:] = 0
    fields['is_padding'][size:] = True
    return tensors, fields


def actor_batch_parameters(padded_sessions, real_sessions):
    """One optimizer step, normalized by real sessions rather than dispatch padding."""
    if not 0 < real_sessions <= padded_sessions:
        raise ValueError('Invalid real/padded session counts')
    return dict(mini_batch_size=padded_sessions, global_batch_size=real_sessions, epochs=1)


class AxionAsyncBatchAdapter:
    """Narrow integration with the existing fully asynchronous verl trainer."""

    def __init__(self, config):
        self.config = config
        actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
        checks = {
            'eight rollouts': rollout.n == 8,
            '24 prompt groups': actor.ppo_mini_batch_size == 24 and config.async_training.require_batches == 1,
            'GRPO': config.algorithm.adv_estimator == 'grpo',
            'normalized GRPO advantages': config.algorithm.get('norm_adv_by_std_in_grpo', True),
            'one PPO epoch': actor.ppo_epochs == 1,
            'session loss': actor.loss_agg_mode == 'seq-mean-token-mean',
            'CP1': actor.megatron.context_parallel_size == 1,
            'sync interval 2': config.async_training.trigger_parameter_sync_step == 2,
            'staleness 11': config.async_training.staleness_threshold == 11,
            'no reward KL': not config.algorithm.use_kl_in_reward,
            'rollout log probability bypass': config.algorithm.rollout_correction.bypass_mode,
            'fresh-start training': config.trainer.resume_mode == 'disable',
            'no prefix grouping': not actor.get('use_prefix_grouper', False),
            'no critic warmup': config.trainer.critic_warmup == 0,
        }
        transformer = actor.megatron.get('override_transformer_config', {})
        checks['no MoE auxiliary loss'] = transformer.get('moe_aux_loss_coeff', 0) == 0
        checks['no MoE z loss'] = transformer.get('moe_z_loss_coeff', 0) == 0
        target = config.trainer.get('total_training_steps')
        checks['explicit positive even update target'] = isinstance(target, Integral) and target > 0 and target % 2 == 0
        for label, valid in checks.items():
            if not valid:
                raise ValueError(f'Axion paper recipe requires {label}')
        self.target_updates = int(target)
        self.rejected = Counter()

    def classify(self, sample, current_version):
        batch = sample.full_batch
        flags = batch.non_tensor_batch.get('drop_from_training', [])
        if any(bool(flag) for flag in flags):
            return 'reward_failure'
        if 'rm_scores' not in batch.batch:
            raise ValueError('Axion prompt filtering requires rollout-computed final rewards')
        rewards = batch.batch['rm_scores'].sum(-1)
        reason = prompt_group_rejection(rewards, batch.non_tensor_batch['uid'],
            batch.non_tensor_batch['min_global_steps'], batch.non_tensor_batch['max_global_steps'], current_version)
        if reason is None:
            batch.batch['trajectory_advantage'] = normalize_trajectory_rewards(rewards)
        return reason

    def expand(self, batch, trainer):
        from verl import DataProto
        rollout = self.config.actor_rollout_ref.rollout
        tensors, fields, real_sessions = expand_session_rows(
            batch.non_tensor_batch['session_outputs'], batch.batch['rm_scores'].sum(-1),
            batch.batch['trajectory_advantage'], batch.non_tensor_batch,
            prompt_length=rollout.prompt_length, response_length=rollout.response_length,
            pad_token_id=trainer.tokenizer.pad_token_id, require_routes=True,
        )
        dp_size = trainer._get_dp_size(trainer.actor_rollout_wg, 'actor')
        # Reference-policy inference can have its own DP topology.
        ref_dp_size = trainer._get_dp_size(trainer.ref_policy_wg, 'ref') if trainer.use_reference_policy else 1
        actor = self.config.actor_rollout_ref.actor
        divisor = math.lcm(dp_size, ref_dp_size)
        if not actor.use_dynamic_bsz:
            divisor = math.lcm(divisor, dp_size * actor.ppo_micro_batch_size_per_gpu)
        ref_config = self.config.actor_rollout_ref.get('ref', {})
        if trainer.use_reference_policy and not ref_config.get('log_prob_use_dynamic_bsz', actor.use_dynamic_bsz):
            ref_microbatch = ref_config.get('log_prob_micro_batch_size_per_gpu')
            if not isinstance(ref_microbatch, Integral) or ref_microbatch <= 0:
                raise ValueError('Static reference inference needs a positive per-GPU microbatch size')
            divisor = math.lcm(divisor, ref_dp_size * ref_microbatch)
        tensors, fields = pad_session_rows(tensors, fields, divisor)
        meta = dict(batch.meta_info)
        meta.update(axion_real_sessions=real_sessions, axion_prompt_groups=24,
                    fully_async_session_rows=real_sessions)
        result = DataProto.from_dict(tensors=tensors, non_tensors=fields, meta_info=meta)
        if self.config.trainer.balance_batch:
            trainer._balance_batch(result, metrics=trainer.metrics)
        result.meta_info['global_token_num'] = result.batch['attention_mask'].sum(-1).tolist()
        return result

    def compute_advantage(self, trainer, batch):
        # The formal recipe uses rollout log probabilities directly. Its PPO
        # policy-loss helper handles any configured rollout correction at update.
        batch.batch['token_level_scores'] = trainer.reward_tensor
        batch.batch['token_level_rewards'] = trainer.reward_tensor
        for key, values in trainer.reward_extra_infos_dict.items():
            batch.non_tensor_batch[key] = _object_array(values)
        advantages = batch.batch['trajectory_advantage'].unsqueeze(-1) * batch.batch['response_mask']
        batch.batch['advantages'] = advantages
        batch.batch['returns'] = advantages.clone()
        return batch
