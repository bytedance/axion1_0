# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import os
import random
from collections import Counter
from typing import List

import torch

from swift.utils import get_logger

logger = get_logger()


def _cg_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _cg_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _iter_dataset_chain(dataset):
    seen = set()
    candidate = dataset
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        yield candidate
        candidate = getattr(candidate, 'dataset', None)


def _read_cg_column(dataset) -> List[int]:
    for candidate in _iter_dataset_chain(dataset):
        if candidate is None:
            continue
        features = getattr(candidate, 'features', None) or getattr(candidate, 'column_names', None) or {}
        for column in ('__cg_id', 'cg_ids'):
            if column in features:
                try:
                    return [int(x) for x in candidate[column]]
                except Exception:
                    pass
        raw_data = getattr(candidate, 'data', None)
        if isinstance(raw_data, list):
            return [int(item.get('__cg_id', item.get('cg_ids', 0))) if isinstance(item, dict) else 0 for item in raw_data]
    return []


def _collect_cg_ids(dataset) -> List[int]:
    cg_ids = _read_cg_column(dataset)
    packed_idx = None
    for candidate in _iter_dataset_chain(dataset):
        packed_idx = getattr(candidate, 'packed_idx', None)
        if packed_idx is not None:
            break
    if packed_idx is not None and cg_ids:
        packed_cg_ids = []
        for sequence in packed_idx:
            first_idx = sequence[0] if sequence else 0
            packed_cg_ids.append(cg_ids[int(first_idx)] if int(first_idx) < len(cg_ids) else 0)
        logger.info(f'[_collect_cg_ids] found packed cg ids: distribution={dict(Counter(packed_cg_ids))}')
        return packed_cg_ids
    if cg_ids:
        logger.info(f'[_collect_cg_ids] found cg ids: distribution={dict(Counter(cg_ids))}')
        return cg_ids
    n = len(dataset) if hasattr(dataset, '__len__') else 0
    logger.info(f'[_collect_cg_ids] no cg id column found, falling back to all-zeros for {n} samples')
    return [0] * n


class MegatronCgRoundRobinBatchSampler:
    """Round-robin sampler over context groups for AIP multi-head regression."""

    def __init__(
        self,
        dataset,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
    ):
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size

        shuffle = _cg_env_flag('VEGETA_CG_RR_SHUFFLE', True)
        shuffle_seed = _cg_env_int('VEGETA_CG_RR_SHUFFLE_SEED', 42)

        cg_ids = _collect_cg_ids(dataset)
        explicit_num_cg = getattr(dataset, 'num_cg', None)
        self.num_cg = int(explicit_num_cg) if explicit_num_cg is not None else (max(cg_ids) + 1 if cg_ids else 1)

        self.indices_per_cg: List[List[int]] = [[] for _ in range(self.num_cg)]
        for idx, cg_id in enumerate(cg_ids):
            cg_id = max(0, min(int(cg_id), self.num_cg - 1))
            self.indices_per_cg[cg_id].append(idx)

        if shuffle:
            for cg_id, idxs in enumerate(self.indices_per_cg):
                if len(idxs) > 1:
                    rng = random.Random(shuffle_seed + cg_id)
                    rng.shuffle(idxs)

        sizes = [len(x) for x in self.indices_per_cg]
        max_size = max(sizes) if sizes else 0
        steps_per_cg = math.ceil(max_size / self.micro_batch_times_data_parallel_size) \
            if self.micro_batch_times_data_parallel_size > 0 else 0
        self.order = list(range(self.num_cg)) * steps_per_cg
        self._start_step = consumed_samples // self.micro_batch_times_data_parallel_size \
            if self.micro_batch_times_data_parallel_size > 0 else 0

        if data_parallel_rank == 0:
            logger.info(
                f'MegatronCgRoundRobinBatchSampler: num_cg={self.num_cg}, sizes_per_cg={sizes}, '
                f'global_bs={self.micro_batch_times_data_parallel_size}, steps_per_cg={steps_per_cg}, '
                f'total_steps={len(self.order)}, start_step={self._start_step}, shuffle={shuffle}')

    def __len__(self):
        return max(0, len(self.order) - self._start_step)

    def __iter__(self):
        step_used = [0] * self.num_cg
        for cg in self.order[:self._start_step]:
            step_used[cg] += 1

        step = 0
        for cg in self.order[self._start_step:]:
            idxs = self.indices_per_cg[cg]
            n = len(idxs)
            if n == 0:
                continue
            start = step_used[cg] * self.micro_batch_times_data_parallel_size + \
                self.data_parallel_rank * self.micro_batch_size
            step_used[cg] += 1
            batch = [idxs[(start + i) % n] for i in range(self.micro_batch_size)]
            if self.data_parallel_rank == 0 and (step < 10 or step % 50 == 0):
                logger.info(f'[cg_rr] step={self._start_step + step} cg={cg} indices={batch}')
            step += 1
            yield batch


# Code borrowed from megatron-lm
class MegatronPretrainingSampler:

    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


# Code borrowed from megatron-lm
class MegatronPretrainingRandomSampler:

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        shuffle: bool = True,
        group_by_length: bool = False,
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        if group_by_length:
            if data_sharding:
                data_sharding = False
                logger.warning('`group_by_length=True` is incompatible with `data_sharding=True`. '
                               'Setting `data_sharding=False` to enable length grouping.')
            if not shuffle:
                raise ValueError('shuffle must be True when group_by_length is True')
        self.data_sharding = data_sharding
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.lengths = self.dataset['lengths'] if group_by_length else None
        if self.lengths is not None:
            self.lengths = [max(length) if isinstance(length, list) else length for length in self.lengths]
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, 'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, (
            'data_parallel_rank should be smaller than data size: {}, '
            '{}'.format(self.data_parallel_rank, data_parallel_size))

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if self.shuffle:
            # data sharding and random sampling
            if self.data_sharding:
                bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
                bucket_offset = current_epoch_samples // self.data_parallel_size
                start_idx = self.data_parallel_rank * bucket_size

                g = torch.Generator()
                g.manual_seed(self.epoch)
                random_idx = torch.randperm(bucket_size, generator=g).tolist()
                idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
            else:
                full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
                full_bucket_offset = current_epoch_samples
                g = torch.Generator()
                g.manual_seed(self.epoch)
                if self.group_by_length:
                    from transformers.trainer_pt_utils import get_length_grouped_indices
                    idx_range_total = get_length_grouped_indices(
                        self.lengths, self.micro_batch_times_data_parallel_size, generator=g)
                else:
                    idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
                idx_range_active = idx_range_total[full_bucket_offset:]
                idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            idx_range = range(full_bucket_offset + self.data_parallel_rank, full_bucket_size, self.data_parallel_size)

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
