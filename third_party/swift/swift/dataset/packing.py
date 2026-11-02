# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import multiprocessing as mp
import torch.distributed as dist
from itertools import chain
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from typing import Optional

from swift.template import MaxLengthError
from swift.utils import get_logger, is_dist, is_master, split_list

logger = get_logger()


def _pack_dual_constraint(sequences, max_tokens: int, max_visual_tokens: int, is_finished: bool = True):
    bins = []
    for item in sequences:
        _, tok, vis = item
        placed = False
        for b in bins:
            if b[1] + tok <= max_tokens and b[2] + vis <= max_visual_tokens:
                b[0].append(item)
                b[1] += tok
                b[2] += vis
                placed = True
                break
        if not placed:
            bins.append([[item], tok, vis])
    result = [b[0] for b in bins]
    if result and not is_finished:
        return result[:-1], result[-1]
    return result, []


def calculate_matched_group(sequences, packing_length: int, is_finished: bool = True):
    if len(sequences) == 0:
        return [], []
    # https://arxiv.org/pdf/2404.10830
    import binpacking
    sequences = binpacking.to_constant_volume(sequences, packing_length, weight_pos=1)
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        ret_sequences = []
    return sequences, ret_sequences


class PackingDataset(Dataset):
    PACKING_BATCH_SIZE = 1000

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        packing_length: Optional[int] = None,
        packing_num_proc: int = 1,
        max_visual_tokens: Optional[int] = None,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.load_from_cache_file = load_from_cache_file
        self.packing_length = packing_length or self.template.max_length
        self.max_visual_tokens = max_visual_tokens
        self.packing_num_proc = min(packing_num_proc, math.ceil(len(dataset) / self.PACKING_BATCH_SIZE))
        self._out_queue = mp.Queue()
        if is_master():
            lengths = self._read_column('lengths', 'length', 'total_tokens')
            visual_lengths = None
            use_visual = self.max_visual_tokens is not None
            if use_visual:
                visual_lengths = self._read_column('visual_tokens')
                use_visual = visual_lengths is not None
            offset = 0
            chunked_lengths = split_list(lengths, self.packing_num_proc)
            chunked_visual = split_list(visual_lengths, self.packing_num_proc) if use_visual else [None] * self.packing_num_proc
            for i in range(self.packing_num_proc):
                worker = mp.Process(
                    target=self.create_packed_idx, args=(
                        i,
                        offset,
                        chunked_lengths[i],
                        chunked_visual[i],
                        use_visual,
                    ), daemon=True)
                worker.start()
                offset += len(chunked_lengths[i])
            self.packed_idx = [[] for _ in range(self.packing_num_proc)]
            self.packed_length = [[] for _ in range(self.packing_num_proc)]
            desc = 'Packing: ' if self.packing_num_proc == 1 else f'Packing (num_proc={self.packing_num_proc}): '
            with tqdm(total=len(lengths), dynamic_ncols=True, desc=desc) as prog_bar:
                finished_workers = 0
                while finished_workers < self.packing_num_proc:
                    rank, sequences, data_len = self._out_queue.get()
                    if data_len == -1:
                        finished_workers += 1
                        continue
                    prog_bar.update(data_len)
                    self.packed_idx[rank] += [[x[0] for x in seq] for seq in sequences]
                    self.packed_length[rank] += [sum(x[1] for x in seq) for seq in sequences]
            self.packed_idx = list(chain.from_iterable(self.packed_idx))
            self.packed_length = list(chain.from_iterable(self.packed_length))
        else:
            self.packed_idx, self.packed_length = None, None
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]

    def _read_column(self, *names):
        for name in names:
            try:
                col = self.dataset[name]
                if col is not None:
                    return col
            except (KeyError, TypeError, AttributeError, ValueError):
                pass
        return None

    def create_packed_idx(self, rank, offset, lengths, visual_lengths=None, use_visual=False):
        if use_visual and visual_lengths is not None:
            data = [(i + offset, sum(length) if isinstance(length, list) else length,
                     visual_lengths[i]) for i, length in enumerate(lengths)]
        else:
            data = [(i + offset, sum(length) if isinstance(length, list) else length) for i, length in enumerate(lengths)]
        i = 0
        input_data = []
        while True:
            new_data = data[i:i + self.PACKING_BATCH_SIZE]
            input_data += new_data
            if not input_data:
                break
            i += self.PACKING_BATCH_SIZE
            is_finished = i >= len(data)
            if use_visual:
                sequences, input_data = _pack_dual_constraint(
                    input_data, self.packing_length, self.max_visual_tokens, is_finished=is_finished)
            else:
                sequences, input_data = calculate_matched_group(input_data, self.packing_length, is_finished=is_finished)
            self._out_queue.put((rank, sequences, len(new_data)))
        self._out_queue.put((rank, [], -1))

    def __getitem__(self, index):
        sequence = self.packed_idx[index]
        row = [self.dataset[i] for i in sequence]
        return row

    def __len__(self):
        return len(self.packed_idx)


class IterablePackingDataset(IterableDataset):

    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        packing_length: Optional[int] = None,
        strict: bool = False,
        cyclic: bool = False,
        **kwargs,
    ):
        template.packing = True
        template.padding_free = True  # TODO: remove
        self.template = template
        self.dataset = dataset
        self.num_proc = num_proc
        self.strict = strict
        self.packing_length = packing_length or self.template.max_length

        self.packing_interval = packing_interval
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        self.workers = []
        self.cyclic = cyclic
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _processor(self):
        while True:
            i, data = self._in_queue.get()
            encoded_data = {}
            try:
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            self._out_queue.put((i, encoded_data))

    def _put_data_in_queue(self, iterator) -> int:
        for i in range(self.packing_interval):
            try:
                data = next(iterator)
            except StopIteration:
                return i
            self._in_queue.put((i, data))
        return i + 1

    def _fetch_data_out_queue(self, last_res, num_samples):
        res = [None] * num_samples
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            if not data:
                continue
            res[i] = (data, len(data['input_ids']))
        res = [data for data in res if data]
        last_res += res
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        while True:
            for x in iterable:
                yield x

    def __iter__(self):
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        data = []
        while True:
            num_samples = self._put_data_in_queue(iterator)
            finished = num_samples != self.packing_interval
            data = self._fetch_data_out_queue(data, num_samples)
            sequences, data = calculate_matched_group(data, self.packing_length, is_finished=finished)
            res = []
            for row in sequences:
                res.append([r[0] for r in row])
            yield from res
            if finished:
                break
