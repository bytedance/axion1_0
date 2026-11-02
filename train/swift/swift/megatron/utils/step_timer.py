# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Optional

import torch

from swift.utils import get_logger, is_last_rank

logger = get_logger()


class MegatronStepTimer:
    """Wall/GPU timing for Megatron train steps.

    Enable with ``MS_SWIFT_STEP_TIMER=1``. Logs rolling averages on the last rank
    every ``MS_SWIFT_STEP_TIMER_LOG_EVERY`` iterations (default 5).
    """

    def __init__(self):
        self.enabled = os.environ.get('MS_SWIFT_STEP_TIMER', '0') == '1'
        self.nvtx = os.environ.get('MS_SWIFT_NSYS_NVTX', '0') == '1'
        self.log_every = max(1, int(os.environ.get('MS_SWIFT_STEP_TIMER_LOG_EVERY', '5')))
        self._wall_start: Optional[float] = None
        self._phase: Dict[str, float] = {}
        self._accum = defaultdict(float)
        self._counts = defaultdict(int)
        self._step_event_start = None
        self._step_event_end = None

    def step_begin(self):
        if not self.enabled:
            return
        self._phase = {}
        self._wall_start = time.perf_counter()
        torch.cuda.synchronize()
        self._step_event_start = torch.cuda.Event(enable_timing=True)
        self._step_event_end = torch.cuda.Event(enable_timing=True)
        self._step_event_start.record()

    def add(self, name: str, ms: float):
        if self.enabled:
            self._phase[name] = self._phase.get(name, 0.0) + ms

    def step_end(self, iteration: int) -> Dict[str, float]:
        if not self.enabled or self._wall_start is None:
            return {}
        self._step_event_end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - self._wall_start) * 1000.0
        gpu_ms = self._step_event_start.elapsed_time(self._step_event_end)
        stats = dict(self._phase)
        stats['step_wall_ms'] = wall_ms
        stats['step_gpu_ms'] = gpu_ms
        # CUDA-event busy% counts stream work (incl. NCCL), not SM occupancy; see nsys for true GPU util.
        stats['cuda_stream_busy_pct'] = 100.0 * gpu_ms / wall_ms if wall_ms > 0 else 0.0
        fb_ms = stats.get('forward_backward_ms', 0.0)
        dl_ms = stats.get('dataloader_fetch_ms', 0.0) + stats.get('batch_prepare_ms', 0.0)
        opt_ms = stats.get('optimizer_step_ms', 0.0)
        if wall_ms > 0:
            stats['fb_pct'] = 100.0 * fb_ms / wall_ms
            stats['dataloader_pct'] = 100.0 * dl_ms / wall_ms
            stats['optimizer_pct'] = 100.0 * opt_ms / wall_ms
        for key, value in stats.items():
            self._accum[key] += value
            self._counts[key] += 1
        if is_last_rank() and iteration > 0 and iteration % self.log_every == 0:
            avg = {k: round(self._accum[k] / self._counts[k], 2) for k in sorted(self._accum)}
            logger.info(f'[step_timer] iter={iteration} avg_ms={avg}')
        return {k: round(v, 2) for k, v in stats.items()}

    @contextmanager
    def range(self, name: str, record_gpu: bool = True):
        from swift.megatron.utils.nvtx_profiler import is_gui_mode, nvtx_label, nvtx_pop, nvtx_push

        use_nvtx = self.nvtx
        use_timer = self.enabled
        if not use_nvtx and not use_timer:
            yield
            return
        # GUI mode: keep NVTX shallow (STEP/FB/EP/BWD). Skip get_batch / outer train_step.
        if use_nvtx and not (is_gui_mode() and name in ('training_step', 'get_batch')):
            nvtx_push(nvtx_label(name))
        wall_t0 = time.perf_counter() if use_timer else None
        ev_start = ev_end = None
        if use_timer and record_gpu:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
        try:
            yield
        finally:
            if ev_end is not None:
                ev_end.record()
                torch.cuda.synchronize()
                gpu_ms = ev_start.elapsed_time(ev_end)
                self._phase[f'{name}_gpu_ms'] = self._phase.get(f'{name}_gpu_ms', 0.0) + gpu_ms
            if wall_t0 is not None:
                wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                self._phase[f'{name}_ms'] = self._phase.get(f'{name}_ms', 0.0) + wall_ms
            if use_nvtx and not (is_gui_mode() and name in ('training_step', 'get_batch')):
                nvtx_pop()
