# Copyright (c) ModelScope Contributors. All rights reserved.
"""Env-gated torch.profiler that emits a Perfetto/Chrome trace over a few train steps.

Enable with ``MS_SWIFT_TORCH_PROFILER=1``. The profiler skips ``wait`` steps (compile /
cudnn autotune / dataloader warmup), profiles ``warmup`` + ``active`` steps, then writes a
Perfetto-loadable trace per selected rank and (by default) stops training. Load the
resulting ``*.pt.trace.json.gz`` in https://ui.perfetto.dev or via the ``perfetto``
``TraceProcessor`` for SQL analysis.

Env knobs (all optional):
  MS_SWIFT_TORCH_PROFILER        1 to enable (default 0)
  MS_SWIFT_TORCH_PROFILER_WAIT   steps to skip before profiling (default 8)
  MS_SWIFT_TORCH_PROFILER_WARMUP warmup steps inside the profiler (default 2)
  MS_SWIFT_TORCH_PROFILER_ACTIVE steps to actually record (default 5)
  MS_SWIFT_TORCH_PROFILER_DIR    output dir (default <output_dir>/torch_profiler)
  MS_SWIFT_TORCH_PROFILER_RANKS  'last' (default), '0', 'all', or a comma list of ranks
  MS_SWIFT_TORCH_PROFILER_SHAPES 1 to record tensor shapes (heavier trace; default 0)
  MS_SWIFT_TORCH_PROFILER_STACK  1 to record python stacks (much heavier; default 0)
  MS_SWIFT_TORCH_PROFILER_MEMORY 1 to record CUDA memory events (default 0)
  MS_SWIFT_TORCH_PROFILER_EXIT   1 to stop training after the trace is written (default 1)
"""
import os
from typing import Optional

import torch

from swift.utils import get_logger

logger = get_logger()


def torch_profiler_enabled() -> bool:
    return os.environ.get('MS_SWIFT_TORCH_PROFILER', '0') == '1'


def _cur_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _selected_ranks(spec: str, world_size: int) -> set:
    spec = (spec or 'last').strip().lower()
    if spec == 'all':
        return set(range(world_size))
    if spec == 'last':
        return {world_size - 1}
    out = set()
    for tok in spec.split(','):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out or {world_size - 1}


class MegatronTorchProfiler:
    """A thin, no-op-when-disabled wrapper over ``torch.profiler.profile``.

    Usage::

        profiler = MegatronTorchProfiler(args.output_dir)
        with profiler:
            while ...:
                run_train_step(...)
                profiler.step()
                if profiler.should_stop():
                    break
    """

    def __init__(self, output_dir: Optional[str]):
        self.enabled = torch_profiler_enabled()
        self._prof = None
        self._stop = False
        self._active_on_this_rank = False
        if not self.enabled:
            return

        self.wait = max(0, int(os.environ.get('MS_SWIFT_TORCH_PROFILER_WAIT', '8')))
        self.warmup = max(0, int(os.environ.get('MS_SWIFT_TORCH_PROFILER_WARMUP', '2')))
        self.active = max(1, int(os.environ.get('MS_SWIFT_TORCH_PROFILER_ACTIVE', '5')))
        self.exit_after = os.environ.get('MS_SWIFT_TORCH_PROFILER_EXIT', '1') == '1'
        self.record_shapes = os.environ.get('MS_SWIFT_TORCH_PROFILER_SHAPES', '0') == '1'
        self.with_stack = os.environ.get('MS_SWIFT_TORCH_PROFILER_STACK', '0') == '1'
        self.profile_memory = os.environ.get('MS_SWIFT_TORCH_PROFILER_MEMORY', '0') == '1'
        self.output_dir = os.environ.get('MS_SWIFT_TORCH_PROFILER_DIR') or os.path.join(output_dir or '.',
                                                                                        'torch_profiler')

        world_size = torch.distributed.get_world_size() if (
            torch.distributed.is_available() and torch.distributed.is_initialized()) else 1
        self.rank = _cur_rank()
        ranks = _selected_ranks(os.environ.get('MS_SWIFT_TORCH_PROFILER_RANKS', 'last'), world_size)
        self._active_on_this_rank = self.rank in ranks
        # Total steps we must observe before we can stop training.
        self._budget = self.wait + self.warmup + self.active
        self._seen = 0

    # -- context manager ---------------------------------------------------
    def __enter__(self):
        if not (self.enabled and self._active_on_this_rank):
            return self
        os.makedirs(self.output_dir, exist_ok=True)
        from torch.profiler import ProfilerActivity, profile, schedule
        sched = schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=1)
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            profile_memory=self.profile_memory,
        )
        self._prof.__enter__()
        logger.info(f'[torch_profiler] rank={self.rank} enabled: wait={self.wait} warmup={self.warmup} '
                    f'active={self.active} -> {self.output_dir}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prof is not None:
            self._prof.__exit__(exc_type, exc_val, exc_tb)
            self._prof = None
        return False

    # -- per-iteration hooks ----------------------------------------------
    def step(self):
        if not self.enabled:
            return
        self._seen += 1
        if self._prof is not None:
            self._prof.step()
        # Once every rank that profiles has consumed its budget, allow the loop to stop.
        if self._seen >= self._budget:
            self._stop = True

    def should_stop(self) -> bool:
        # Only tear training down early when the profiler asked to (and every rank agrees via
        # a fixed budget, so this stays collective-safe: all ranks stop at the same iteration).
        return self.enabled and self.exit_after and self._seen >= (self.wait + self.warmup + self.active)

    def _on_trace_ready(self, prof):
        path = os.path.join(self.output_dir, f'rank{self.rank}.pt.trace.json.gz')
        try:
            prof.export_chrome_trace(path)
            logger.info(f'[torch_profiler] rank={self.rank} wrote Perfetto trace: {path}')
        except Exception as e:  # noqa
            logger.error(f'[torch_profiler] rank={self.rank} failed to export trace: {e}')
        # A compact top-kernels summary in the log, so a bottleneck is visible without the UI.
        try:
            table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)
            logger.info(f'[torch_profiler] rank={self.rank} top CUDA kernels:\n{table}')
        except Exception as e:  # noqa
            logger.warning(f'[torch_profiler] rank={self.rank} key_averages summary failed: {e}')
