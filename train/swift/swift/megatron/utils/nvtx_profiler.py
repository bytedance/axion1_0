# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional

import torch

# GUI-friendly labels: short, sortable, distinct colors in Nsight Systems.
_GUI_LABELS = {
    'training_step': '[STEP] train_step',
    'zero_grad': '[ZERO] zero_grad',
    'forward_backward': '[FB] forward_backward',
    'optimizer_step': '[OPT] optimizer',
    'prefetch_batch': '[DATA] prefetch',
    'get_batch': '[DATA] get_batch',
    'dataloader_fetch': '[DATA] dataloader',
    'batch_prepare': '[DATA] batch_prepare',
    'forward_step': '[FWD] forward',
    'backward_step': '[BWD] backward',
    'loss_compute': '[FWD] loss',
    'finalize_grads': '[FB] finalize_grads',
    'model_forward': '[FWD] model_forward',
    'ep_dispatch': '[EP] dispatch',
    'ep_combine': '[EP] combine',
    'pp_p2p': '[PP] p2p',
}


def ep_layer_label(layer_number: Optional[int], phase: str) -> str:
    """Per-layer EP label for 3-level Nsight GUI: ``[EP] L03 dispatch``."""
    ln = 0 if layer_number is None else int(layer_number)
    return f'[EP] L{ln:02d} {phase}'


def make_ep_phase_name(phase: str) -> Callable:
    """Build a dynamic NVTX name callback for ``MoELayer`` methods."""

    def _name(self, *_args, **_kwargs):
        return ep_layer_label(getattr(self, 'layer_number', None), phase)

    return _name


def layer_phase_label(prefix: str, layer_number: Optional[int], phase: str = '') -> str:
    """Per-layer non-MoE label for 3-level Nsight GUI: ``[ATTN] L03`` or ``[NORM] L03 pre_mlp``."""
    ln = 0 if layer_number is None else int(layer_number)
    if phase:
        return f'[{prefix}] L{ln:02d} {phase}'
    return f'[{prefix}] L{ln:02d}'


def make_layer_phase_name(prefix: str, phase: str = '') -> Callable:
    """Build a dynamic NVTX name callback for ``TransformerLayer`` / ``Attention`` methods."""

    def _name(self, *_args, **_kwargs):
        return layer_phase_label(prefix, getattr(self, 'layer_number', None), phase)

    return _name


def is_nvtx_enabled() -> bool:
    return os.environ.get('MS_SWIFT_NSYS_NVTX', '0') == '1'


def is_gui_mode() -> bool:
    return is_nvtx_enabled() and os.environ.get('MS_SWIFT_NSYS_NVTX_GUI', '0') == '1'


def nvtx_label(key: str, *, suffix: str = '') -> str:
    if is_gui_mode():
        base = _GUI_LABELS.get(key, key)
        return f'{base}{suffix}' if suffix else base
    return f'ms_swift/{key}{suffix}' if suffix else f'ms_swift/{key}'


def nvtx_push(name: str) -> None:
    if is_nvtx_enabled():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    if is_nvtx_enabled():
        torch.cuda.nvtx.range_pop()


def nvtx_mark(name: str) -> None:
    if is_nvtx_enabled():
        torch.cuda.nvtx.mark(name)


@contextmanager
def nvtx_range(name: str):
    if not is_nvtx_enabled():
        yield
        return
    nvtx_push(name)
    try:
        yield
    finally:
        nvtx_pop()


def wrap_nvtx(orig: Callable, name: str) -> Callable:
    @wraps(orig)
    def wrapper(*args, **kwargs):
        if not is_nvtx_enabled():
            return orig(*args, **kwargs)
        nvtx_push(name)
        try:
            return orig(*args, **kwargs)
        finally:
            nvtx_pop()

    return wrapper


def wrap_module_forward_nvtx(module, label: str) -> None:
    """Wrap ``module.forward`` with a fixed NVTX range label."""
    if module is None or not hasattr(module, 'forward'):
        return
    orig_forward = module.forward

    @wraps(orig_forward)
    def forward(*args, **kwargs):
        if not is_nvtx_enabled():
            return orig_forward(*args, **kwargs)
        nvtx_push(label)
        try:
            return orig_forward(*args, **kwargs)
        finally:
            nvtx_pop()

    module.forward = forward


def patch_method_nvtx(cls, method_name: str, nvtx_name: str, *, dynamic_name: Optional[Callable] = None) -> None:
    """Patch ``cls.method_name`` with an NVTX range (no-op when NVTX is disabled)."""
    orig = getattr(cls, method_name)

    if dynamic_name is not None:

        @wraps(orig)
        def wrapper(*args, **kwargs):
            if not is_nvtx_enabled():
                return orig(*args, **kwargs)
            label = dynamic_name(*args, **kwargs)
            nvtx_push(label)
            try:
                return orig(*args, **kwargs)
            finally:
                nvtx_pop()

        setattr(cls, method_name, wrapper)
        return

    setattr(cls, method_name, wrap_nvtx(orig, nvtx_name))
