# Copyright 2026 The Axion Authors
# SPDX-License-Identifier: Apache-2.0
"""Token-aligned routing records shared by rollout clients and agent loops."""

import numbers

import torch


def merge_routed_experts(previous, current, prompt_length, response_length, *, required=False, max_experts=None):
    """Retain already sampled routes and append newly covered causal rows.

    A completion of R tokens from P prompt tokens records P+R-1 input rows:
    the last sampled token has not itself been forwarded yet. On resume or a
    new tool turn, the returned prompt includes the previous context. Replacing
    that prefix would overwrite routes sampled under earlier parameter versions.
    """
    if not response_length:
        return previous
    if current is None:
        if required:
            raise ValueError('R3 completion is missing routed_experts')
        return previous
    # Ray may return read-only numpy buffers. Own the CPU tensor before retaining it.
    routes = torch.tensor(current) if not isinstance(current, torch.Tensor) else current.detach().cpu()
    if routes.ndim != 3 or not all(routes.shape[1:]):
        raise ValueError('routed_experts must have shape [tokens, layers, topk]')
    if routes.dtype == torch.bool or routes.is_floating_point() or routes.is_complex():
        raise TypeError('routed_experts must contain integer expert IDs')
    expected = prompt_length + response_length - 1
    if routes.shape[0] != expected:
        raise ValueError(f'routed_experts has {routes.shape[0]} rows; expected {expected}')
    lower, upper = int(routes.min()), int(routes.max())
    if lower < 0 or (max_experts is not None and upper >= max_experts):
        raise ValueError(f'routed_experts IDs outside expert range [0, {max_experts})')
    routes = routes.to(torch.uint8 if upper <= 255 else torch.int64)
    if previous is None:
        return routes
    if previous.shape[1:] != routes.shape[1:] or previous.shape[0] > expected:
        raise ValueError('routed_experts prefix shape changed across generation requests')
    return torch.cat((previous, routes[previous.shape[0]:]), dim=0)


def session_routing(experts, prompt_length, response_ids, response_mask, response_length, *, required=False):
    """Clip a session and identify observed routes, excluding unused tool suffixes."""
    response_ids = response_ids[:response_length]
    response_mask = response_mask[:response_length]
    if len(response_ids) != len(response_mask):
        raise ValueError('response token/mask length mismatch')
    total = prompt_length + len(response_ids)
    policy_positions = [i for i, value in enumerate(response_mask) if value]
    needed = prompt_length + policy_positions[-1] if policy_positions else 0
    routes = experts[:total] if experts is not None else None
    coverage = torch.zeros(total, dtype=torch.bool)
    if routes is not None:
        coverage[:len(routes)] = True
    if required and needed and (routes is None or len(routes) < needed):
        raise ValueError(f'R3 session lacks causal route coverage for {needed} required rows')
    return routes, coverage


def update_version_bounds(bounds, output_fields, *, required=False):
    """Aggregate the oldest/newest contributing rollout parameter versions."""
    low = output_fields.get('min_global_steps', output_fields.get('global_steps'))
    high = output_fields.get('max_global_steps', output_fields.get('global_steps'))
    if low is None or high is None:
        if required:
            raise ValueError('Rollout output is missing parameter version bounds')
        return
    if not isinstance(low, numbers.Integral) or not isinstance(high, numbers.Integral) or low < 0 or high < low:
        raise ValueError('Invalid rollout parameter version bounds')
    bounds['min_global_steps'] = min(bounds.get('min_global_steps', int(low)), int(low))
    bounds['max_global_steps'] = max(bounds.get('max_global_steps', int(high)), int(high))


def bshd_router_rows(routes, replay_mask, scatter):
    """Match Megatron's [sequence, batch] hidden-state/SP router ordering."""
    rows = scatter(routes.transpose(0, 1).contiguous()).flatten(0, 1)
    mask = None
    if replay_mask is not None:
        mask = scatter(replay_mask.transpose(0, 1).contiguous()).flatten(0, 1)
    return rows, mask
