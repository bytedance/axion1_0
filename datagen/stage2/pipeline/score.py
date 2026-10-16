#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Deterministic answer matching against a generator's reference answer.

A reference answer produced by construction is exact, so scoring needs no
model. The item's `answer_type` and `tolerance` decide which comparison
applies:

    numeric / tolerance set   relative tolerance, with an absolute floor
                              derived from the reference answer's precision
    integer reference         exact equality
    decimal reference         relative tolerance, default 0.002
    otherwise                 normalised string equality, or containment when
                              the reference is long enough for that to be safe

The absolute floor matters: a reference written as "1.03e3" carries three
significant figures, so comparing at full float precision would reject correct
answers that were rounded the same way.
"""
import re

_SUPERSCRIPT = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')


def num(s):
    """First number in a string, or None. Tolerates commas and superscripts."""
    t = str(s).translate(_SUPERSCRIPT).replace(',', '')
    t = re.sub(r'(\d)\s*[x×]\s*10\s*\^?\s*(-?\d+)', r'\1e\2', t)
    m = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', t)
    return float(m[0]) if m else None


def norm(s):
    """Lowercase, strip trailing punctuation and markdown emphasis."""
    return re.sub(r'[\s.,;:]+$', '', str(s).strip().lower().replace('*', ''))


def _abs_floor(gt_text, tol):
    """Absolute tolerance floor implied by how precisely the reference is written.

    "1.03e3" states three significant figures, so an answer rounded the same
    way must not be rejected for differing in the digits that were never
    claimed.
    """
    g = num(gt_text)
    if g is None:
        return 0.0
    rel = abs(g) * tol
    m = re.search(r'\.(\d+)', str(gt_text))
    if m:
        return max(rel, 0.5 * 10 ** (-len(m.group(1))))
    sig = re.sub(r'[^1-9]', '', str(gt_text).split('e')[0])
    if sig:
        return max(rel, 0.5 * 10 ** (len(str(int(abs(g)))) - len(sig)))
    return rel


def correct(item, pred):
    """True when `pred` matches the item's reference answer."""
    gt = str(item['ground_truth'])
    answer_type = item.get('answer_type', '')
    tol = item.get('tolerance')

    if answer_type == 'numeric' or tol:
        g, p = num(gt), num(pred)
        return p is not None and abs(p - g) <= _abs_floor(gt, tol or 0.002)

    if re.fullmatch(r'-?\d+', gt):
        p = num(pred)
        return p is not None and p == float(gt)

    if re.fullmatch(r'-?\d+\.\d+', gt):
        g, p = float(gt), num(pred)
        return p is not None and abs(p - g) <= _abs_floor(gt, 0.002)

    return norm(pred) == norm(gt) or (len(gt) >= 3 and norm(gt) in norm(pred))


def _boxed(text):
    """Contents of every \\boxed{...}, handling nested braces."""
    out = []
    for m in re.finditer(r'\\boxed\s*\{', text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        out.append(text[m.end():i - 1])
    return out


def final_answer(text):
    """Extract a model's final answer from free-form output.

    Looks for <answer>...</answer>, then \\boxed{...}, then falls back to the
    last non-empty line.
    """
    m = re.findall(r'<answer>(.*?)</answer>', text, re.S)
    if m:
        return m[-1].strip()
    b = _boxed(text)
    if b:
        return b[-1].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ''
