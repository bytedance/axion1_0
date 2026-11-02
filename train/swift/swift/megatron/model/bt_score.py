# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared constants/helpers for BT-SFT verbalized scoring.

BT-SFT trains a *regular* Qwen3VL LM (no regression head) to rank samples by a
0..N-1 safety score that is read directly off the LM head.  Each score k is
relabeled to a dedicated single special token ``<score_k>`` so the score sits at
exactly one position and one vocab id (no multi-token "10" ambiguity).  At that
position we take the N score-token logits, softmax them, and use the *expected
value* ``s = Σ k·softmax(logits)_k`` as a differentiable continuous score that
feeds the existing Bradley-Terry loss (``regression_loss_type='bradley_terry'``).

The number of buckets N is configurable (``--num_score_buckets``); the helpers
here build the token list for a given N so the relabeling tool
(``tools/relabel_score_single_token.py``) and the model forward stay in sync ---
the vocab ids used to slice the logits always match the ids the data was labeled
with.  ``NUM_SCORE_CLASSES`` / ``SCORE_SPECIAL_TOKENS`` remain as the default
(N=11) for backward compatibility.
"""
from typing import List, Optional

NUM_SCORE_CLASSES = 11  # default number of score buckets (scores 0..10 inclusive)


def score_special_tokens(num_classes: int = NUM_SCORE_CLASSES) -> List[str]:
    """Return the ordered ``<score_0>``..``<score_{num_classes-1}>`` token list."""
    if num_classes < 2:
        raise ValueError(f'num_classes must be >= 2, got {num_classes}')
    return [f'<score_{k}>' for k in range(num_classes)]


SCORE_SPECIAL_TOKENS: List[str] = score_special_tokens(NUM_SCORE_CLASSES)


def score_to_token(score: int, num_classes: int = NUM_SCORE_CLASSES) -> str:
    """Map an integer score 0..num_classes-1 to its ``<score_k>`` special token."""
    if not 0 <= score < num_classes:
        raise ValueError(f'score out of range [0,{num_classes - 1}]: {score}')
    return f'<score_{score}>'


def get_score_token_ids(tokenizer, num_classes: int = NUM_SCORE_CLASSES) -> List[int]:
    """Return the num_classes vocab ids for ``<score_0>``..``<score_{num_classes-1}>``.

    Raises if any token is not a single known id (i.e. the special tokens were
    not registered via ``--new_special_tokens``), which would silently break the
    logit slicing.
    """
    tokens = score_special_tokens(num_classes)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unk = getattr(tokenizer, 'unk_token_id', None)
    bad = [t for t, i in zip(tokens, ids) if i is None or i == unk]
    if bad:
        raise ValueError(
            f'Score special tokens not registered in the tokenizer: {bad}. '
            f'Pass --new_special_tokens "{" ".join(tokens)}".')
    return list(ids)


def per_cg_score_token(cg: int, score: int, num_classes: int = NUM_SCORE_CLASSES) -> str:
    """Per-CG score token ``<cg_{cg}_score_{score}>`` (each CG owns its own N score tokens)."""
    if cg < 0:
        raise ValueError(f'cg must be >= 0, got {cg}')
    if not 0 <= score < num_classes:
        raise ValueError(f'score out of range [0,{num_classes - 1}]: {score}')
    return f'<cg_{cg}_score_{score}>'


def per_cg_score_tokens(num_cg: int, num_classes: int = NUM_SCORE_CLASSES) -> List[str]:
    """Flat ordered list of the per-CG score tokens: for each cg 0..C-1, its
    ``<cg_c_score_0>``..``<cg_c_score_{N-1}>`` (``num_cg * num_classes`` tokens, cg-major)."""
    if num_cg < 1:
        raise ValueError(f'num_cg must be >= 1, got {num_cg}')
    if num_classes < 2:
        raise ValueError(f'num_classes must be >= 2, got {num_classes}')
    return [f'<cg_{c}_score_{k}>' for c in range(num_cg) for k in range(num_classes)]


def get_per_cg_score_token_ids(tokenizer, num_cg: int, num_classes: int = NUM_SCORE_CLASSES) -> List[List[int]]:
    """Return the per-CG score-token vocab ids as a ``[num_cg][num_classes]`` nested list.

    Raises if any token is not a single known id (i.e. the per-CG score tokens were not
    registered via ``--new_special_tokens``), which would silently break the logit slicing.
    """
    tokens = per_cg_score_tokens(num_cg, num_classes)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unk = getattr(tokenizer, 'unk_token_id', None)
    bad = [t for t, i in zip(tokens, ids) if i is None or i == unk]
    if bad:
        raise ValueError(
            f'Per-CG score special tokens not registered in the tokenizer: {bad[:6]}{"..." if len(bad) > 6 else ""}. '
            f'Pass --new_special_tokens with the per-CG score tokens.')
    return [list(ids[c * num_classes:(c + 1) * num_classes]) for c in range(num_cg)]


SCORE_UNK_TOKEN = '<score_unk>'  # placeholder for unlabeled CG slots (CE-masked in training)


def cg_marker_tokens(num_cg: int) -> List[str]:
    """Return the ordered ``<cg_0>``..``<cg_{num_cg-1}>`` category-marker tokens.

    In multi-CG BT-SFT the assistant emits ``<cg_c><score_k>`` per category; the
    marker tokens delimit which category each score belongs to (fixed order).
    """
    if num_cg < 1:
        raise ValueError(f'num_cg must be >= 1, got {num_cg}')
    return [f'<cg_{c}>' for c in range(num_cg)]


def get_cg_marker_ids(tokenizer, num_cg: int) -> List[int]:
    """Return the num_cg vocab ids for ``<cg_0>``..``<cg_{num_cg-1}>`` (must be registered)."""
    tokens = cg_marker_tokens(num_cg)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    unk = getattr(tokenizer, 'unk_token_id', None)
    bad = [t for t, i in zip(tokens, ids) if i is None or i == unk]
    if bad:
        raise ValueError(
            f'CG marker tokens not registered in the tokenizer: {bad}. '
            f'Pass --new_special_tokens "{" ".join(tokens)}".')
    return list(ids)


def get_score_unk_id(tokenizer) -> int:
    """Return the vocab id for the ``<score_unk>`` placeholder (must be registered)."""
    i = tokenizer.convert_tokens_to_ids(SCORE_UNK_TOKEN)
    unk = getattr(tokenizer, 'unk_token_id', None)
    if i is None or i == unk:
        raise ValueError(
            f'{SCORE_UNK_TOKEN} not registered in the tokenizer. '
            f'Pass --new_special_tokens "{SCORE_UNK_TOKEN}".')
    return int(i)


def parse_multicg_scores(text: str, num_cg: int) -> List[Optional[int]]:
    """Parse a generated ``<cg_0><score_k0>...<cg_{C-1}><score_k{C-1}>`` block into
    a list of bucket ints per category (None if a category is missing/malformed)."""
    import re
    out: List[Optional[int]] = [None] * num_cg
    for c in range(num_cg):
        m = re.search(rf'<cg_{c}>\s*<score_(\d+)>', text)
        if m:
            out[c] = int(m.group(1))
    return out


def expected_score_from_logits(score_logits, temperature: float = 1.0):
    """E[score] = Σ_{k=0}^{N-1} k · softmax(score_logits / T)_k.

    score_logits: [..., N] (logits already sliced to the score-token ids, in
    ascending score order).  Returns [...] --- a differentiable score in [0, N-1].
    """
    import torch
    probs = torch.softmax(score_logits.float() / temperature, dim=-1)
    k = torch.arange(score_logits.shape[-1], device=score_logits.device, dtype=probs.dtype)
    return (probs * k).sum(dim=-1)


def multicg_special_tokens(num_cg: int, num_classes: int = NUM_SCORE_CLASSES,
                           per_cg_score_tokens_mode: bool = False) -> List[str]:
    """Full ordered ``--new_special_tokens`` list for multi-CG BT-SFT:

      markers <cg_0>..<cg_{C-1}>  +  score tokens  +  <score_unk>

    where the score tokens are the C*N per-CG tokens (``per_cg_score_tokens_mode=True``)
    or the N shared ``<score_k>`` tokens (default). Single source of truth so the data
    builder, model, export and inference all register/read the same vocab.
    """
    markers = cg_marker_tokens(num_cg)
    scores = (per_cg_score_tokens(num_cg, num_classes) if per_cg_score_tokens_mode
              else score_special_tokens(num_classes))
    return markers + scores + [SCORE_UNK_TOKEN]
