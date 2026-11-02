#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared prompts for Stage I, and the two calls that use them.

A Stage I item is produced by three prompts in sequence:

    GEN     (gen_from_seed.py) turn a seed problem into a harder new question
    SOLVE   write out the complete derivation and a final answer
    AUDIT   re-read that derivation, without being shown the answer, and say
            whether it holds

The audit decides whether an answer becomes the reference answer. It is shown
the reasoning only, never the answer, so it judges the work rather than agreeing
with a number. SOLVE therefore has to produce the full derivation: that text is
what the audit reads.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.
"""
import os
import sys

if __package__:
    from ..utils.llm_client import LLMClient
else:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
    from llm_client import LLMClient

_client = None


def client(model=None, qpm=60):
    """Lazily built shared client, so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = LLMClient(model=model, qpm=qpm)
    return _client


JUDGE = """Please determine whether the predicted answer matches the correct answer.

Question: {question}

Correct answer: {gt}

Predicted answer:
{pred}

Compare the predicted answer and the correct answer carefully and decide whether they are consistent.

If the predicted answer explicitly contains the correct answer, or its core meaning matches the correct answer, respond "true".
If the predicted answer does not match or contradicts the correct answer, respond "false".
For multiple-choice questions, if the predicted answer mentions the correct option (e.g., "A", "B"), treat it as consistent.
For numerical answers, if the predicted answer contains the number, treat it as consistent.
Reply with only "true" or "false". Do not add any other content."""


def judge(question, gt, pred, model=None):
    """Ask the model whether `pred` matches the reference answer `gt`."""
    text = client(model).chat(
        [{'role': 'user', 'content': JUDGE.format(question=question, gt=gt, pred=pred)}],
        max_tokens=8192, temperature=1.0, tag='judge')
    return to_bool(text)


def to_bool(s):
    """Parse a true/false verdict, returning None when the reply is neither."""
    s = (s or '').strip().lower()
    if s.startswith('true'):
        return True
    if s.startswith('false'):
        return False
    return None


DERIV_CLAUSE = """Write the COMPLETE derivation as you work: state what you are trying, carry out every
algebraic step explicitly, check special cases, and say when an approach fails and why you
switch. Do not compress or summarize -- write the reasoning out in full, the way a careful
person thinks on paper. This full text is the deliverable, not just the answer."""

SOLVE_BASE = """Solve this problem. If it is ill-posed, self-contradictory, under-specified, or has a
non-unique answer, say so instead of guessing.

""" + DERIV_CLAUSE + """

End with EXACTLY:
FINAL: <the final answer only>
or if broken:
FINAL: ILL_POSED - <one line reason>

PROBLEM:
{q}"""

# Variant used by gen_from_seed.py: also reports the seed quantity, so the
# derivation can be checked for actually passing through the seed answer.
SOLVE_WITH_SEEDQ = """Solve this problem. If it is ill-posed, self-contradictory or
under-specified, say so instead of guessing.

""" + DERIV_CLAUSE + """

While solving, you must also report the value you obtain for this specific quantity:
  >>> {sq}

End with EXACTLY:
SEED_Q_VALUE: <the value you obtained for the quantity above, or NOT_NEEDED if your
derivation never requires it>
FINAL: <the final answer only>
or if broken:
FINAL: ILL_POSED - <one line reason>

PROBLEM:
{q}"""

AUDIT = """You are auditing a solution to a problem. You are NOT given the correct answer --
judge purely from the reasoning itself.

Check the derivation step by step. Look for: algebra slips, unjustified jumps, a misread of
the problem statement, a case that was silently dropped, an arithmetic error, or a final
answer that does not follow from the work shown. Also say if the PROBLEM itself is ill-posed,
self-contradictory or under-specified.

End with EXACTLY these two lines:
FIRST_BAD_STEP: <short quote of the first step that is wrong, or NONE>
VERDICT: <CORRECT if the final answer is right, INCORRECT if wrong, ILL_POSED if the problem
is broken, UNSURE if you genuinely cannot tell>

PROBLEM:
{q}

SOLUTION TO AUDIT:
{sol}"""


def clip_for_audit(text, head=5000, tail=9000, cap=14000):
    """Clip a derivation for the audit, keeping both ends.

    The opening states which approach was chosen and the closing carries the
    decisive steps, so a middle-out clip preserves what the audit needs.
    """
    t = text or ''
    return t if len(t) <= cap else t[:head] + '\n\n...[middle omitted]...\n\n' + t[-tail:]