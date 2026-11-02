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

"""Rollout loading and answer selection shared by the judges.

A rollout directory holds one file per sampling round, `iter1.jsonl` ...
`iterN.jsonl`, one JSON object per line with the keys `filename`, `question`,
`prediction`, `messages` and `termination`.

Two answer sources are available. `prediction` is the content of the final
`<answer></answer>` tag; `last_assistant` is the whole last assistant turn.
Benchmarks that compare a final answer use the first, benchmarks that grade a
derivation against a rubric or marking scheme use the second.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

#: Harness-appended instructions, stripped from a question before grading.
APPENDED_INSTRUCTIONS = [
    "\n\nPut your reasoning content inside <think></think>. When you have gathered "
    "sufficient information and are ready to provide the definitive response, you must "
    "enclose the entire final answer within <answer></answer> tags.",
]


def strip_appended(question: str) -> str:
    """Remove trailing harness instructions from a question."""
    q = question or ""
    changed = True
    while changed:
        changed = False
        for hint in APPENDED_INSTRUCTIONS:
            if q.endswith(hint):
                q = q[: -len(hint)]
                changed = True
    return q.strip()


def last_assistant(row: dict) -> str:
    """The whole last assistant turn, or '' if there is none."""
    msgs = row.get("messages") or row.get("full_messages") or []
    for m in reversed(msgs):
        if m.get("role") == "assistant" and str(m.get("content") or "").strip():
            return str(m["content"])
    return ""


def answer_text(row: dict, source: str = "prediction") -> str:
    """Text to grade, falling back to the other source when this one is empty."""
    primary = (row.get("prediction") or "") if source == "prediction" else last_assistant(row)
    primary = primary.strip()
    if primary:
        return primary
    other = last_assistant(row) if source == "prediction" else (row.get("prediction") or "")
    return other.strip()


#: Placeholders some harnesses write in place of an answer.
NO_ANSWER_SENTINELS = {"", "No answer found.", "[Failed]", "None"}


def is_no_answer(text: str) -> bool:
    return (text or "").strip() in NO_ANSWER_SENTINELS


def load_dataset(path: str, key: str = "filename") -> dict:
    """Read a prepared dataset jsonl into {key: row}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row[key]] = row
    return out


def load_rollouts(
    input_dir: str,
    rollouts: int,
    dataset: dict,
    answer_source: str = "prediction",
    question_fixup: Optional[Callable[[str], str]] = strip_appended,
) -> list:
    """Collect gradeable tasks from iter1..iterN.jsonl.

    Returns dicts with `_key` ("<round>::<filename>"), `round`, `filename`,
    `pred`, `question` and `termination`. Rows whose `filename` is absent from
    the dataset are dropped with a warning.
    """
    indir = Path(input_dir)
    tasks: list = []
    unknown: dict = {}
    for k in range(1, rollouts + 1):
        f = indir / f"iter{k}.jsonl"
        if not f.exists():
            print(f"[warn] missing {f}, skipping round {k}")
            continue
        seen = set()
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fn = str(row.get("filename") or "").strip()
                if not fn or fn not in dataset:
                    unknown[fn or "<no filename>"] = unknown.get(fn or "<no filename>", 0) + 1
                    continue
                if fn in seen:          # keep the first of any duplicate
                    continue
                seen.add(fn)
                q = row.get("question") or dataset[fn].get("question") or ""
                tasks.append({
                    "_key": f"{k}::{fn}",
                    "round": k,
                    "filename": fn,
                    "pred": answer_text(row, answer_source),
                    "question": question_fixup(q) if question_fixup else q,
                    "termination": row.get("termination") or "",
                })
        print(f"  iter{k}: {len(seen)} items")
    if unknown:
        total = sum(unknown.values())
        print(f"[warn] {total} rows had a filename absent from the dataset: "
              f"{list(unknown)[:3]}")
    return tasks


def check_count(tasks: Iterable[dict], expected: int, rollouts: int) -> None:
    """Warn when a round holds fewer than `expected` items."""
    per_round: dict = {}
    for t in tasks:
        per_round[t["round"]] = per_round.get(t["round"], 0) + 1
    for k in range(1, rollouts + 1):
        n = per_round.get(k, 0)
        if n == 0:
            print(f"[warn] round {k}: no items")
        elif n != expected:
            print(f"[warn] round {k}: {n} items, expected {expected} "
                  f"({expected - n} missing -- the score for this round is over "
                  f"a smaller denominator)")
