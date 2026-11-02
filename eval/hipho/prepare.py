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

"""Build the HiPhO question files.

Source: https://huggingface.co/datasets/SciYu/HiPhO (MIT)

    pip install huggingface_hub
    python prepare.py --out-dir .

From a local clone of the dataset repo:

    python prepare.py --src /path/to/HiPhO/data --out-dir .

Writes `hipho.jsonl` (13 exams, 360 problems, the version the paper reports)
and `hipho_to.jsonl` (the 102 text-only problems of those 360). `CPhO_2025` is
excluded: it was added to the Hub after publication and is not part of the 360.
Both counts are asserted.

Record 0 of each exam file is that exam's constants table rather than a
problem, and is prepended to the prompt when it has content. Grading fields
(`marking`, `points`, `answer`, `unit`, `answer_type`) are carried through
verbatim. Image paths are written absolute, so prepare on the machine that will
run inference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

EXCLUDE = {"CPhO_2025"}
EXPECT_PROBLEMS = 360
EXPECT_EXAMS = 13
EXPECT_TEXT_ONLY = 102

INSTRUCTION = (
    "\n\nSolve the problem above. Show your full derivation: intermediate steps are graded "
    "against the official marking scheme, so state the reasoning and formulas you use, not "
    "just the result. Put each final result in \\boxed{}."
)

#: Values of `information` that mean the exam has no constants table.
_INFO_ABSENT = {"", "none", "none.", "n/a", "na", "null"}


def build_question(info: str, ctx: str, q: str) -> str:
    """Assemble one prompt from the exam data sheet, shared context and question."""
    parts = []
    if info and str(info).strip().lower() not in _INFO_ABSENT:
        parts.append("[Data Sheet]\n" + str(info).strip())
    if ctx and ctx.strip():
        parts.append("[Problem]\n" + ctx.strip())
    parts.append("[Question]\n" + q.strip())
    return "\n\n".join(parts) + INSTRUCTION


def fetch_data_dir() -> str:
    """Download the dataset and return its data/ directory."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id="SciYu/HiPhO", repo_type="dataset")
    return os.path.join(root, "data")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="", help="local HiPhO data/ dir; omit to download")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    src = args.src or fetch_data_dir()
    files = sorted(f for f in os.listdir(src)
                   if f.endswith(".json") and os.path.splitext(f)[0] not in EXCLUDE)
    assert len(files) == EXPECT_EXAMS, \
        f"expected {EXPECT_EXAMS} exams, got {len(files)}: {files}"

    items = []
    for fn in files:
        exam = os.path.splitext(fn)[0]
        with open(os.path.join(src, fn), encoding="utf-8") as f:
            recs = json.load(f)
        info_recs = [r for r in recs if "information" in r]
        assert len(info_recs) <= 1, f"{fn} has more than one information record"
        info = info_recs[0]["information"] if info_recs else ""

        for r in recs:
            if "information" in r:
                continue
            images = []
            for rel in r.get("images") or []:
                p = rel if os.path.isabs(rel) else os.path.join(src, rel)
                assert os.path.exists(p), f"missing image: {rel} (in {r['id']})"
                images.append(os.path.abspath(p))
            items.append({
                "question": build_question(info, r.get("context", ""), r["question"]),
                "answer": r.get("answer") or [],
                "answer_type": r.get("answer_type") or ["Open-End"],
                "unit": r.get("unit") or [],
                "points": r.get("points") or [],
                "marking": r.get("marking") or [],
                "images": images,
                "filename": f"{exam}__{r['id']}",
                "task_id": str(r["id"]),
                "modality": r.get("modality", ""),
                "field": r.get("field", ""),
                "source": r.get("source", exam),
            })

    assert len(items) == EXPECT_PROBLEMS, \
        f"expected {EXPECT_PROBLEMS} problems, got {len(items)}"
    assert len({it["filename"] for it in items}) == len(items), "filename is not unique"
    for it in items:
        assert len(it["points"]) == len(it["answer"]), \
            f"{it['filename']}: points and answer have different lengths"

    out = os.path.join(args.out_dir, "hipho.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[ok] {len(items)} problems from {len(files)} exams -> {out}")

    sub = [it for it in items if it["modality"] == "text-only"]
    assert len(sub) == EXPECT_TEXT_ONLY, \
        f"expected {EXPECT_TEXT_ONLY} text-only problems, got {len(sub)}"
    assert all(not it["images"] for it in sub), "a text-only problem carries an image"
    subout = os.path.join(args.out_dir, "hipho_to.jsonl")
    with open(subout, "w", encoding="utf-8") as f:
        for it in sub:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_marking = sum(1 for it in sub if it["marking"])
    print(f"[ok] text-only subset: {len(sub)} problems from "
          f"{len({it['source'] for it in sub})} exams -> {subout}")
    print(f"     of those, {n_marking} have a marking scheme and "
          f"{len(sub) - n_marking} do not (graded at answer level only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
