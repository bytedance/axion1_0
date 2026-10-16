#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Build the FS-O and FS-R question files.

Source: https://huggingface.co/datasets/openai/frontierscience
        split `olympiad` -> FS-O (100 items)
        split `research` -> FS-R (60 items)

    pip install datasets
    python prepare.py --out-dir .

With a local copy, expecting `<src>/olympiad/test.jsonl` and
`<src>/research/test.jsonl`:

    python prepare.py --src /path/to/frontierscience --out-dir .

`problem` is copied through unchanged, including the benchmark's own
answer-format instruction. `filename` is the item key used by the rollout loop
and the judge and is made unique per item; `task_group_id` is not unique in the
official research split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SPLITS = {"olympiad": ("fs_o", 100), "research": ("fs_r", 60)}


def rows_from_hub(split: str) -> list:
    """Download one split from the Hub."""
    from datasets import load_dataset
    ds = load_dataset("openai/frontierscience", split=split)
    return [dict(r) for r in ds]


def rows_from_dir(src: str, split: str) -> list:
    """Read one split from `<src>/<split>/test.jsonl`."""
    path = os.path.join(src, split, "test.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="", help="local copy; omit to download from the Hub")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    for split, (name, expected) in SPLITS.items():
        rows = rows_from_dir(args.src, split) if args.src else rows_from_hub(split)

        seen: dict = {}
        items = []
        for i, r in enumerate(rows):
            gid = r["task_group_id"]
            seen[gid] = seen.get(gid, 0) + 1
            uid = gid if seen[gid] == 1 else f"{gid}_{seen[gid]}"
            items.append({
                "question": r["problem"],
                "answer": r["answer"],          # FS-O: final answer; FS-R: the rubric
                "filename": f"{name}_{i:04d}_{uid}",
                "task_id": uid,
                "subject": r.get("subject", ""),
                "task_group_id": gid,
            })

        assert len({it["filename"] for it in items}) == len(items), "filename is not unique"
        assert len(items) == expected, f"{split}: expected {expected} items, got {len(items)}"

        ndup = len(items) - len({it["question"].strip() for it in items})
        if ndup:
            print(f"[note] {split}: {ndup} item(s) duplicate another item's text "
                  f"(present in the official data); unique filenames keep them apart")

        out = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[ok] {split}: {len(items)} items -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
