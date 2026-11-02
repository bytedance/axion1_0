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

"""Build the SEAL-0 question file.

Source: https://huggingface.co/datasets/vtllms/sealqa, config `seal_0`

    pip install datasets
    python prepare.py --current-year 2026 --out-dir .

Each item carries the year it was written. The prompt states that year and the
current one and asks for the answer as of the former; the wrapper is
`QUESTION_TEMPLATE` below. `--current-year` goes into the prompt, so pass the
year inference runs in.

Reference answers change over time. The output file name and its
`snapshot_prepared` field record when it was built; grade rollouts against the
snapshot that was current when they ran.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

#: Framing wrapper: `{raw}` the published question, `{created}` the year it was
#: written, `{current}` the year inference runs in.
QUESTION_TEMPLATE = """{raw}

Please note:
1. The current year is {current}.
2. This question was created on {created}.
3. You must answer based on the facts that were correct at the time the question was created, not the current date.

If the question contains incorrect facts or false assumptions, do not follow them. Instead, identify the mistake and provide the correct answer based on real-world facts."""

EXPECTED_COUNT = 111


def rows_from_hub() -> list:
    """Download the seal_0 config from the Hub."""
    from datasets import load_dataset
    ds = load_dataset("vtllms/sealqa", "seal_0")
    split = "test" if "test" in ds else list(ds.keys())[0]
    return [dict(r) for r in ds[split]]


def rows_from_file(path: str) -> list:
    """Read items from a local jsonl."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick(row: dict, *names, default=""):
    """First non-empty value among `names`, else `default`."""
    for n in names:
        if row.get(n) not in (None, ""):
            return row[n]
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="", help="local jsonl; omit to download from the Hub")
    ap.add_argument("--current-year", default="",
                    help="year stated in the prompt; defaults to the current year")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-name", default="")
    args = ap.parse_args()

    import datetime
    current = args.current_year or str(datetime.date.today().year)
    today = datetime.date.today().isoformat()

    rows = rows_from_file(args.src) if args.src else rows_from_hub()

    items = []
    for i, r in enumerate(rows):
        raw = str(pick(r, "question", "query"))
        created = str(pick(r, "effective_year", "created_year", "year", default="unknown"))
        items.append({
            "question": QUESTION_TEMPLATE.format(raw=raw, current=current, created=created),
            "raw_question": raw,
            "answer": pick(r, "answer", "gold_answer", "target"),
            "filename": f"seal0_{i:04d}",
            "id": str(pick(r, "id", default=i)),
            "effective_year": created,
            "freshness": pick(r, "freshness"),
            "topic": pick(r, "topic"),
            "snapshot_prepared": today,
            "current_year_in_prompt": current,
        })

    assert len({it["filename"] for it in items}) == len(items), "filename is not unique"
    if len(items) != EXPECTED_COUNT:
        print(f"[warn] {len(items)} items, expected {EXPECTED_COUNT}", file=sys.stderr)

    name = args.out_name or f"seal0_{today.replace('-', '')}.jsonl"
    out = os.path.join(args.out_dir, name)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[ok] {len(items)} items -> {out}")
    print(f"     prompt says the current year is {current}; snapshot taken {today}")
    for key in ("freshness", "effective_year"):
        c = collections.Counter(it[key] for it in items)
        print(f"     {key}: " + "  ".join(f"{k or '?'}={v}" for k, v in sorted(
            c.items(), key=lambda x: str(x[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
