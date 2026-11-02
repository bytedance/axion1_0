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

"""Build the HLE question files, including the 130-item text-only subset.

Source: https://huggingface.co/datasets/cais/hle

The repo is gated. Accept the terms at the dataset page, create a token at
https://huggingface.co/settings/tokens, then `export HF_TOKEN=hf_...` or
`huggingface-cli login`.

    pip install huggingface_hub pandas pyarrow
    python prepare.py --out-dir .

With the parquet already downloaded:

    python prepare.py --parquet test-00000-of-00001.parquet --out-dir .

Writes `hle_full.jsonl` (every item), `hle_text_only.jsonl` (items with no
image) and `hle_text_only_130.jsonl` (the ids in `subset_130_ids.txt`, which
holds ids only, no questions and no answers).

Only `question`, `id` and `answer` are written. The source `rationale` field
holds a worked solution and is dropped. Images are not written: the text-only
split exists because an item whose image carries part of the question cannot be
answered from text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _strip_socks_proxy() -> None:
    """Drop ALL_PROXY, which breaks httpx when it is a socks URL and socksio is absent."""
    for v in ("ALL_PROXY", "all_proxy"):
        if os.environ.pop(v, None):
            print(f"[prepare] cleared {v} (socks proxy; httpx needs socksio for it)")


def fetch_parquet(out_dir: str = HERE) -> str:
    """Download the test parquet and return its local path."""
    from huggingface_hub import hf_hub_download
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        print("[prepare] warning: no HF_TOKEN set; a gated repo will return 401",
              file=sys.stderr)
    return hf_hub_download(
        repo_id="cais/hle", repo_type="dataset",
        filename="data/test-00000-of-00001.parquet",
        local_dir=os.path.join(out_dir, "_hle_raw"),
        token=tok or None,
    )


def has_image(row: dict) -> bool:
    """True when the row carries an image in `image` or `image_preview`."""
    for k in ("image", "image_preview"):
        v = row.get(k)
        if isinstance(v, (bytes, bytearray)) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, dict) and v:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", default=None, help="local parquet; omit to download")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--subset-ids", default=os.path.join(HERE, "subset_130_ids.txt"))
    args = ap.parse_args()

    _strip_socks_proxy()
    os.makedirs(args.out_dir, exist_ok=True)
    path = args.parquet or fetch_parquet(args.out_dir)
    print(f"[prepare] parquet = {path}")

    import pandas as pd
    df = pd.read_parquet(path)
    print(f"[prepare] {len(df)} rows, columns: {list(df.columns)}")

    full, text_only = [], []
    for _, r in df.iterrows():
        row = r.to_dict()
        rec = {
            "question": row.get("question") or "",
            "id": row.get("id") or "",
            "answer": row.get("answer") or "",
        }
        if not rec["question"] or not rec["id"]:
            continue
        full.append(rec)
        if not has_image(row):
            text_only.append(rec)

    if len(full) < 2000:
        print(f"[prepare] ERROR: only {len(full)} items; the conversion looks incomplete",
              file=sys.stderr)
        return 1

    def write(rows, name):
        out = os.path.join(args.out_dir, name)
        with open(out, "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[prepare] {len(rows):5d} items -> {out}")

    write(full, "hle_full.jsonl")
    write(text_only, "hle_text_only.jsonl")
    print(f"[prepare] {len(full) - len(text_only)} items carry an image and are "
          f"excluded from the text-only split")

    with open(args.subset_ids, encoding="utf-8") as f:
        want = [line.strip() for line in f if line.strip()]
    by_id = {rec["id"]: rec for rec in text_only}
    subset = [by_id[i] for i in want if i in by_id]
    missing = [i for i in want if i not in by_id]
    if missing:
        print(f"[prepare] ERROR: {len(missing)} of the {len(want)} subset ids are not in "
              f"the text-only split, e.g. {missing[:3]}", file=sys.stderr)
        return 1
    write(subset, "hle_text_only_130.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
