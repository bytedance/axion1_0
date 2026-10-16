# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""CPU-only check: run each dataset through the LlamaFactory pipeline and
assert that nothing is dropped.

The expected count is the number of lines in the jsonl, so a conversation lost
to an invalid role tag, an over-long sample or a piece that does not end on an
assistant turn shows up as a mismatch. Those are dropped with a warning that is
easy to miss in the training log, which is why this runs as its own step.

Check every dataset, not a sample of the easiest one: a dataset where every row
holds a single conversation passes trivially, while one with
multi-conversation rows loses them silently.

    python check_data.py --model <path> --data-dir ./data \
        --datasets sft_objective sft_openended
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="base model path, for the tokenizer and template")
ap.add_argument("--data-dir", default="./data", help="directory holding dataset_info.json")
ap.add_argument("--datasets", nargs="+", required=True, help="dataset names from dataset_info.json")
ap.add_argument("--template", default="qwen3_5")
ap.add_argument("--cutoff-len", type=int, default=32768)
a = ap.parse_args()

# Imported after argument parsing on purpose, so --help works without the
# training stack installed.
import numpy as np  # noqa: E402
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_tokenizer  # noqa: E402

info = json.load(open(os.path.join(a.data_dir, "dataset_info.json")))

def line_count(name):
    fn = info[name]["file_name"]
    path = os.path.join(a.data_dir, fn)
    if os.path.isdir(path):
        return sum(sum(1 for _ in open(os.path.join(path, f), encoding="utf-8"))
                   for f in sorted(os.listdir(path)))
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)

fail = False
for name in a.datasets:
    expect = line_count(name)
    args = dict(stage="sft", do_train=True, model_name_or_path=a.model,
                dataset=name, dataset_dir=a.data_dir, template=a.template,
                cutoff_len=a.cutoff_len, output_dir="./tmp_check", overwrite_cache=True,
                preprocessing_num_workers=8, finetuning_type="full")
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
    tm = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tm["tokenizer"], data_args)
    d = get_dataset(template, model_args, data_args, training_args, stage="sft", **tm)["train_dataset"]
    got = len(d)
    ok = got == expect
    fail |= not ok
    print(f"\n{'ok' if ok else 'FAIL'} {name}: expected {expect:,} got {got:,}  dropped {expect - got:,}")

    ids = np.array(d[0]["input_ids"]); lab = np.array(d[0]["labels"])
    txt = tm["tokenizer"].decode(ids)
    m = (lab != -100).astype(int)
    blocks = int(((m[1:] - m[:-1]) == 1).sum() + m[0])
    print(f"   sample 0: {len(ids)} tokens, {int(m.sum())} in the loss ({m.mean() * 100:.1f}%), "
          f"{txt.count('<think>')} <think>, {txt.count('<|im_start|>assistant')} assistant turns, "
          f"{blocks} supervised spans")
    print(f"   system messages: {txt.count('<|im_start|>system')} "
          f"(must be 1; more means a row was not split)")

print("\n" + ("FAIL: rows were dropped, do not start training" if fail else "ok: nothing dropped"))
sys.exit(1 if fail else 0)
