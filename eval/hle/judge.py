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

"""Grade HLE: is the model's final answer equivalent to the reference answer?

    export OPENAI_API_KEY=...
    export OPENAI_MODEL=...
    python judge.py --dataset hle_text_only_130.jsonl --input-dir rollouts/ --rollouts 3

Reports per-round accuracy, the mean, and Best@k, the fraction of questions
answered correctly in at least one of the first k rounds.

The judge prompt is the one that ships with the agent harness, reproduced
verbatim, not the official HLE grader (`hle_eval/run_judge_results.py` in
github.com/centerforaisafety/hle). They differ in four ways:

| | official | here |
|--|--|--|
| answer extraction | the judge model, from the whole response | a regex, from the last `<answer>...</answer>` |
| response with no final answer | graded anyway | scored wrong without a judge call |
| judge input | question + whole response + reference | question + extracted answer + reference |
| metric | accuracy + standard error + calibration error | accuracy + Best@k |

State the grader when placing these scores beside a published HLE number.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))
sys.path.insert(0, str(HERE.parent.parent / "datagen" / "utils"))

import rollouts as R  # noqa: E402
from llm_client import LLMClient, run_batch  # noqa: E402

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

#: Reproduced verbatim from the agent harness.
JUDGE_PROMPT = """Please determine whether the predicted answer matches the correct answer.

Question: {question}

Correct answer: {ground_truth}

Predicted answer:
{predicted_answer}

Compare the predicted answer and the correct answer carefully and decide whether they are consistent.

If the predicted answer explicitly contains the correct answer, or its core meaning matches the correct answer, respond "true".
If the predicted answer does not match or contradicts the correct answer, respond "false".
For multiple-choice questions, if the predicted answer mentions the correct option (e.g., "A", "B"), treat it as consistent.
For numerical answers, if the predicted answer contains the number, treat it as consistent.
Reply with only "true" or "false". Do not add any other content."""


def extract_answer_tag(row: dict) -> str | None:
    """Content of the last <answer></answer> tag in the final message, else None."""
    msgs = row.get("messages") or []
    if not msgs:
        return None
    matches = _ANSWER_TAG_RE.findall(str(msgs[-1].get("content") or ""))
    return matches[-1].strip() if matches else None


def parse_verdict(text: str):
    """"true" / "false" -> True / False, None when unparseable.

    None is distinct from False: an unparseable verdict is a judge failure, not
    a wrong answer.
    """
    if not text:
        return None
    t = text.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    if "true" in t and "false" not in t:
        return True
    if "false" in t and "true" not in t:
        return False
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="jsonl with question / id / answer")
    ap.add_argument("--input-dir", required=True, help="directory holding iter1.jsonl ...")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--qpm", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="judge sampling temperature; some deployments reject 0")
    ap.add_argument("--judge-max-tokens", type=int, default=8192,
                    help="token budget per judge call")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # HLE items are keyed by `id`, not `filename`.
    dataset = R.load_dataset(args.dataset, key="id")
    print(f"[judge] reference answers: {len(dataset)}")

    indir = Path(args.input_dir)
    tasks = []
    for k in range(1, args.rollouts + 1):
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
                qid = str(row.get("id") or "").strip()
                if not qid or qid not in dataset or qid in seen:
                    continue
                seen.add(qid)
                tasks.append({
                    "_key": f"{k}::{qid}",
                    "round": k,
                    "id": qid,
                    "answer_tag": extract_answer_tag(row),
                })
        print(f"  iter{k}: {len(seen)} items")
    if not tasks:
        print("[error] nothing to grade")
        return 1
    R.check_count(tasks, len(dataset), args.rollouts)

    client = LLMClient(qpm=args.qpm)
    print(f"[judge] model={client.model} temperature={args.temperature} "
          f"max_tokens={args.judge_max_tokens} workers={args.workers} qpm={args.qpm}")

    def work(t):
        gt = dataset[t["id"]]
        pred = t["answer_tag"]
        if pred is None or not pred.strip():
            # No answer tag: scored wrong without a judge call.
            return {"round": t["round"], "id": t["id"], "is_correct": False,
                    "judge_failed": False, "no_answer_tag": True, "verdict": ""}
        reference = str(gt.get("answer") or "")
        if not reference.strip():
            return {"round": t["round"], "id": t["id"], "is_correct": False,
                    "judge_failed": False, "no_reference": True, "verdict": ""}
        raw = client.chat(
            [{"role": "user", "content": JUDGE_PROMPT.format(
                question=gt.get("question", ""), ground_truth=reference,
                predicted_answer=pred)}],
            validator=lambda t2: parse_verdict(t2) is not None,
            temperature=args.temperature, max_tokens=args.judge_max_tokens,
        )
        v = parse_verdict(raw)
        return {"round": t["round"], "id": t["id"],
                "is_correct": bool(v),
                "judge_failed": v is None,
                "verdict": (raw or "")[:200],
                "predicted_answer": pred[:1000]}

    ckpt = indir / "judge_hle_checkpoint.jsonl"
    impl = "harness-prompt-v1"
    done, failed = run_batch(tasks, lambda t: t["_key"], work, str(ckpt),
                             workers=args.workers, passes=4, impl=impl)
    if failed:
        print(f"[warn] {len(failed)} items failed every pass and are excluded")

    rows = list(done.values())
    by_round = collections.defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)

    solved: dict = {}
    best_at_k = {}
    n_questions = len({r["id"] for r in rows})
    for k in sorted(by_round):
        for r in by_round[k]:
            solved[r["id"]] = solved.get(r["id"], False) or r["is_correct"]
        n_ok = sum(1 for v in solved.values() if v)
        best_at_k[f"best@{k}"] = {
            "correct": n_ok, "total": n_questions,
            "accuracy_pct": round(100.0 * n_ok / n_questions, 2) if n_questions else 0.0,
        }

    per_round = []
    rounds = {}
    n_judge_failed = sum(1 for r in rows if r.get("judge_failed"))
    for k in sorted(by_round):
        rs = by_round[k]
        acc = 100.0 * sum(1 for r in rs if r["is_correct"]) / len(rs)
        rounds[k] = {
            "n": len(rs),
            "correct": sum(1 for r in rs if r["is_correct"]),
            "accuracy_pct": round(acc, 2),
            "no_answer_tag": sum(1 for r in rs if r.get("no_answer_tag")),
            "judge_failed": sum(1 for r in rs if r.get("judge_failed")),
        }
        per_round.append(acc)

    report = {
        "bench": "hle",
        "dataset": args.dataset,
        "input_dir": str(indir),
        "grader": "agent-harness prompt, not the official HLE grader "
                  "(see the module docstring for the four differences)",
        "judge_model": client.model,
        "judge_params": {"temperature": args.temperature,
                         "max_tokens": args.judge_max_tokens,
                         "workers": args.workers, "qpm": args.qpm,
                         "rollouts": args.rollouts},
        "impl": impl,
        "n_reference": len(dataset),
        "n_judge_failed": n_judge_failed,
        "rounds": rounds,
        "best_at_k": best_at_k,
        "summary": {
            "per_round_pct": [round(x, 2) for x in per_round],
            "mean_pct": round(statistics.mean(per_round), 2) if per_round else 0.0,
            "std_pct": round(statistics.pstdev(per_round), 2) if len(per_round) > 1 else 0.0,
        },
    }
    if n_judge_failed:
        report["judge_failed_warning"] = (
            f"{n_judge_failed} items could not be graded (the judge kept returning "
            f"nothing or erroring) and are counted wrong, so every accuracy above is "
            f"a lower bound. Re-judge those items."
        )

    out = Path(args.out) if args.out else indir / "judge_hle_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"HLE  per round: {report['summary']['per_round_pct']}")
    print(f"HLE  mean {report['summary']['mean_pct']}%  "
          f"(std {report['summary']['std_pct']})")
    for k, v in best_at_k.items():
        print(f"  {k}: {v['accuracy_pct']}%  ({v['correct']}/{v['total']})")
    n_untagged = sum(v["no_answer_tag"] for v in rounds.values())
    if n_untagged:
        print(f"  {n_untagged} response(s) had no <answer> tag and were scored wrong "
              f"without a judge call")
    if n_judge_failed:
        print(f"  WARNING: {n_judge_failed} item(s) could not be graded; "
              f"the scores above are a lower bound")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
