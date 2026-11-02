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

"""Grade HiPhO with the official grader.

    export OPENAI_API_KEY=...
    python judge.py --dataset hipho.jsonl --input-dir rollouts/ --rollouts 3 \
        --judge-model gemini-2.5-flash

`official_impl/` holds three files copied unmodified from VLMEvalKit; this
script drives their `_evaluate_single_problem`. See `official_impl/README.md`.

Grading has two levels: a step level against the exam's marking scheme and an
answer level against the final answers. The official score per problem is the
higher of the two.

`leaderboard_points_mean` is the arithmetic mean of the exams' raw point
totals, the metric the HiPhO leaderboard uses. `score_rate` is total points
over total maximum and is a different quantity. Judge model and sampling
parameters are flags and are recorded in the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ and __package__.startswith("axion."):
    from ..utils import rollouts as R
    from ...datagen.utils.llm_client import LLMClient, run_batch
    from .official_impl import shim
else:
    sys.path.insert(0, str(HERE / "official_impl"))
    sys.path.insert(0, str(HERE.parent / "utils"))
    sys.path.insert(0, str(HERE.parent.parent / "datagen" / "utils"))

    import rollouts as R  # noqa: E402
    import shim  # noqa: E402
    from llm_client import LLMClient, run_batch  # noqa: E402

#: Instruction appended by prepare.py, stripped before grading.
APPENDED = (
    "\n\nSolve the problem above. Show your full derivation: intermediate steps are graded "
    "against the official marking scheme, so state the reasoning and formulas you use, not "
    "just the result. Put each final result in \\boxed{}."
)


def strip_appended(q: str) -> str:
    """Remove the trailing answer-format instruction from a question."""
    while q.endswith(APPENDED):
        q = q[: -len(APPENDED)]
    return q


class JudgeModel:
    """The `.generate(prompt) -> str` object the official grader calls.

    Errors propagate: the official code reads an unparsable response as zero, so
    a swallowed exception would become a silent zero. `run_batch` retries.
    """

    def __init__(self, model: str, qpm: int, max_tokens: int, temperature: float):
        self.client = LLMClient(model=model, qpm=qpm)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kw) -> str:
        return self.client.chat([{"role": "user", "content": prompt}],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens)


def to_official_row(gt: dict, prediction: str) -> dict:
    """Shape one item the way `_evaluate_single_problem` expects.

    `answer`, `answer_type`, `unit`, `marking` and `points` must be JSON
    strings, not lists: the official `_safe_parse_json_field` tests
    `pd.isna(field_value)` before any isinstance check, which raises on a list.
    """
    def js(v):
        return json.dumps(v, ensure_ascii=False)

    return {
        "prediction": prediction,
        "answer": js(gt.get("answer") or []),
        "answer_type": js(gt.get("answer_type") or ["Open-End"]),
        "unit": js(gt.get("unit") or []),
        "points": js(gt.get("points") or []),
        "marking": js(gt.get("marking") or []),
        "question": strip_appended(gt.get("question") or ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="hipho.jsonl or hipho_to.jsonl")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--qpm", type=int, default=200)
    ap.add_argument("--judge-model", default="gemini-2.5-flash",
                    help="model the official grader calls for step-level grading; "
                         "the default is the one it uses upstream")
    ap.add_argument("--judge-max-tokens", type=int, default=4096,
                    help="token budget per judge call")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="judge sampling temperature; some deployments reject 0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    scorer, mod = shim.new_scorer()
    print(f"[judge] official grader: {mod.HiPhODataset.__module__}.HiPhODataset "
          f"(vendored, see official_impl/README.md)")

    dataset = R.load_dataset(args.dataset)
    print(f"[judge] reference items: {len(dataset)}   judge model: {args.judge_model}")

    # Both grading levels read the whole response.
    tasks = R.load_rollouts(args.input_dir, args.rollouts, dataset,
                            answer_source="prediction", question_fixup=None)
    if not tasks:
        print("[error] nothing to grade")
        return 1
    R.check_count(tasks, len(dataset), args.rollouts)

    judge = JudgeModel(args.judge_model, args.qpm, args.judge_max_tokens, args.temperature)
    print(f"[judge] model={args.judge_model} temperature={args.temperature} "
          f"max_tokens={args.judge_max_tokens} workers={args.workers} qpm={args.qpm}")

    def work(t):
        row = to_official_row(dataset[t["filename"]], t["pred"])
        r = scorer._evaluate_single_problem(judge, row, 0, {})   # official method, unmodified
        return {"round": t["round"], "filename": t["filename"],
                "fine": r["fine_grained_score"],
                "coarse": r["coarse_grained_score"],
                "final": r["final_score"],
                "max": r["item_total_points"],
                "extracted_pred": str(r.get("extracted_pred"))[:300],
                "n_criteria": len(r.get("marking_detailed_scores") or [])}

    indir = Path(args.input_dir)
    ckpt = indir / "judge_hipho_official_checkpoint.jsonl"
    impl = f"official-{args.judge_model}"
    done, failed = run_batch(tasks, lambda t: t["_key"], work, str(ckpt),
                             workers=args.workers, passes=4, impl=impl)
    if failed:
        print(f"[warn] {len(failed)} items failed every pass and are excluded")

    n_exams_dataset = len({row.get("source") for row in dataset.values()})
    is_subset = len(dataset) != 360
    report = {
        "bench": "hipho",
        "impl": "VLMEvalKit vlmeval/dataset/hipho.py, vendored unmodified",
        "judge_model": args.judge_model,
        "judge_params": {"temperature": args.temperature,
                         "max_tokens": args.judge_max_tokens,
                         "workers": args.workers, "qpm": args.qpm,
                         "rollouts": args.rollouts},
        "dataset": args.dataset,
        "n_reference": len(dataset),
        "n_exams": n_exams_dataset,
        "n_judge_failed": len(failed),
        "note": (f"The official protocol samples each problem 8 times and averages; "
                 f"this run used {args.rollouts} round(s)."),
        "rounds": {},
    }
    if is_subset:
        report["subset_warning"] = (
            f"This dataset has {len(dataset)} problems, not the full 360. Per-exam point "
            f"totals cover part of each exam, so leaderboard_points_mean here compares "
            f"only with another run on the same subset."
        )

    leaderboard_per_round = []
    rate_per_round = []
    for k in range(1, args.rollouts + 1):
        rows = [r for r in done.values() if r.get("round") == k]
        if not rows:
            continue
        points = sum(r["final"] for r in rows)
        maximum = sum(r["max"] for r in rows)
        rate = 100.0 * points / maximum if maximum else 0.0

        def group(field):
            g = collections.defaultdict(list)
            for r in rows:
                g[str(dataset[r["filename"]].get(field))].append(r)
            out = {}
            for key, v in sorted(g.items()):
                p = sum(x["final"] for x in v)
                m = sum(x["max"] for x in v)
                out[key] = {"n": len(v), "points": round(p, 2), "max": round(m, 2),
                            "score_rate": round(100.0 * p / m, 2) if m else 0.0}
            return out

        per_exam = group("source")
        leaderboard = statistics.mean(v["points"] for v in per_exam.values())

        report["rounds"][k] = {
            "n": len(rows),
            "leaderboard_points_mean": round(leaderboard, 2),
            "score_rate": round(rate, 2),
            "points": round(points, 2),
            "max_points": round(maximum, 2),
            "fine_total": round(sum(r["fine"] for r in rows), 2),
            "coarse_total": round(sum(r["coarse"] for r in rows), 2),
            "n_fine_wins": sum(1 for r in rows if r["fine"] > r["coarse"]),
            "n_coarse_wins": sum(1 for r in rows if r["coarse"] > r["fine"]),
            "per_exam": per_exam,
            "per_modality": group("modality"),
            "per_field": group("field"),
        }
        leaderboard_per_round.append(round(leaderboard, 2))
        rate_per_round.append(round(rate, 2))

    report["summary"] = {
        "metric": "leaderboard_points_mean = arithmetic mean of per-exam raw point totals",
        "per_round_leaderboard_points": leaderboard_per_round,
        "mean_leaderboard_points": round(statistics.mean(leaderboard_per_round), 2)
            if leaderboard_per_round else 0.0,
        "std_leaderboard_points": round(statistics.pstdev(leaderboard_per_round), 2)
            if len(leaderboard_per_round) > 1 else 0.0,
        "per_round_score_rate": rate_per_round,
        "mean_score_rate": round(statistics.mean(rate_per_round), 2) if rate_per_round else 0.0,
    }

    out = Path(args.out) if args.out else indir / "judge_hipho_official_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"HiPhO, official grader, judge model {args.judge_model}")
    s = report["summary"]
    print(f"  leaderboard (mean of {n_exams_dataset} exams' raw points): "
          f"{s['per_round_leaderboard_points']}  mean {s['mean_leaderboard_points']}")
    print(f"  score rate (diagnostic, not the leaderboard metric): "
          f"{s['per_round_score_rate']}  mean {s['mean_score_rate']}%")
    if is_subset:
        print(f"  NOTE  {len(dataset)} problems, not 360 -- comparable only with "
              f"another run on the same subset")
    r1 = report["rounds"].get(1)
    if r1:
        print(f"  round 1: step-level total {r1['fine_total']}, answer-level total "
              f"{r1['coarse_total']}, final = max of the two = {r1['points']}/{r1['max_points']}")
        print(f"           step higher on {r1['n_fine_wins']} problems, "
              f"answer higher on {r1['n_coarse_wins']}")
        print("\n  per exam (points / max)")
        for name, v in r1["per_exam"].items():
            print(f"    {name:24s} {v['points']:7.2f}/{v['max']:7.2f} = "
                  f"{v['score_rate']:6.2f}%  (n={v['n']})")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
