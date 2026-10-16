#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Grade SEAL-0: is the model's answer equivalent to the reference answer?

    export OPENAI_API_KEY=...
    export OPENAI_MODEL=...
    python judge.py --dataset seal0_<date>.jsonl --input-dir rollouts/ --rollouts 3

Reports `mean_pct` (accuracy averaged over rounds, Avg. Pass@N) and
`pass_at_n` (answered correctly in at least one round, Pass@N), plus
breakdowns by `freshness` and `effective_year`.

Two judge prompts ship here. `--prompt from_a1` (default) is the Agents-A1
reference judge, extracted verbatim, with binary "Correct" / "Incorrect"
output. `--prompt official` is the SealQA paper's appendix prompt, with
three-way A / B / C output. The verdict parser accepts both.

The judge sees the wrapped question, the one the model was given, matching the
reference evaluation. `raw_question` is used only for grouping. Judge model and
sampling parameters are recorded in the report.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))
sys.path.insert(0, str(HERE.parent.parent / "datagen" / "utils"))

import rollouts as R  # noqa: E402
from llm_client import LLMClient, run_batch  # noqa: E402

PROMPT_FILES = {
    "from_a1": "judge_from_a1.txt",
    "official": "judge_official.txt",
}


def is_correct(verdict: str) -> bool:
    """True when the verdict is "correct" or starts with A, as in the reference rule.

    The A1 prompt answers "Correct"; the official prompt answers A for correct.
    """
    low = (verdict or "").strip().lower()
    if not low:
        return False
    return low.startswith("correct") or low[0] == "a"


def parseable(verdict: str) -> bool:
    """True when the verdict has a recognised shape.

    An unrecognised or empty verdict is a judge failure, not a wrong answer, so
    it is retried and finally reported as a failed item.
    """
    low = (verdict or "").strip().lower()
    if not low:
        return False
    return (low.startswith("correct") or low.startswith("incorrect")
            or low.startswith("not attempted") or low[0] in "abc")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="jsonl from prepare.py")
    ap.add_argument("--input-dir", required=True, help="directory holding iter1.jsonl ...")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="judge concurrency")
    ap.add_argument("--qpm", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="judge sampling temperature; some deployments reject 0")
    ap.add_argument("--judge-max-tokens", type=int, default=4096,
                    help="token budget per judge call")
    ap.add_argument("--prompt", choices=sorted(PROMPT_FILES), default="from_a1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prompt_path = HERE / "prompts" / PROMPT_FILES[args.prompt]
    prompt_text = prompt_path.read_text(encoding="utf-8")
    print(f"[judge] prompt: {args.prompt} ({prompt_path.name})")

    dataset = R.load_dataset(args.dataset)
    print(f"[judge] reference answers: {len(dataset)}")

    tasks = R.load_rollouts(args.input_dir, args.rollouts, dataset,
                            answer_source="prediction", question_fixup=None)
    if not tasks:
        print("[error] nothing to grade")
        return 1
    R.check_count(tasks, len(dataset), args.rollouts)

    client = LLMClient(qpm=args.qpm)
    print(f"[judge] model={client.model} temperature={args.temperature} "
          f"max_tokens={args.judge_max_tokens} workers={args.workers} qpm={args.qpm}")

    def work(t):
        gt = dataset[t["filename"]]
        question = gt.get("question", "")
        reference = gt.get("answer", "")
        pred = t["pred"]
        if args.prompt == "official":
            msg = prompt_text.format(question=question, target=reference,
                                     predicted_answer=pred)
        else:
            msg = prompt_text.format(question=question, correct_answer=reference,
                                     response=pred)
        raw = client.chat([{"role": "user", "content": msg}],
                          validator=parseable, temperature=args.temperature,
                          max_tokens=args.judge_max_tokens)
        return {"round": t["round"], "filename": t["filename"],
                "is_correct": is_correct(raw),
                "verdict": (raw or "").strip()[:200],
                "prediction": pred[:2000],
                "effective_year": gt.get("effective_year"),
                "freshness": gt.get("freshness"),
                "topic": gt.get("topic")}

    indir = Path(args.input_dir)
    ckpt = indir / f"judge_seal0_{args.prompt}_checkpoint.jsonl"
    impl = f"seal0-{args.prompt}"
    done, failed = run_batch(tasks, lambda t: t["_key"], work, str(ckpt),
                             workers=args.workers, passes=4, impl=impl)

    rows = list(done.values())
    by_round = collections.defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)

    per_round = []
    rounds = {}
    for k in sorted(by_round):
        rs = by_round[k]
        acc = 100.0 * sum(1 for r in rs if r["is_correct"]) / len(rs)
        rounds[k] = {"n": len(rs), "correct": sum(1 for r in rs if r["is_correct"]),
                     "accuracy_pct": round(acc, 2)}
        per_round.append(acc)

    solved = collections.defaultdict(bool)
    for r in rows:
        solved[r["filename"]] = solved[r["filename"]] or r["is_correct"]
    pass_at_n = 100.0 * sum(1 for v in solved.values() if v) / len(solved) if solved else 0.0

    def breakdown(field):
        g = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            g[str(r.get(field))][1] += 1
            g[str(r.get(field))][0] += bool(r["is_correct"])
        return {k: {"correct": v[0], "n": v[1],
                    "accuracy_pct": round(100.0 * v[0] / v[1], 2) if v[1] else 0.0}
                for k, v in sorted(g.items())}

    report = {
        "bench": "seal0",
        "dataset": args.dataset,
        "input_dir": str(indir),
        "judge_prompt": args.prompt,
        "judge_model": client.model,
        "judge_params": {"temperature": args.temperature,
                         "max_tokens": args.judge_max_tokens,
                         "workers": args.workers, "qpm": args.qpm,
                         "rollouts": args.rollouts},
        "impl": impl,
        "n_reference": len(dataset),
        "n_judge_failed": len(failed),
        "rounds": rounds,
        "summary": {
            "per_round_pct": [round(x, 2) for x in per_round],
            "mean_pct": round(statistics.mean(per_round), 2) if per_round else 0.0,
            "std_pct": round(statistics.pstdev(per_round), 2) if len(per_round) > 1 else 0.0,
            "pass_at_n": round(pass_at_n, 2),
            "n_rounds": len(per_round),
        },
        "by_freshness": breakdown("freshness"),
        "by_effective_year": breakdown("effective_year"),
    }
    if failed:
        report["invalid"] = (
            f"{len(failed)} item(s) could never be graded (the judge kept returning "
            f"an unparseable or empty verdict). They are excluded from the denominator, "
            f"but the usual cause is judge-side rate limiting, so lower --workers and "
            f"re-run before reporting this score."
        )

    out = Path(args.out) if args.out else indir / f"judge_seal0_{args.prompt}_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print("\n" + "=" * 60)
    print(f"SEAL-0  per round: {s['per_round_pct']}")
    print(f"SEAL-0  Avg. Pass@{s['n_rounds']} {s['mean_pct']}%  (std {s['std_pct']})   "
          f"Pass@{s['n_rounds']} {s['pass_at_n']}%")
    print("  by freshness: " + "  ".join(
        f"{k}={v['correct']}/{v['n']}({v['accuracy_pct']:.0f}%)"
        for k, v in report["by_freshness"].items()))
    if failed:
        print(f"  WARNING: {len(failed)} item(s) could not be graded at all -- "
              f"lower --workers and re-run before reporting")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
