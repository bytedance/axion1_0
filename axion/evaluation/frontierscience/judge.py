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

"""Grade FrontierScience-Olympiad (FS-O) and FrontierScience-Research (FS-R).

    export OPENAI_API_KEY=...
    export OPENAI_MODEL=...
    python judge.py --bench fs_o --dataset fs_o.jsonl --input-dir rollouts/ --rollouts 3

Judge prompts are `prompts/fs_o_judge.txt` and `prompts/fs_r_judge.txt`,
extracted verbatim from the FrontierScience paper's appendix. Both end in a
`VERDICT:` line, which is what this script parses.

FS-O compares a single final answer and is reported as accuracy. FS-R grades
against a rubric totalling 10 points; the official metric is the fraction of
responses scoring at least `--fsr-threshold` (7.0), reported as `mean_pct`.
`alt_mean_rubric_pct` is the continuous mean of rubric fractions.

Judge sampling parameters are flags and are recorded in the report.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if __package__ and __package__.startswith("axion."):
    from ..utils import rollouts as R
    from ...datagen.utils.llm_client import LLMClient, run_batch
else:
    sys.path.insert(0, str(HERE.parent / "utils"))
    sys.path.insert(0, str(HERE.parent.parent / "datagen" / "utils"))

    import rollouts as R  # noqa: E402
    from llm_client import LLMClient, run_batch  # noqa: E402

EXPECTED_COUNT = {"fs_o": 100, "fs_r": 60}

#: Default token budget per judge call, per benchmark.
DEFAULT_JUDGE_MAX_TOKENS = {"fs_o": 16384, "fs_r": 32768}

FSO_PROMPT = (HERE / "prompts" / "fs_o_judge.txt").read_text(encoding="utf-8").strip()
FSR_PROMPT = (HERE / "prompts" / "fs_r_judge.txt").read_text(encoding="utf-8").strip()

import re  # noqa: E402

_VERDICT_RE = re.compile(r"VERDICT\s*[:：]\s*[\"“”']?\s*([A-Za-z0-9_.\-]+)",
                         re.IGNORECASE)


def parse_verdict(text: str):
    """Value of the last `VERDICT:` line, or None if there is none."""
    hits = _VERDICT_RE.findall(text or "")
    return hits[-1].strip() if hits else None


def verdict_correct(text: str):
    """FS-O: CORRECT / INCORRECT -> True / False, None when unparseable."""
    v = parse_verdict(text)
    if v is None:
        return None
    v = v.upper()
    if v.startswith("INCORRECT"):
        return False
    if v.startswith("CORRECT"):
        return True
    return None


def verdict_points(text: str):
    """FS-R: VERDICT: <total> -> float, None when unparseable."""
    v = parse_verdict(text)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


#: FS-R rubric items, `Points: 0.5, Item: ...`; their points sum to the maximum.
RUBRIC_ITEM_RE = re.compile(r"^Points:\s*([0-9]*\.?[0-9]+)\s*,\s*Item:", re.M)


def rubric_max(answer: str) -> float:
    """Total points available in a rubric."""
    return sum(float(x) for x in RUBRIC_ITEM_RE.findall(answer or ""))


def clip(s, n: int) -> str:
    """Truncate to `n` characters, noting the original length."""
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"\n...[truncated, original length {len(s)}]"


def impl_fingerprint(bench: str) -> str:
    """Fingerprint of the prompt text and grading code, stored per checkpoint row.

    Cached rows carrying a different fingerprint are re-judged rather than
    reused, so changing the prompt or `judge_one` invalidates them.
    """
    parts = [bench, FSO_PROMPT, FSR_PROMPT]
    try:
        parts.append(inspect.getsource(judge_one))
    except OSError:
        parts.append("<source unavailable>")
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def judge_one(client: LLMClient, bench: str, reference: str, pred: str, question: str,
              temperature: float, max_tokens: int) -> dict:
    """Grade one response. Returns points earned, maximum, and a score in [0,1]."""
    pred = clip(pred, 60000)
    question = clip(question, 20000)

    if bench == "fs_o":
        # One official placeholder name contains a space, so str.format cannot be used.
        prompt = (FSO_PROMPT
                  .replace("{problem}", question)
                  .replace("{reference answer}", clip(reference, 8000))
                  .replace("{answer}", pred))
        raw = client.chat([{"role": "user", "content": prompt}],
                          validator=lambda t: verdict_correct(t) is not None,
                          temperature=temperature, max_tokens=max_tokens)
        ok = verdict_correct(raw)
        return {"score": 1.0 if ok else 0.0, "points": 1.0 if ok else 0.0, "max": 1.0,
                "verdict_tail": raw[-600:]}

    mx = rubric_max(reference) or 10.0
    prompt = (FSR_PROMPT
              .replace("{problem}", question)
              .replace("{rubric}", clip(reference, 40000))
              .replace("{answer}", pred))
    raw = client.chat([{"role": "user", "content": prompt}],
                      validator=lambda t: verdict_points(t) is not None,
                      temperature=temperature, max_tokens=max_tokens)
    pts = max(0.0, min(float(verdict_points(raw)), mx))
    return {"score": pts / mx if mx else 0.0, "points": pts, "max": mx,
            "verdict_tail": raw[-600:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, choices=["fs_o", "fs_r"])
    ap.add_argument("--dataset", required=True,
                    help="jsonl from prepare.py (filename / question / answer)")
    ap.add_argument("--input-dir", required=True, help="directory holding iter1.jsonl ...")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="judge concurrency")
    ap.add_argument("--qpm", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="judge sampling temperature; some deployments reject 0")
    ap.add_argument("--judge-max-tokens", type=int, default=0,
                    help="token budget per judge call (default: 16384 for fs_o, "
                         "32768 for fs_r)")
    ap.add_argument("--fsr-threshold", type=float, default=7.0,
                    help="FS-R: rubric points out of 10 at which a response counts as "
                         "a success (the paper's value is 7.0)")
    ap.add_argument("--answer-source", choices=["prediction", "last_assistant", "auto"],
                    default="auto",
                    help="auto: fs_o grades the final answer, fs_r grades the whole "
                         "last assistant turn (see utils/rollouts.py)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dataset = R.load_dataset(args.dataset)
    print(f"[judge] bench={args.bench}  reference answers: {len(dataset)}")

    source = args.answer_source
    if source == "auto":
        source = "prediction" if args.bench == "fs_o" else "last_assistant"
    print(f"[judge] grading text: {source}")

    tasks = R.load_rollouts(args.input_dir, args.rollouts, dataset, answer_source=source)
    if not tasks:
        print("[error] nothing to grade")
        return 1
    R.check_count(tasks, EXPECTED_COUNT[args.bench], args.rollouts)

    judge_max_tokens = args.judge_max_tokens or DEFAULT_JUDGE_MAX_TOKENS[args.bench]
    client = LLMClient(qpm=args.qpm)
    print(f"[judge] model={client.model} temperature={args.temperature} "
          f"max_tokens={judge_max_tokens} workers={args.workers} qpm={args.qpm}")

    def work(t):
        if R.is_no_answer(t["pred"]):
            res = {"score": 0.0, "points": 0.0, "max": 1.0, "no_answer": True}
        else:
            res = judge_one(client, args.bench, dataset[t["filename"]]["answer"],
                            t["pred"], t["question"],
                            temperature=args.temperature, max_tokens=judge_max_tokens)
        res.update({"round": t["round"], "filename": t["filename"],
                    "termination": t["termination"]})
        return res

    indir = Path(args.input_dir)
    ckpt = indir / f"judge_{args.bench}_{source}_checkpoint.jsonl"
    impl = impl_fingerprint(args.bench)
    print(f"[judge] impl={impl} (cached rows with a different fingerprint are re-judged)")
    done, failed = run_batch(tasks, lambda t: t["_key"], work, str(ckpt),
                             workers=args.workers, passes=4, impl=impl)
    if failed:
        print(f"[warn] {len(failed)} items failed every pass and are excluded")

    by_round = collections.defaultdict(list)
    for row in done.values():
        by_round[row["round"]].append(row)

    report = {
        "bench": args.bench,
        "dataset": args.dataset,
        "input_dir": str(indir),
        "answer_source": source,
        "impl": impl,
        "judge_model": client.model,
        "judge_params": {"temperature": args.temperature,
                         "max_tokens": judge_max_tokens,
                         "workers": args.workers, "qpm": args.qpm,
                         "rollouts": args.rollouts},
        "n_reference": len(dataset),
        "n_judge_failed": len(failed),
        "rounds": {},
    }
    per_round = []
    for k in sorted(by_round):
        rows = by_round[k]
        if args.bench == "fs_r":
            score = 100.0 * statistics.mean(
                1.0 if r["points"] >= args.fsr_threshold else 0.0 for r in rows)
            entry = {
                "n": len(rows),
                "mean_pct": round(score, 2),
                "metric": f"official: one response scoring >= {args.fsr_threshold}/10 "
                          f"counts as a success",
                "alt_mean_rubric_pct": round(
                    100.0 * statistics.mean(r["score"] for r in rows), 2),
                "mean_points_of_10": round(statistics.mean(r["points"] for r in rows), 3),
            }
        else:
            score = 100.0 * statistics.mean(r["score"] for r in rows)
            entry = {"n": len(rows), "mean_pct": round(score, 2)}
        entry["no_answer"] = sum(1 for r in rows if r.get("no_answer"))
        report["rounds"][k] = entry
        per_round.append(score)

    report["summary"] = {
        "per_round_pct": [round(x, 2) for x in per_round],
        "mean_pct": round(statistics.mean(per_round), 2) if per_round else 0.0,
        "std_pct": round(statistics.pstdev(per_round), 2) if len(per_round) > 1 else 0.0,
    }
    if args.bench == "fs_r":
        report["judge_params"]["fsr_threshold"] = args.fsr_threshold
        report["note"] = (
            f"Official metric: fraction of responses scoring >= {args.fsr_threshold}/10 "
            f"rubric points. alt_mean_rubric_pct is the continuous mean of rubric "
            f"fractions, a different quantity."
        )

    out = Path(args.out) if args.out else indir / f"judge_{args.bench}_{source}_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{args.bench}  per round: {report['summary']['per_round_pct']}")
    print(f"{args.bench}  mean {report['summary']['mean_pct']}%  "
          f"(std {report['summary']['std_pct']})")
    if args.bench == "fs_r":
        alt = [report["rounds"][k]["alt_mean_rubric_pct"] for k in sorted(report["rounds"])]
        print(f"{args.bench}  continuous rubric mean (lower noise, internal use): {alt}")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
