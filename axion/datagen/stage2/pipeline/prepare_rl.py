# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Prepare paper RL tasks from four offline trajectories per current checkpoint.

Task JSONL::

    {"id":"q1", "question":"...", "reference":{"answer":"42",
     "answer_type":"numeric", "tolerance":0, "absolute_tolerance":0},
     "hops":[{"id":"entity", "description":"Identify the enzyme", "match":"all",
              "atoms":[{"type":"str", "value":"P12345", "grounded":true}]}]}

``hops`` lists only intermediate targets in dependency order; an explicit empty
list means no intermediate targets. Text answers use answer_type="text" and
may supply also_acceptable aliases. Numeric references and atoms require
explicit relative tolerance and absolute_tolerance, including zero for exact
matching. Numeric atoms may supply positive scales for known unit conversions;
string atoms may supply aliases. Every atom declares whether tool evidence is
required; known identifier spellings always require it. Legacy tree/critical
flags are deliberately not interpreted as the paper's ordered targets.

Attempt JSONL (exactly four unique attempt_id values per task)::

    {"task_id":"q1", "attempt_id":"0", "model_checkpoint":"revision-or-digest",
     "messages":[{"role":"assistant", "content":"<answer>42</answer>"}]}

Messages must contain the complete original trajectory. Tool observations use
role="tool", or the existing agent's user-role <tool_response> blocks following
an assistant tool call. Never put the reference or hops into the sampled model's
prompt: use only question.

Both heldout and defects are required JSON manifests with task_ids and
questions lists. Match ids and exact question text after whitespace/case
normalization. Include any externally reviewed near-duplicate exclusions in
these manifests; no semantic-overlap threshold is guessed here.

Use ``candidates`` before sampling to apply these exclusions, then ``select``
to score the four raw attempts with the same reward used in training. Sampling
itself is performed by the deployment's agent harness at the stated checkpoint.
The parquet and companion .selection.json record the exact selection evidence.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from axion.training.rl.process_reward import Judge, score, validate_hops, validate_reference


ATTEMPTS_PER_TASK = 4
CRITERION = "at_least_one_correct_and_nonzero_reward_variance"


def _question_key(value: str) -> str:
    return " ".join(value.casefold().split())


def validate_tasks(tasks: list[dict]) -> list[dict]:
    normalized, ids, questions = [], set(), set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"].strip():
            raise ValueError("Every task requires a nonempty string id")
        if task["id"] in ids:
            raise ValueError(f"Duplicate task id: {task['id']}")
        ids.add(task["id"])
        question = task.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Task {task['id']} requires a nonempty question")
        key = _question_key(question)
        if key in questions:
            raise ValueError("Duplicate task question")
        questions.add(key)
        if "hops" not in task:
            raise ValueError(f"Task {task['id']} requires explicit ordered hops")
        normalized.append({"id": task["id"], "question": question,
                           "reference": validate_reference(task.get("reference")),
                           "hops": validate_hops(task["hops"])})
    return normalized


def _exclusions(manifest: Any) -> tuple[set[str], set[str]]:
    if not isinstance(manifest, dict):
        raise ValueError("A reviewed exclusion manifest is required")
    for key in ("task_ids", "questions"):
        if not isinstance(manifest.get(key), list) or any(not isinstance(x, str) for x in manifest[key]):
            raise ValueError(f"Exclusion manifest requires a {key} string list (possibly empty)")
    return set(manifest["task_ids"]), {_question_key(q) for q in manifest["questions"]}


def eligible_tasks(tasks: list[dict], heldout: dict, defects: dict) -> tuple[list[dict], dict[str, str]]:
    tasks = validate_tasks(tasks)
    exclusions = [("heldout", _exclusions(heldout)), ("defects", _exclusions(defects))]
    eligible, rejected = [], {}
    for task in tasks:
        for kind, (ids, questions) in exclusions:
            if task["id"] in ids or _question_key(task["question"]) in questions:
                rejected[task["id"]] = kind
                break
        else:
            eligible.append(task)
    return eligible, rejected


async def select_tasks(*, tasks: list[dict], attempts: list[dict], heldout: dict, defects: dict,
                       model_checkpoint: str, judge: Judge | None = None) -> tuple[list[dict], dict]:
    """Score original trajectories; cached or supplied reward numbers are ignored."""
    if not isinstance(model_checkpoint, str) or not model_checkpoint.strip():
        raise ValueError("An explicit model_checkpoint identity is required")
    eligible, rejected = eligible_tasks(tasks, heldout, defects)
    task_ids = {task["id"] for task in tasks}
    groups = defaultdict(list)
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("task_id") not in task_ids:
            raise ValueError("Attempt references an unknown task_id")
        groups[attempt["task_id"]].append(attempt)
    rows, evidence = [], {}
    for task in eligible:
        task_id = task["id"]
        group = groups[task_id]
        if len(group) != ATTEMPTS_PER_TASK:
            rejected[task_id] = "attempt_count"
            continue
        if any(item.get("model_checkpoint") != model_checkpoint for item in group):
            rejected[task_id] = "model_checkpoint"
            continue
        ids = [item.get("attempt_id") for item in group]
        if any(not isinstance(item, str) or not item.strip() for item in ids):
            rejected[task_id] = "invalid_attempt_id"
            continue
        if len(set(ids)) != ATTEMPTS_PER_TASK:
            rejected[task_id] = "duplicate_attempt"
            continue
        results = [await score(messages=attempt.get("messages"), reference=task["reference"],
                               hops=task["hops"], question=task["question"], judge=judge)
                   for attempt in group]
        rewards = [result["score"] for result in results]
        correctness = [bool(result.get("final_ok")) for result in results]
        evidence[task_id] = {"attempt_ids": ids, "rewards": rewards, "final_correct": correctness,
                             "model_checkpoint": model_checkpoint, "attempts_per_task": ATTEMPTS_PER_TASK}
        if any(result["drop_from_training"] for result in results):
            rejected[task_id] = "evaluation_failed"
            continue
        if not any(correctness):
            rejected[task_id] = "no_correct_final"
            continue
        if max(rewards) == min(rewards):
            rejected[task_id] = "zero_reward_variance"
            continue
        rows.append({
            "data_source": "axion_paper_rl",
            "prompt": [{"role": "user", "content": task["question"]}],
            "reward_model": {"style": "rule", "ground_truth": {**task["reference"], "task_id": task_id}},
            "extra_info": {"task_id": task_id, "question": task["question"], "hops": task["hops"],
                           "selection": evidence[task_id]},
        })
    report = {"schema_version": 1, "selected_count": len(rows), "model_checkpoint": model_checkpoint,
              "attempts_per_task": ATTEMPTS_PER_TASK, "criterion": CRITERION,
              "rejected": rejected, "rejected_counts": dict(Counter(rejected.values())),
              "scored_attempts": evidence}
    return rows, report


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset(rows: list[dict], output: Path, report: dict, inputs: dict[str, Path]) -> None:
    """Write parquet plus a manifest binding selections and exclusion inputs."""
    if not rows:
        raise ValueError("No task passed selection; refusing to create an empty training dataset")
    if output.suffix != ".parquet":
        raise ValueError("Training output must have .parquet suffix")
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    temporary.replace(output)
    metadata = {**report, "parquet_sha256": _sha256(output),
                "input_sha256": {name: _sha256(path) for name, path in inputs.items()}}
    manifest = output.with_suffix(".selection.json")
    manifest_tmp = manifest.with_name(manifest.name + f".tmp.{os.getpid()}")
    manifest_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("candidates", "select"):
        command = commands.add_parser(name)
        command.add_argument("--tasks", type=Path, required=True, help="Canonical task JSONL with explicit ordered hops")
        command.add_argument("--heldout", type=Path, required=True, help="Reviewed held-out exclusion JSON")
        command.add_argument("--defects", type=Path, required=True, help="Reviewed known-defect exclusion JSON")
        command.add_argument("--out", type=Path, required=True)
        if name == "select":
            command.add_argument("--attempts", type=Path, required=True, help="Four raw trajectories per task, JSONL")
            command.add_argument("--model-checkpoint", required=True, help="Exact identity recorded in all four attempts")
            command.add_argument("--judge-address", default=os.environ.get("S2_JUDGE_BASE_URL"))
            command.add_argument("--judge-model", default=os.environ.get("S2_JUDGE_MODEL"))
    args = parser.parse_args(argv)
    tasks = _jsonl(args.tasks)
    heldout = json.loads(args.heldout.read_text(encoding="utf-8"))
    defects = json.loads(args.defects.read_text(encoding="utf-8"))
    if args.command == "candidates":
        eligible, rejected = eligible_tasks(tasks, heldout, defects)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eligible), encoding="utf-8")
        print(json.dumps({"eligible": len(eligible), "rejected_counts": dict(Counter(rejected.values()))}))
        return
    if bool(args.judge_address) != bool(args.judge_model):
        parser.error("--judge-address and --judge-model must be supplied together")
    judge = None
    if args.judge_address:
        from axion.training.rl.reward import make_paper_judge
        judge = make_paper_judge(eval_llm_address=args.judge_address, eval_llm_model=args.judge_model)
    rows, report = asyncio.run(select_tasks(tasks=tasks, attempts=_jsonl(args.attempts), heldout=heldout,
                                            defects=defects, model_checkpoint=args.model_checkpoint, judge=judge))
    write_dataset(rows, args.out, report, {name: getattr(args, name) for name in ("tasks", "attempts", "heldout", "defects")})
    print(json.dumps({"selected_count": len(rows), "rejected_counts": report["rejected_counts"]}))


if __name__ == "__main__":
    main()
