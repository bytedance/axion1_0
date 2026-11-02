"""Offline selection exercises raw attempts and the actual parquet writer."""
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(os.environ.get("AXION_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))


def selection_module():
    path = ROOT / "axion/datagen/stage2/pipeline/prepare_rl.py"
    assert path.is_file(), "Offline four-attempt paper selection must be packaged"
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("paper_selection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task(task_id="q1"):
    return {"id": task_id, "question": "Compute the reference quantity.",
            "reference": {"answer": "42", "answer_type": "numeric", "tolerance": 0, "absolute_tolerance": 0},
            "hops": []}


def attempts(answers=("42", "99", "99", "99"), *, task_id="q1"):
    return [{"task_id": task_id, "attempt_id": str(i), "model_checkpoint": "checkpoint-sha",
             "messages": [{"role": "assistant", "content": f"<answer>{answer}</answer>"}]}
            for i, answer in enumerate(answers)]


EMPTY = {"task_ids": [], "questions": []}


def select(records, *, tasks=None, heldout=None, defects=None, judge=None):
    return asyncio.run(selection_module().select_tasks(
        tasks=tasks if tasks is not None else [task()], attempts=records,
        heldout=heldout if heldout is not None else EMPTY,
        defects=defects if defects is not None else EMPTY,
        model_checkpoint="checkpoint-sha", judge=judge))


def test_selects_four_current_model_attempts_and_preserves_question_reference_targets():
    item = task()
    item["hops"] = [{"id": "parameter", "description": "Find the parameter", "match": "all",
                     "atoms": [{"type": "str", "value": "P12345", "grounded": True}]}]
    rows, report = select(attempts(), tasks=[item])
    assert len(rows) == 1
    assert rows[0]["prompt"] == [{"role": "user", "content": item["question"]}]
    assert rows[0]["reward_model"]["ground_truth"]["answer"] == "42"
    assert rows[0]["extra_info"]["hops"] == [
        {"id": "parameter", "description": "Find the parameter", "match": "all",
         "atoms": [{"type": "str", "value": "P12345", "grounded": True, "aliases": []}]}]
    assert rows[0]["extra_info"]["selection"]["rewards"] == [1, 0, 0, 0]
    assert report["selected_count"] == 1


@pytest.mark.parametrize("records,reason", [
    (attempts(("42", "42", "42", "42")), "zero_reward_variance"),
    (attempts(("99", "99", "99", "99")), "no_correct_final"),
    (attempts(("42", "99", "99")), "attempt_count"),
    (attempts(("42", "99", "99", "99", "99")), "attempt_count"),
])
def test_rejects_degenerate_or_incomplete_offline_groups(records, reason):
    rows, report = select(records)
    assert rows == []
    assert report["rejected"]["q1"] == reason


def test_process_variance_without_any_correct_final_is_insufficient():
    item = task()
    item["hops"] = [{"id": "h1", "description": "Find mass", "match": "all", "atoms": [
        {"type": "num", "value": 100, "tolerance": 0, "absolute_tolerance": 0, "grounded": False}]}]
    records = attempts(("99", "99", "99", "99"))
    for row in records[1:]:
        row["messages"][0]["content"] = "<think>I use mass 100.</think><answer>99</answer>"
    async def judge(prompt):
        return '{"1.1":true}'
    rows, report = select(records, tasks=[item], judge=judge)
    assert rows == []
    assert report["rejected"]["q1"] == "no_correct_final"
    assert report["scored_attempts"]["q1"]["rewards"] == [0, .3, .3, .3]


@pytest.mark.parametrize("change,reason", [("checkpoint", "model_checkpoint"), ("duplicate", "duplicate_attempt")])
def test_rejects_mixed_checkpoints_and_duplicate_attempts(change, reason):
    records = attempts()
    if change == "checkpoint":
        records[1]["model_checkpoint"] = "old-checkpoint"
    else:
        records[1]["attempt_id"] = records[0]["attempt_id"]
    rows, report = select(records)
    assert rows == []
    assert report["rejected"]["q1"] == reason


@pytest.mark.parametrize("kind,manifest", [
    ("heldout", {"task_ids": ["q1"], "questions": []}),
    ("heldout", {"task_ids": [], "questions": ["  Compute   the reference quantity.  "]}),
    ("defects", {"task_ids": ["q1"], "questions": []}),
])
def test_exclusion_manifests_apply_before_reward_evaluation(kind, manifest):
    rows, report = select(attempts(), **{kind: manifest})
    assert rows == []
    assert report["rejected"]["q1"] == kind
    assert "q1" not in report["scored_attempts"]


def test_ambiguous_legacy_targets_are_rejected_before_probing():
    item = task()
    item.pop("hops")
    item["tree"] = {"steps": [{"expected": "P12345", "critical": True}]}
    with pytest.raises(ValueError, match="hops"):
        select(attempts(), tasks=[item])


def test_required_judge_failure_rejects_entire_offline_group():
    records = attempts()
    records[1]["messages"][0]["content"] = "<answer>The final result after reasoning is ninety nine.</answer>"
    async def unavailable(prompt):
        return None
    rows, report = select(records, judge=unavailable)
    assert rows == []
    assert report["rejected"]["q1"] == "evaluation_failed"


def test_numeric_tolerance_must_be_explicit_in_input_contract():
    item = task()
    item["reference"].pop("tolerance")
    with pytest.raises(ValueError, match="tolerance"):
        select(attempts(), tasks=[item])


def test_cli_scores_raw_attempts_writes_real_parquet_and_auditable_manifest(tmp_path):
    files = {}
    for name, content in {"tasks": [task()], "attempts": attempts(), "heldout": EMPTY, "defects": EMPTY}.items():
        path = tmp_path / (name + (".jsonl" if isinstance(content, list) else ".json"))
        path.write_text("".join(json.dumps(row) + "\n" for row in content) if isinstance(content, list)
                        else json.dumps(content))
        files[name] = path
    output = tmp_path / "train.parquet"
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "axion.datagen.stage2.pipeline.prepare_rl", "select",
                             "--tasks", str(files["tasks"]), "--attempts", str(files["attempts"]),
                             "--heldout", str(files["heldout"]), "--defects", str(files["defects"]),
                             "--model-checkpoint", "checkpoint-sha", "--out", str(output)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    import pyarrow.parquet as pq
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 1
    assert rows[0]["extra_info"]["task_id"] == "q1"
    manifest = json.loads(output.with_suffix(".selection.json").read_text())
    assert manifest["selected_count"] == 1
    assert manifest["attempts_per_task"] == 4
    assert manifest["model_checkpoint"] == "checkpoint-sha"
    assert manifest["parquet_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["input_sha256"]["heldout"] == hashlib.sha256(files["heldout"].read_bytes()).hexdigest()


def test_candidates_command_requires_both_review_manifests(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "axion.datagen.stage2.pipeline.prepare_rl", "candidates"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "--heldout" in result.stderr and "--defects" in result.stderr


def test_parquet_metadata_is_accepted_after_pandas_nested_array_conversion(tmp_path):
    import pandas as pd
    from axion.training.rl.process_reward import score
    item = task()
    item["hops"] = [{"id": "entity", "description": "Identify enzyme", "match": "all",
                     "atoms": [{"type": "str", "value": "P12345", "grounded": True}]}]
    rows, report = select(attempts(), tasks=[item])
    output = tmp_path / "train.parquet"
    selection_module().write_dataset(rows, output, report, {})
    loaded = pd.read_parquet(output).iloc[0].to_dict()
    result = asyncio.run(score(messages=[{"role": "assistant", "content": "<answer>42</answer>"}],
                               reference=loaded["reward_model"]["ground_truth"],
                               hops=loaded["extra_info"]["hops"]))
    assert result["score"] == 1
    assert not result["drop_from_training"]
