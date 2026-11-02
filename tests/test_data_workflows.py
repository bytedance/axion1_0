"""Regression checks for command paths and training-data semantics."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(os.environ.get("AXION_REPO", Path(__file__).resolve().parents[1]))


def run(entry, *args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, *entry, *map(str, args)], cwd=cwd, env=env,
        text=True, capture_output=True, timeout=30,
    )


def entries(path, module):
    return [[str(ROOT / path)], ["-m", module]]


@pytest.mark.parametrize("entry", entries(
    "datagen/stage2/pipeline/judge.py", "axion.datagen.stage2.pipeline.judge"
), ids=["legacy", "module"])
def test_scoring_preserves_per_item_results(entry, tmp_path):
    keys = [
        {"id": "a", "question": "Density?", "walk": "EXO", "ground_truth": "1.50", "answer_type": "numeric", "tolerance": 0.02},
        {"id": "b", "question": "Count?", "walk": "COUNT", "ground_truth": "42"},
        {"id": "c", "question": "Galaxy?", "walk": "NAME", "ground_truth": "NGC 1300"},
    ]
    key = tmp_path / "key.jsonl"
    key.write_text("".join(json.dumps(x) + "\n" for x in keys))
    rounds = []
    for index, answers in enumerate([["1.52", "43", "NGC 1300"], ["1.70", "42", "M31"]], 1):
        path = tmp_path / f"iter{index}.jsonl"
        path.write_text("".join(json.dumps({
            "id": item["id"], "question": item["question"],
            "prediction": f"<answer>{answer}</answer>", "num_rounds": 2,
        }) + "\n" for item, answer in zip(keys, answers)))
        rounds.append(path)
    output = tmp_path / "report.json"
    result = run(entry, "--key", key, "--out", output, *rounds, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == {
        "per_id": {"a": [True, False], "b": [False, True], "c": [True, False]},
        "all_wrong": [],
    }


@pytest.mark.parametrize("entry", entries(
    "datagen/utils/to_sft.py", "axion.datagen.utils.to_sft"
), ids=["legacy", "module"])
@pytest.mark.parametrize("tools,expected_count", [("PythonInterpreter", 1), ("PythonInterpreter,search", 2)])
def test_trajectory_conversion_preserves_messages_and_stage_tool_policy(entry, tools, expected_count, tmp_path):
    direct = [
        {"role": "system", "content": "Solve carefully."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "<think>Compute the sum.</think><answer>4</answer>"},
    ]
    retrieval = [
        {"role": "system", "content": "Use the available tools."},
        {"role": "user", "content": "Find the reference value."},
        {"role": "assistant", "content": '<think>Find a source.</think><tool_call>{"name":"search","arguments":{"query":"reference"}}</tool_call>'},
        {"role": "user", "content": "The source gives 7."},
        {"role": "assistant", "content": "<think>The source supports the value.</think><answer>7</answer>"},
    ]
    rows = [
        {"question": "What is 2 + 2?", "termination": "answer", "messages": direct},
        {"question": "Find the reference value.", "termination": "answer", "messages": retrieval},
        {"question": "What is 2 + 2?", "termination": "answer", "messages": direct},
        {"question": "Unfinished", "termination": "max_turns", "messages": direct},
        {"question": "Malformed", "termination": "answer", "messages": direct[:2] + [{"role": "assistant", "content": "<answer>4</answer>"}]},
    ]
    results = tmp_path / "rollouts"
    results.mkdir()
    (results / "iter1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "sft.jsonl"
    result = run(entry, "--results", results, "--out", output, "--tools", tools, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    expected = [{"messages": direct}, {"messages": retrieval}][:expected_count]
    assert [json.loads(line) for line in output.read_text().splitlines()] == expected


@pytest.mark.parametrize("path,module", [
    ("datagen/stage1/gen_from_seed.py", "axion.datagen.stage1.gen_from_seed"),
    ("datagen/stage1/gen_from_solution.py", "axion.datagen.stage1.gen_from_solution"),
    ("datagen/stage2/pipeline/paraphrase.py", "axion.datagen.stage2.pipeline.paraphrase"),
    ("datagen/stage2/pipeline/closedbook_filter.py", "axion.datagen.stage2.pipeline.closedbook_filter"),
    ("eval/frontierscience/judge.py", "axion.evaluation.frontierscience.judge"),
    ("eval/hle/judge.py", "axion.evaluation.hle.judge"),
    ("eval/hipho/judge.py", "axion.evaluation.hipho.judge"),
    ("eval/seal0/judge.py", "axion.evaluation.seal0.judge"),
])
@pytest.mark.parametrize("mode", ["legacy", "legacy_module", "module"])
def test_cli_help_loads_local_imports_and_prompt_resources(path, module, mode, tmp_path):
    entry = [str(ROOT / path)] if mode == "legacy" else ["-m", path[:-3].replace("/", ".") if mode == "legacy_module" else module]
    result = run(entry, "--help", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_package_import_does_not_load_training_frameworks(tmp_path):
    code = "import sys; import axion; assert not any(x in sys.modules for x in ['torch', 'swift', 'verl', 'llamafactory'])"
    result = run(["-c", code], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_packaged_benchmark_resources_are_readable(tmp_path):
    code = '''from importlib.resources import files
prompts = files("axion.evaluation.frontierscience").joinpath("prompts")
assert len(prompts.joinpath("fs_o_judge.txt").read_text()) > 100
assert len(prompts.joinpath("fs_r_judge.txt").read_text()) > 100
ids = files("axion.evaluation.hle").joinpath("subset_130_ids.txt").read_text().splitlines()
assert len({x for x in ids if x.strip()}) == 130
'''
    result = run(["-c", code], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
