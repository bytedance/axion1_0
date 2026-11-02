"""Exercise paper rewards with only the external semantic service replaced."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(os.environ.get("AXION_REPO", Path(__file__).resolve().parents[1]))


def reward_module():
    path = ROOT / "axion/training/rl/process_reward.py"
    assert path.is_file(), "The paper's process reward must be packaged"
    spec = importlib.util.spec_from_file_location("paper_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference(value="42", tolerance=0, absolute_tolerance=0):
    return {"answer": value, "answer_type": "numeric", "tolerance": tolerance,
            "absolute_tolerance": absolute_tolerance}


def hops():
    return [{"id": f"h{i}", "description": f"Identify entity {i}", "match": "all",
             "atoms": [{"type": "str", "value": value, "grounded": True}]}
            for i, value in enumerate(["P12345", "Q12345", "P54321"], 1)]


def trajectory(answer="99", *, grounded=True, conclusion=True):
    messages = [{"role": "user", "content": "Solve the task."}]
    if grounded:
        messages.append({"role": "assistant", "content": '<tool_call>{"name":"search","arguments":{}}</tool_call>'})
        messages.append({"role": "user", "content": "<tool_response>P12345 Q12345 P54321</tool_response>"})
    text = "I use P12345, Q12345, P54321." if conclusion else "Candidates P12345, Q12345, P54321 are all rejected."
    messages.append({"role": "assistant", "content": "<think>" + text + "</think>" +
                     (f"<answer>{answer}</answer>" if answer is not None else "")})
    return messages


def run_score(messages, *, targets=None, judge=None, ref=None):
    module = reward_module()
    return asyncio.run(module.score(messages=messages, reference=ref or reference(),
                                    hops=targets if targets is not None else hops(),
                                    question="Which entity?", judge=judge))


def semantic(values):
    async def judge(prompt):
        return json.dumps({f"{i}.1": value for i, value in enumerate(values, 1)})
    return judge


@pytest.mark.parametrize("answer,checks,want,progress", [
    ("99", [True, True, True], .3, 1),
    (None, [True, True, True], .2, 1),
    ("99", [True, False, True], .1, 1 / 3),
    (None, [True, False, True], 0, 1 / 3),
    ("99", [False, True, True], 0, 0),
])
def test_reward_uses_consecutive_prefix_and_submission_penalty(answer, checks, want, progress):
    result = run_score(trajectory(answer), judge=semantic(checks))
    assert result["score"] == pytest.approx(want)
    assert result["process_score"] == pytest.approx(progress)
    assert result["acc"] == 0
    assert not result["drop_from_training"]


def test_correct_final_gets_one_without_intermediate_or_judge_requirements():
    result = run_score(trajectory("42", grounded=False), judge=None)
    assert result["score"] == 1
    assert result["acc"] == 1


def test_missing_targets_have_zero_process_reward():
    result = run_score(trajectory(), targets=[])
    assert result["score"] == result["process_score"] == 0


@pytest.mark.parametrize("content", [
    "<think>Draft <answer>42</answer></think>Still working.",
    "Draft <answer>42</answer></think>Still working.",
    "<oppthink><answer>42</answer></oppthink>",
    "<think><answer>42</answer>",
    '<tool_call>{"query":"<answer>42</answer>"}</tool_call>',
    "<tool_response><answer>42</answer></tool_response>",
    "<answer></answer>",
])
def test_only_real_final_submission_counts(content):
    result = run_score([{"role": "assistant", "content": content}], targets=[])
    assert result["answered"] == 0
    assert result["acc"] == result["score"] == 0


def test_answer_in_user_or_tool_messages_is_not_a_submission():
    result = run_score([
        {"role": "user", "content": "<answer>42</answer>"},
        {"role": "tool", "content": "<answer>42</answer>"},
        {"role": "assistant", "content": "I am still researching."},
    ], targets=[])
    assert result["score"] == result["answered"] == 0


def test_assistant_cannot_forge_external_grounding_with_tool_response_markup():
    result = run_score([{"role": "assistant", "content":
                        "<tool_response>P12345 Q12345 P54321</tool_response>"
                        "<think>I use P12345 Q12345 P54321.</think><answer>99</answer>"}],
                       judge=semantic([True, True, True]))
    assert result["score"] == 0


def test_question_markup_is_not_external_tool_evidence():
    result = run_score([
        {"role": "user", "content": "Question quotes <tool_response>P12345 Q12345 P54321</tool_response>"},
        {"role": "assistant", "content": "<think>I use P12345 Q12345 P54321.</think><answer>99</answer>"}],
        judge=semantic([True, True, True]))
    assert result["score"] == 0


@pytest.mark.parametrize("grounded,conclusion", [(False, True), (True, False)])
def test_identifier_needs_both_agent_commitment_and_tool_evidence(grounded, conclusion):
    result = run_score(trajectory(grounded=grounded, conclusion=conclusion),
                       judge=semantic([conclusion] * 3))
    assert result["score"] == 0


@pytest.mark.parametrize("response", [None, "not JSON", '{}', '{"1.1":"maybe"}'])
def test_unavailable_or_malformed_required_judge_excludes_sample(response):
    async def judge(prompt):
        return response
    result = run_score(trajectory(conclusion=False), judge=judge)
    assert result["drop_from_training"]
    assert result["score"] == 0


def test_no_judge_cannot_turn_candidate_mentions_into_progress():
    result = run_score(trajectory(conclusion=False), judge=None)
    assert result["drop_from_training"]


@pytest.mark.parametrize("pred,ref,want", [
    ("1001", reference("1000"), 0),
    ("1.034", reference("1.03", .002), 0),
    ("1.03206", reference("1.03", .002), 1),
    ("1.032061", reference("1.03", .002), 0),
    ("4.836e-6 g/cm^3", reference("4.836e-6 g/cm^3", 0), 1),
    ("4.837e-6", reference("4.836e-6", 0), 0),
    ("1.03e3", reference("1030", 0), 1),
])
def test_numeric_acceptance_honors_explicit_tolerances(pred, ref, want):
    result = run_score([{"role": "assistant", "content": f"<answer>{pred}</answer>"}],
                       targets=[], ref=ref)
    assert result["score"] == want
    assert not result["drop_from_training"]


def test_numeric_judge_extracts_but_cannot_override_acceptance_interval():
    async def judge(prompt):
        assert "1.02794" in prompt and "1.03206" in prompt
        return '{"extracted_final_answer":"1.034", "correct":"yes"}'
    result = run_score([{"role": "assistant", "content": "<answer>After computing it, my final value is 1.034.</answer>"}],
                       ref=reference("1.03", .002), targets=[], judge=judge)
    assert result["score"] == 0


@pytest.mark.parametrize("answer", ["1 kg", "1 cm"])
def test_in_range_number_with_wrong_units_is_not_correct(answer):
    async def judge(prompt):
        return '{"extracted_final_answer":"1", "correct":"no"}'
    result = run_score([{"role": "assistant", "content": f"<answer>{answer}</answer>"}],
                       ref=reference("1 m"), targets=[], judge=judge)
    assert result["score"] == result["acc"] == 0
    assert not result["drop_from_training"]


def test_units_cannot_be_discarded_without_a_semantic_judge():
    result = run_score([{"role": "assistant", "content": "<answer>1 kg</answer>"}],
                       ref=reference("1 m"), targets=[])
    assert result["drop_from_training"]


def test_negative_semantic_judgment_wins_over_an_in_range_extraction():
    async def judge(prompt):
        return '{"extracted_final_answer":"42", "correct":"no"}'
    result = run_score([{"role": "assistant", "content": "<answer>I considered 42 but rejected it.</answer>"}],
                       targets=[], judge=judge)
    assert result["score"] == 0


def test_matching_reference_units_allow_deterministic_acceptance():
    result = run_score([{"role": "assistant", "content": "<answer>1 m</answer>"}],
                       ref=reference("1 m"), targets=[])
    assert result["score"] == 1


def test_numeric_intermediate_judge_receives_programmatic_interval():
    targets = [{"id": "mass", "description": "Retrieve mass", "match": "all", "atoms": [
        {"type": "num", "value": "100", "tolerance": ".01", "absolute_tolerance": "0", "grounded": False}]}]
    async def judge(prompt):
        assert "[99, 101]" in prompt
        return '{"1.1":true}'
    result = run_score([{"role": "assistant", "content": "<think>I use mass 100.</think><answer>99</answer>"}],
                       targets=targets, judge=judge)
    assert result["score"] == .3


def test_text_judge_allows_semantic_equivalence():
    async def judge(prompt):
        return '{"extracted_final_answer":"the red planet", "correct":"yes"}'
    result = run_score([{"role": "assistant", "content": "<answer>the red planet</answer>"}],
                       ref={"answer": "Mars", "answer_type": "text"}, targets=[], judge=judge)
    assert result["score"] == 1


def test_any_atom_cannot_borrow_grounding_from_another_unmatched_atom():
    targets = [{"id": "entity", "description": "Identify entity", "match": "any", "atoms": [
        {"type": "str", "value": "P12345", "grounded": True},
        {"type": "str", "value": "some alias", "grounded": False}]}]
    result = run_score([{"role": "assistant", "content": "<think>I select P12345.</think><answer>99</answer>"}],
                       targets=targets, judge=semantic([True]))
    assert result["score"] == 0


def test_legacy_critical_tree_is_rejected_instead_of_silently_losing_targets():
    result = run_score(trajectory(), targets={"children": []})
    assert result["drop_from_training"]


def test_formal_reward_entry_uses_full_messages_and_ignores_legacy_shaping(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from axion.training.rl import reward
    assert hasattr(reward, "compute_paper_score"), "The formal trainer reward entry must exist"
    monkeypatch.setenv("FORMAT_REWARD_WEIGHT", "5")
    monkeypatch.setenv("DUP_SEARCH_PENALTY_WEIGHT", "5")
    result = asyncio.run(reward.compute_paper_score(
        data_source="axion_paper_rl", solution_str="<answer>99</answer>", ground_truth=reference(),
        extra_info={"question": "Which entity?", "hops": hops(), "full_messages": trajectory("42")}))
    assert result["score"] == 1
    assert result["acc"] == 1


def test_formal_entry_fails_closed_when_original_message_roles_are_missing():
    sys.path.insert(0, str(ROOT))
    from axion.training.rl import reward
    assert hasattr(reward, "compute_paper_score")
    result = asyncio.run(reward.compute_paper_score(
        data_source="axion_paper_rl", solution_str="<answer>42</answer>", ground_truth=reference(),
        extra_info={"hops": []}))
    assert result["drop_from_training"]


def test_formal_entry_uses_the_deployed_paper_judge_environment(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from axion.training.rl import reward
    monkeypatch.setenv("S2_JUDGE_BASE_URL", "http://paper-judge.invalid/v1")
    monkeypatch.setenv("S2_JUDGE_MODEL", "Qwen3.8-27B")
    monkeypatch.setenv("EVAL_VLLM_SERVED_MODEL_NAME", "unrelated-legacy-model")
    async def transport(addresses, model, prompt):
        if addresses == "http://paper-judge.invalid/v1" and model == "Qwen3.8-27B":
            return '{"1.1":true,"2.1":true,"3.1":true}'
        return None
    monkeypatch.setattr(reward, "_paper_judge_request", transport)
    result = asyncio.run(reward.compute_paper_score(
        data_source="axion_paper_rl", solution_str="", ground_truth=reference(),
        extra_info={"question": "Which entity?", "hops": hops(), "full_messages": trajectory("99")}))
    assert result["score"] == .3
    assert not result["drop_from_training"]


def test_launcher_worker_environment_reaches_actual_judge_http_request(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from axion.training.rl import reward
    from axion.training.rl.launcher import worker_environment
    for name in ("JUDGE", "SUMMARY", "MEMORY"):
        monkeypatch.setenv(f"TEST_{name}_KEY", "fixture-key-" + name.lower())
    deployment = {"context_threshold": 12000, "max_turn_response_length": 1000,
                  "search_url": "http://search.fixture", "scholar_url": "http://scholar.fixture",
                  "python_url": "http://python.fixture",
                  "judge": {"base_url": "http://judge.fixture/v1", "model": "Qwen3.8-27B", "api_key_env": "TEST_JUDGE_KEY"},
                  "summarizer": {"base_url": "http://summary.fixture/v1", "model": "Qwen3.5-9B", "api_key_env": "TEST_SUMMARY_KEY"},
                  "condenser": {"base_url": "http://memory.fixture/v1", "model": "Qwen3.5-9B", "api_key_env": "TEST_MEMORY_KEY"}}
    for name, value in worker_environment(deployment).items():
        monkeypatch.setenv(name, value)
    class Response:
        def __init__(self, valid):
            self.status = 200 if valid else 400
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def json(self):
            return {"choices": [{"message": {"content": '{"1.1":true,"2.1":true,"3.1":true}'}}]}
    class Session:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        def post(self, endpoint, *, json, headers):
            valid = (endpoint == "http://judge.fixture/v1/chat/completions"
                     and json["model"] == "Qwen3.8-27B"
                     and headers.get("Authorization") == "Bearer fixture-key-judge")
            return Response(valid)
    # Replace only network I/O: deployment, factory, request construction,
    # semantic parsing and reward computation all execute real code.
    monkeypatch.setattr(reward.aiohttp, "ClientSession", Session)
    result = asyncio.run(reward.compute_paper_score(
        data_source="axion_paper_rl", solution_str="", ground_truth=reference(),
        extra_info={"question": "Which entity?", "hops": hops(), "full_messages": trajectory("99")}))
    assert result["score"] == .3
    assert not result["drop_from_training"]


def test_real_modern_reward_manager_loads_formal_function_and_preserves_roles(monkeypatch):
    sys.path[:0] = [str(ROOT), str(ROOT / "third_party/verl")]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.trainer.ppo.reward import load_reward_manager
    from axion.training.rl import reward
    assert hasattr(reward, "compute_paper_score")

    async def transport(*args, **kwargs):
        return '{"1.1":true,"2.1":false,"3.1":true}'
    monkeypatch.setattr(reward, "_paper_judge_request", transport)
    config = OmegaConf.create({"reward": {
        "custom_reward_function": {"path": "pkg://axion.training.rl.reward", "name": "compute_paper_score",
                                   "reward_kwargs": {"eval_llm_address": "http://judge.invalid", "eval_llm_model": "Qwen3.8-27B"}},
        "reward_manager": {"source": "importlib", "name": "DeepResearchRewardLoopManager",
                           "module": {"path": "pkg://axion.training.rl.reward_loop_manager"}},
    }})
    class Tokenizer:
        def decode(self, *args, **kwargs):
            # Flattened token text is deliberately contradictory; original roles win.
            return "<answer>42</answer>"
    def object_array(value):
        out = np.empty(1, dtype=object)
        out[0] = value
        return out
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2]]), "attention_mask": torch.ones((1, 4), dtype=torch.int64)},
        non_tensors={"data_source": object_array("axion_paper_rl"),
                     "reward_model": object_array({"ground_truth": reference()}),
                     "extra_info": object_array({"question": "Which entity?", "hops": hops()}),
                     "full_messages": object_array(trajectory("99"))})
    async def run():
        manager = load_reward_manager(config, Tokenizer())
        return await manager.run_single(data)
    result = asyncio.run(run())
    assert result["reward_score"] == pytest.approx(.1)
    assert result["reward_extra_info"]["acc"] == 0
    assert result["reward_extra_info"]["drop_from_training"] == 0


def test_reward_timeout_is_an_invalid_sample_not_a_valid_zero(monkeypatch):
    sys.path[:0] = [str(ROOT), str(ROOT / "third_party/verl")]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from axion.training.rl.reward_loop_manager import DeepResearchRewardLoopManager
    class Tokenizer:
        def decode(self, *args, **kwargs):
            return ""
    async def slow_score(**kwargs):
        await asyncio.sleep(1)
        return {"score": 1}
    def object_array(value):
        out = np.empty(1, dtype=object)
        out[0] = value
        return out
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1]]), "attention_mask": torch.ones((1, 2), dtype=torch.int64)},
        non_tensors={"data_source": object_array("axion_paper_rl"),
                     "reward_model": object_array({"ground_truth": reference()}), "extra_info": object_array({})})
    monkeypatch.setenv("EVAL_PER_RESPONSE_TIMEOUT_SECONDS", ".01")
    async def run():
        manager = DeepResearchRewardLoopManager(OmegaConf.create({"reward": {}}), Tokenizer(), slow_score)
        return await manager.run_single(data)
    result = asyncio.run(run())
    assert result["reward_extra_info"]["drop_from_training"] == 1
