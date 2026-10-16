# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rollout telemetry shared by the legacy (ray_trainer) and V1 (v1/trainer_base) trainers.

Lives in one module because it once lived only in the V1 class while the running stack
dispatched to the legacy one — every metric silently vanished (2026-09-07). Whatever trainer
runs, these are called from its _log_rollout_data.
"""

from __future__ import annotations

import collections
import json
import os
import re

_TOOL_CALL_RE = re.compile(r"<function=([A-Za-z0-9_\-]+)>")
_BW_WARNED = False


def group_spread_metrics(uids, scores) -> dict:
    """batch/same_score_group_ratio: fraction of items whose n rollouts all received the SAME
    score (zero GRPO advantage — no learning signal; rising ratio = curriculum exhausting).
    batch/effective_group_ratio is its complement. Padding rows excluded. Guarded — telemetry
    never kills a step."""
    try:
        groups = collections.defaultdict(list)
        for uid, sc in zip(uids, scores):
            u = str(uid)
            if u.startswith("pad"):
                continue
            groups[u.rsplit("_", 2)[0]].append(float(sc))
        if not groups:
            return {}
        same = sum(1 for v in groups.values() if max(v) - min(v) < 1e-6)
        return {
            "batch/same_score_group_ratio": same / len(groups),
            "batch/effective_group_ratio": 1.0 - same / len(groups),
            "batch/n_groups": len(groups),
        }
    except Exception as e:  # noqa: BLE001
        print(f"[group-spread] metrics skipped: {e!r}", flush=True)
        return {}


def tool_call_metrics(outputs) -> dict:
    """Per-tool call counts from decoded rollout outputs (qwen XML wire format: one
    `<function=NAME>` per attempted call; observations carry no such markup and the system
    prompt's worked example lives in the input). Per step: tools/calls/<name> (batch total),
    tools/traj_frac/<name> (fraction of trajectories with >=1 call), tools/calls_per_traj."""
    try:
        totals: dict[str, int] = {}
        traj_with: dict[str, int] = {}
        n_calls = 0
        n_traj = 0
        for out in outputs:
            if not out:
                continue
            n_traj += 1
            names = _TOOL_CALL_RE.findall(out)
            n_calls += len(names)
            for name in names:
                totals[name] = totals.get(name, 0) + 1
            for name in set(names):
                traj_with[name] = traj_with.get(name, 0) + 1
        if n_traj == 0:
            return {}
        m = {"tools/calls_per_traj": n_calls / n_traj}
        for name, cnt in sorted(totals.items()):
            m[f"tools/calls/{name}"] = cnt
            m[f"tools/traj_frac/{name}"] = traj_with.get(name, 0) / n_traj
        return m
    except Exception as e:  # noqa: BLE001
        print(f"[tool-calls] metrics skipped: {e!r}", flush=True)
        return {}


_TOOL_RESP_RE = re.compile(r"(<tool_response>\s*)(.*?)(\s*</tool_response>)", re.S)


def _squeeze_trajectory(text: str, obs_cap: int = 400, total_cap: int = 20000) -> str:
    """Selective per-turn truncation for review display: tool OBSERVATIONS (policy documents
    are ~4k chars each and dominate length) are squeezed to head+tail excerpts, while every
    MODEL turn — including the stage-3 conclusion at the very end — stays intact. The former
    flat head-cap of 6,000 chars cut nearly every trajectory's conclusion off. A generous
    total cap remains as a backstop and preserves the TAIL (where the verdict lives)."""
    def _squash(m):
        body = m.group(2)
        if len(body) <= obs_cap:
            return m.group(0)
        return (m.group(1) + body[: obs_cap // 2] +
                f" …[{len(body) - obs_cap} obs chars trimmed]… " +
                body[-obs_cap // 2:] + m.group(3))

    out = _TOOL_RESP_RE.sub(_squash, text)
    if len(out) > total_cap:
        keep_tail = 5000
        out = out[: total_cap - keep_tail] + "\n…[middle trimmed]…\n" + out[-keep_tail:]
    return out


def dump_best_worst(uids, inputs, outputs, scores, rollout_data_dir, global_steps) -> None:
    """One best- and one worst-scored trajectory per item (GRPO group), written to BOTH sinks:
    <rollout_data_dir>/best_worst/<step>.jsonl always (survives any wandb backend trouble),
    and a wandb table when a run is active (absence reported once, loudly). Enabled by the
    caller via VERL_WANDB_BEST_WORST=1."""
    global _BW_WARNED
    groups: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        groups.setdefault(str(u).rsplit("_", 2)[0], []).append(i)
    rows = []
    for sample, idxs in groups.items():
        b = max(idxs, key=lambda i: scores[i])
        w = min(idxs, key=lambda i: scores[i])
        rows.append({
            "step": global_steps, "item": sample,
            "best_score": round(float(scores[b]), 4), "worst_score": round(float(scores[w]), 4),
            "best_trajectory": _squeeze_trajectory(outputs[b]),
            "worst_trajectory": _squeeze_trajectory(outputs[w]),
            "prompt_tail": inputs[idxs[0]][-600:],
        })
    if rollout_data_dir:
        try:
            bw_dir = os.path.join(rollout_data_dir, "best_worst")
            os.makedirs(bw_dir, exist_ok=True)
            with open(os.path.join(bw_dir, f"{global_steps}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[best-worst] file dump failed: {e!r}", flush=True)
    try:
        import wandb

        if wandb.run is None:
            if not _BW_WARNED:
                _BW_WARNED = True
                print("[best-worst] wandb.run is None in the trainer process — table logging "
                      "disabled; per-step files still written under rollout_data_dir/best_worst/",
                      flush=True)
            return
        # Step-keyed tables every 10 steps (and step 1): logging one key per step made the
        # UI show only the latest table; distinct keys make each snapshot persistently
        # browsable (user directive 2026-09-08). The local per-step files still cover every
        # step for offline review.
        if global_steps % 10 != 0 and global_steps != 1:
            return
        table = wandb.Table(
            columns=["step", "item", "best_score", "worst_score",
                     "best_trajectory", "worst_trajectory", "prompt_tail"]
        )
        for r in rows:
            table.add_data(r["step"], r["item"], r["best_score"], r["worst_score"],
                           r["best_trajectory"], r["worst_trajectory"], r["prompt_tail"])
        wandb.log({f"rollout/best_worst/step_{global_steps:04d}": table}, step=global_steps)
    except Exception as e:  # noqa: BLE001
        print(f"[best-worst] wandb logging failed: {e!r}", flush=True)


def reward_term_metrics(reward_extra) -> dict:
    """Average the reward function's own terms: reward/<key>/mean.

    A reward that returns a bare float logs one number (critic/score) and hides everything the
    reward version actually does -- outcome TP/FP/TN/FN, the grounding gate, the stage-1 gate,
    invalid-call and economy penalties. Returning a dict from compute_score routes those terms
    to reward_extra_info; this turns them into per-step rates. 0/1 terms average into rates, so
    reward/fp/mean IS the batch false-positive rate.
    """
    rows = [r for r in (reward_extra or []) if isinstance(r, dict)]
    if not rows:
        return {}
    keys = set()
    for row in rows:
        keys.update(k for k, v in row.items() if isinstance(v, (int, float, bool)))
    out = {}
    for key in sorted(keys):
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
        if vals:
            out[f"reward/{key}/mean"] = sum(vals) / len(vals)
    return out


def rollout_step_metrics(uids, inputs, outputs, scores, rollout_data_dir, global_steps,
                         reward_extra=None) -> dict:
    """The one call a trainer's _log_rollout_data makes: best/worst dump (if enabled by env)
    plus the metric dicts, merged, for the caller to fold into its step metrics."""
    if os.environ.get("VERL_WANDB_BEST_WORST", "0") == "1":
        dump_best_worst(uids, inputs, outputs, scores, rollout_data_dir, global_steps)
    extras = {}
    extras.update(group_spread_metrics(uids, scores))
    extras.update(tool_call_metrics(outputs))
    extras.update(reward_term_metrics(reward_extra))
    return extras
