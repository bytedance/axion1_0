# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Repair sampled trajectories whose reasoning is sound but whose form is flawed.

Most of the loss when filtering rollouts comes from formatting rather
than from wrong answers: unclosed tool-call tags, consecutive same-role
messages, and tool failures that the model recovered from on the next turn.
This pass rescues those trajectories instead of discarding them.

Repairs are conservative: a trajectory is only rewritten when the fix provably
does not change the reasoning. Anything still invalid afterwards is dropped.

Usage:
    python repair_traj.py iter1.jsonl --out <dir>
"""
import argparse
import json
import os
import re
from collections import Counter

REMOVABLE = re.compile(r"(query cannot be empty|URL cannot be empty|Empty code\.|"
                       r"tool_call is not valid JSON|Unknown tool|is a required property)")
KEEP = re.compile(r"(TimeoutError|Execution timed out|\[Search Error\]|\[Scholar Error\])")
ANY_ERR = re.compile(r"\[(Tool|Visit|Search|Scholar|Python Interpreter) Error\]")
DSML_ERR = re.compile(r'(SyntaxError|Traceback)', re.S)
DSML_TAG = re.compile(r'<(?:｜DSML｜\s*)?(?:parameter|invoke)\s+name=')

def is_dsml_dead(user_content):
    return bool(DSML_ERR.search(user_content) and DSML_TAG.search(user_content))
REF = re.compile(r"(I made (a|an)[^.]{0,40}error|formatting error|requires a URL|"
                 r"forgot|was empty|is empty|failed|error|failure)", re.I)
NOTE = re.compile(
    r"NOTE: the summarization service was temporarily unavailable \(rate limited\), so the RAW "
    r"page text is given below instead of an extracted summary\. The page itself was fetched "
    r"successfully - do NOT discard this URL\. Read the text yourself\."
    r"(?: \(truncated to the first \d+ of \d+ characters\))?")


def think_counts(msgs):
    return [( (m.get("content") or "").count("<think>"), (m.get("content") or "").count("</think>") )
            for m in msgs if m.get("role") == "assistant"]


def validate(msgs):
    """Hard assertions: strict role alternation, system first, assistant last, no empty content.

        Returns None when the trajectory passes, otherwise the reason it failed.
    """
    if not msgs:
        return "empty messages"
    roles = [m.get("role") for m in msgs]
    if roles[0] != "system":
        return "first message is not system"
    if roles[-1] != "assistant":
        return "last message is not assistant"
    for a, b in zip(roles, roles[1:]):
        if a == b:
            return f"consecutive same role {a}"
    for m in msgs:
        if not (m.get("content") or "").strip():
            return "empty content"
    return None


_TYPO_CLOSE = re.compile(r'</(?:call_call|call|tool_calls|toolcall|tool_result)>')

def fix_typo_close(c):
    """Close a mistyped tool-call tag, but only where a tool call is actually left open.

        Occurrences of the same literal inside prose are left untouched.
    """
    res, i, n = c, 0, 0
    while True:
        op = re.search(r'<tool_call>', res[i:])
        if not op: break
        s = i + op.end()
        real = res.find('</tool_call>', s)
        ty = _TYPO_CLOSE.search(res, s, real if real >= 0 else len(res))
        if ty:
            res = res[:ty.start()] + '</tool_call>' + res[ty.end():]
            n += 1; i = ty.start() + len('</tool_call>')
        else:
            i = real + len('</tool_call>') if real >= 0 else len(res)
    return res, n

def repair(rec, aggressive=False, stats=None):
    """Return (new_messages, actions), or (None, reason) when the trajectory cannot be repaired."""
    ms = [dict(m) for m in (rec.get("messages") or [])]
    acts = []

    for m in ms:
        if m.get("role") != "assistant": continue
        c = m.get("content") or ""
        c2, k = fix_typo_close(c)
        if k:
            m["content"] = c2
            acts.append("fix_typo_close")


    drop = set()
    i = 0
    while i < len(ms):
        m = ms[i]
        c = m.get("content") or ""
        if m.get("role") != "assistant" and ANY_ERR.search(c):
            if KEEP.search(c) or not (REMOVABLE.search(c) or is_dsml_dead(c)):
                i += 1
                continue
            nxt = ms[i + 1] if i + 1 < len(ms) else None
            if nxt is not None and nxt.get("role") == "assistant":
                t = nxt.get("content") or ""
                th = re.search(r"<think>(.*?)</think>", t, re.S)
                if REF.search(th.group(1) if th else t[:600]):
                    if not aggressive:
                        return None, "failure referenced by the next turn"
                    if th:
                        inner = th.group(1)
                        lines = [ln for ln in inner.split("\n")
                                 if not REF.search(ln) or len(ln.strip()) > 120]
                        nxt["content"] = t.replace(th.group(0),
                                                   "<think>" + "\n".join(lines) + "</think>")
                        acts.append("strip_ref")
            if i >= 1 and ms[i - 1].get("role") == "assistant":
                drop.add(i - 1)
            drop.add(i)
            acts.append("drop_dead_turn")
        i += 1
    if drop:
        ms = [m for j, m in enumerate(ms) if j not in drop]

    merged = []
    for m in ms:
        if merged and merged[-1].get("role") == m.get("role"):
            merged[-1] = dict(merged[-1])
            merged[-1]["content"] = (merged[-1].get("content") or "") + "\n\n" + (m.get("content") or "")
            acts.append("merge_consecutive")
        else:
            merged.append(m)
    ms = merged

    why = validate(ms)
    if why:
        return None, f"still invalid after repair: {why}"
    return ms, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--aggressive", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tot = Counter()
    for src in a.inputs:
        rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        out_rows = []
        for rec in rows:
            tot["in"] += 1
            before_ok = validate(rec.get("messages") or []) is None
            _body = "".join((m.get("content") or "") for m in (rec.get("messages") or [])
                            if m.get("role") != "assistant")
            before_err = bool(ANY_ERR.search(_body))
            before_note = bool(NOTE.search(_body))
            if before_ok and not before_err and not before_note:
                tot["already_clean"] += 1
                out_rows.append(rec)
                continue
            th_before = think_counts(rec.get("messages") or [])
            new_ms, info = repair(rec, a.aggressive, tot)
            if new_ms is None:
                tot["unrepairable"] += 1
                tot[f"why::{info}"] += 1
                out_rows.append(rec)
                continue
            if think_counts(new_ms) and sum(map(sum, think_counts(new_ms))) > sum(map(sum, th_before)):
                tot["reject_think"] += 1
                out_rows.append(rec)
                continue
            new = dict(rec)
            new["messages"] = new_ms
            for k in ("prediction", "answer", "question"):
                assert new.get(k) == rec.get(k), f"{k} was modified"
            new["_repaired"] = sorted(set(info))
            tot["repaired"] += 1
            for x in set(info):
                tot[f"act::{x}"] += 1
            out_rows.append(new)
        dst = os.path.join(a.out, os.path.basename(src))
        with open(dst, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {os.path.basename(src)}: {len(rows)} rows -> {dst}")
    print("\nstats:")
    for k in sorted(tot):
        print(f"  {k:<48} {tot[k]}")


if __name__ == "__main__":
    main()
