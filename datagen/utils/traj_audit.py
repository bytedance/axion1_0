# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Normalise and audit tool calls inside sampled trajectories.

`normalize` canonicalises a raw <tool_call> payload and reports whether it is
well formed, so malformed calls can be rejected consistently rather than
rediscovered by every consumer.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st

TC = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
THINK = re.compile(r'<think>(.*?)</think>', re.S)
NAME = re.compile(r'"name"\s*:\s*"([A-Za-z_]+)"')


def brace_sig(s):
    return (s.count('{'), s.count('}'), s.count('['), s.count(']'))


def normalize(raw):
    """Canonicalise one raw tool-call payload.

        Returns (normalised_json_or_None, reason). A None result means the payload
        could not be parsed into a well-formed call.
    """
    m = re.search(r'<code>(.*?)</code>', raw, re.S)
    if m:
        json_part = raw[:m.start()].strip()
    else:
        o = re.search(r'<code>', raw)
        json_part = raw[:o.start()].strip() if o else raw.strip()
    try:
        tc = json.loads(json_part)
    except Exception:
        return False, None, json_part
    if isinstance(tc, dict):
        name, args = tc.get('name', ''), tc.get('arguments', {})
    else:
        name, args = tc, {}
    if isinstance(name, list) and len(name) == 1 and isinstance(name[0], str):
        name = name[0]
    if not isinstance(name, str) or not name.strip():
        return False, None, json_part
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False, name, json_part
    if isinstance(args, list):
        if not (len(args) == 1 and isinstance(args[0], dict)):
            return False, name, json_part
    return True, name.strip(), json_part


def audit_dir(d, judgment=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'iter*.jsonl'))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); r['_file'] = os.path.basename(f); rows.append(r)
    corr = {}
    if judgment and os.path.exists(judgment):
        for j in json.load(open(judgment)):
            k = (j['question'], j['file'] if j['file'].endswith('.jsonl') else j['file'] + '.jsonl')
            corr[k] = corr.get(k, False) or bool(j['is_correct'])

    n_tc = n_bad = 0
    sig = collections.Counter()
    tools = collections.Counter()
    think_lens, think_missing, asst_turns = [], 0, 0
    traj_bad = 0; traj_notool = 0; rounds = []
    per_traj_correct = []
    out_len = []
    for r in rows:
        msgs = r.get('messages') or []
        bad_here = False; tc_here = 0
        for m in msgs:
            if m.get('role') != 'assistant':
                continue
            c = m.get('content') or ''
            asst_turns += 1
            th = THINK.findall(c)
            if th:
                think_lens.append(len(th[-1]))
            else:
                think_missing += 1
            payloads = []
            if '<tool_call>' in c and '</tool_call>' in c:
                ms_ = TC.findall(c)
                if ms_:
                    payloads = [ms_[-1]]
            for payload in payloads:
                n_tc += 1; tc_here += 1
                ok, nm, json_part = normalize(payload)
                if ok:
                    tools[nm] += 1
                else:
                    n_bad += 1; bad_here = True
                    sig[brace_sig(json_part)] += 1
        traj_bad += bad_here
        traj_notool += (tc_here == 0)
        rounds.append(r.get('num_rounds', 0))
        out_len.append(sum(len(m.get('content') or '') for m in msgs if m.get('role') == 'assistant'))
        k = (r.get('question'), r['_file'])
        if k in corr: per_traj_correct.append((corr[k], bad_here, tc_here))

    N = max(1, len(rows))
    print(f"trajectories {len(rows)}   assistant turns {asst_turns}   median turns {st.median(rounds) if rounds else 0:.0f}   median output chars {st.median(out_len) if out_len else 0:.0f}")
    print(f"\n[tool correctness] tool_call total {n_tc}   malformed {n_bad} = {100*n_bad/max(1,n_tc):.1f}%   trajectories with any {traj_bad}/{len(rows)} = {100*traj_bad/max(1,len(rows)):.1f}%")
    if sig:
        print("  brace signature of malformed payloads ({,},[,]):")
        for s, c in sig.most_common(6):
            tag = 'missing outer }' if s == (2, 1, 1, 1) else ('extra }' if s == (2, 3, 1, 1) else '')
            print(f"    {s}  {c:5}  {100*c/n_bad:5.1f}%  {tag}")
    print(f"\n[think] assistant turns with think {asst_turns-think_missing}/{asst_turns} = {100*(asst_turns-think_missing)/max(1,asst_turns):.1f}%")
    if think_lens:
        q = lambda p: sorted(think_lens)[int(len(think_lens)*p)]
        print(f"  think length (chars) median {q(.5)}  p10 {q(.1)}  p90 {q(.9)}  max {max(think_lens)}")
    print(f"\n[tool usage] {dict(tools)}   per trajectory {n_tc/N:.1f}   trajectories with no tool {traj_notool}/{len(rows)} = {100*traj_notool/N:.1f}%")
    if per_traj_correct:
        ok = [c for c, b, t in per_traj_correct]
        print(f"\n[accuracy] {100*sum(ok)/len(ok):.1f}%  (n={len(ok)})")
        clean = [c for c, b, t in per_traj_correct if not b]
        if clean: print(f"  well-formed trajectories only {100*sum(clean)/len(clean):.1f}%  (n={len(clean)})  <- the admissible set")
        withtool = [c for c, b, t in per_traj_correct if t > 0]
        notool = [c for c, b, t in per_traj_correct if t == 0]
        if withtool and notool:
            print(f"  with tools {100*sum(withtool)/len(withtool):.1f}% (n={len(withtool)})   without {100*sum(notool)/len(notool):.1f}% (n={len(notool)})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--judgment', default=None)
    a = ap.parse_args()
    if a.judgment is None:
        c = os.path.join(a.dir, 'judgment_results.json')
        a.judgment = c if os.path.exists(c) else None
    audit_dir(a.dir, a.judgment)
