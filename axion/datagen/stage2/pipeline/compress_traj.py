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

"""Compress over-long trajectories back under a token cutoff.

Performs only transformations that leave the training target untouched. SFT
computes loss on assistant turns alone, so:

  A. collapse degenerate repetition (a tag or line repeated consecutively),
     which is pure generation noise
  B. truncate over-long user-side content (tool returns), which changes the
     context rather than the target
"""
import argparse
import collections
import json
import os
import re
import zlib

RATIO = 2.77
TOKEN = re.compile(r'[A-Za-z]{2,}\d[\w.\-]*|\d[\d,.]{2,}')
RUNTAG = re.compile(r'(</?(?:tool_call|answer|think|tool_response)>)(\s*\1){2,}')
ST = collections.Counter()

DEGEN_THR = 0.12

def is_degen(c, min_len=20000):
    """Detect degenerate output.

        Catches both forms: a single tag repeated, and runaway enumeration where
        every line differs because of an incrementing index. Uses zlib
        compressibility rather than counting repeated substrings, which would only
        find the first form.
    """
    if len(c) < min_len: return False
    b = c.encode()
    return len(zlib.compress(b, 6)) / len(b) < DEGEN_THR

def degen_onset(c, win=3000, step=500):
    """Locate the onset of degeneration: the first position whose following window is highly compressible."""
    for i in range(0, len(c) - win, step):
        seg = c[i:i+win].encode()
        if len(zlib.compress(seg, 6)) / len(seg) < DEGEN_THR:
            return i
    return None

def collapse(c):
    c = RUNTAG.sub(r'\1', c)
    out, prev, run = [], None, 0
    for ln in c.split('\n'):
        if ln == prev: run += 1
        else: prev, run = ln, 0
        if run < 2: out.append(ln)
    return '\n'.join(out)

def squeeze(c, budget, head_frac=0.7):
    """Keep the head and tail, remove the middle, and leave an explicit marker."""
    if len(c) <= budget: return c
    h = int(budget * head_frac); t = budget - h
    return (c[:h] + f"\n\n...[{len(c)-budget} characters omitted]...\n\n" + c[-t:])

def compress(ms, cutoff, user_budget):
    ms = [dict(m) for m in ms]
    for m in ms:
        c = m.get('content') or ''
        c2 = collapse(c)
        if c2 != c: ST['A::collapse_repeats'] += 1; ST['A::chars_saved'] += len(c) - len(c2)
        m['content'] = c2
    for m in ms:
        if m.get('role') != 'assistant': continue
        c = m.get('content') or ''
        if not is_degen(c): continue
        k = degen_onset(c)
        if k is None or k < 200:
            ST['A2::degenerate_no_clean_prefix'] += 1; continue
        ST['A2::cut_degenerate'] += 1; ST['A2::chars_saved'] += len(c) - k
        m['content'] = c[:k].rstrip()
    if sum(len(m.get('content') or '') for m in ms) / RATIO <= cutoff:
        return ms, 'A_sufficient'
    for i, m in enumerate(ms):
        if m.get('role') != 'user': continue
        c = m.get('content') or ''
        if len(c) <= user_budget: continue
        new = squeeze(c, user_budget)
        removed = c[int(user_budget*0.7): len(c)-(user_budget-int(user_budget*0.7))]
        later = ''.join((x.get('content') or '') for x in ms[i+1:] if x.get('role') == 'assistant')
        want = {t for t in TOKEN.findall(later) if t in removed}
        lost = {t for t in want if t not in new}
        if want and len(lost)/len(want) > 0.34:
            ST['B::safety_valve'] += 1; continue
        ST['B::truncated'] += 1; ST['B::chars_saved'] += len(c) - len(new)
        m['content'] = new
        if sum(len(x.get('content') or '') for x in ms) / RATIO <= cutoff:
            return ms, 'B_sufficient'
    return ms, 'still_over'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--cutoff', type=int, default=98000)
    ap.add_argument('--user-budget', type=int, default=8000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    n = rescued = still = untouched = 0
    for f in a.inputs:
        op = os.path.join(a.out, os.path.basename(f))
        with open(op, 'w') as w:
            for line in open(f):
                try: o = json.loads(line)
                except Exception: continue
                n += 1
                before = sum(len(m.get('content') or '') for m in o.get('messages', []))
                if before / RATIO <= a.cutoff:
                    untouched += 1; w.write(line); continue
                ms, how = compress(o['messages'], a.cutoff, a.user_budget)
                after = sum(len(m.get('content') or '') for m in ms)
                o['messages'] = ms
                if after / RATIO <= a.cutoff: rescued += 1
                else: still += 1
                w.write(json.dumps(o, ensure_ascii=False) + '\n')
        print(f"  {os.path.basename(f)} -> {op}")
    print(f"\ntotal {n} | already_ok {untouched} | compressed_ok {rescued} | still_over {still}")
    for k, v in sorted(ST.items()): print(f"  {k}: {v:,}")

if __name__ == '__main__': main()
