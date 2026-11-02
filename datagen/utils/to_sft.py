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

"""Convert sampled trajectories (harness iter*.jsonl) into sharegpt `messages`.

The whole ReAct trajectory is kept as it is, including the per-turn <think>
blocks and tool calls, so the trained model learns the same multi-turn
behaviour the advanced model produced.

Admission criteria (an item is dropped unless all hold; every rejection reason
is counted and printed):

  1. judged correct
  2. termination == 'answer', i.e. the trajectory ended on its own rather than
     being forced to answer by a turn or context limit
  3. no malformed tool_call: items the harness rejected at run time, or that
     fail normalisation, are dropped rather than repaired, because the stored
     assistant text would still carry the bad format the student would learn
  4. every assistant turn has a matching <think>...</think> pair
  5. the final assistant turn contains <answer>...</answer> and the trajectory
     has no server error
  6. deduplication on question text

The system prompt is restored to the evaluation-time version by stripping the
sampling-only formatting blocks, so the training and evaluation
distributions agree. Pass --keep-sampling-prompt to leave them in place.

Usage:
  python to_sft.py --results <dir with iter*.jsonl> [--judgment judgment_results.json]
                   --out <output.jsonl> [--dataset <question jsonl, for subject labels>]
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_audit import TC, normalize

THINK = re.compile(r'<think>(.*?)</think>', re.S)
ANS = re.compile(r'<answer>(.*?)</answer>', re.S)
REJECT_MARKERS = ('[Tool Error] Your tool_call', '[Tool Error] Invalid tool call format',
                  '[Tool Error] Invalid tool arguments')
# Tools a trajectory is allowed to have called. This has to match the harness
# the rollouts came from: Stage I is solved with Python and no retrieval, Stage
# II with the full set. A trajectory that called anything outside the list is
# rejected, which is what catches a harness configured for the wrong stage.
DEFAULT_TOOLS = ('PythonInterpreter', 'search', 'visit', 'google_scholar')


def sampling_prompt_blocks(paths):
    """Load system-prompt segments that were added only for the sampling run.

    The trained model sees the standard prompt at evaluation time, so any
    instruction added only while sampling has to be removed here, or the
    training and evaluation distributions disagree. Pass one --strip-block per
    file, each holding the verbatim text of one such segment.
    """
    blocks = []
    for path in paths or []:
        with open(path, encoding='utf-8') as f:
            b = f.read().strip()
        if b:
            blocks.append(b)
    return blocks


def load_judgment(path):
    corr = {}
    if not path or not os.path.exists(path):
        return corr
    for j in json.load(open(path)):
        f = j['file'] if j['file'].endswith('.jsonl') else j['file'] + '.jsonl'
        k = (j['question'], f)
        corr[k] = corr.get(k, False) or bool(j['is_correct'])
    return corr



TCRE = re.compile(r'<tool_call>.*?</tool_call>', re.S)


def fix_degenerate(c):
    """Repair two kinds of degeneration in assistant turns.

        1. Stray `</tool_call>` tags: after one well-formed call the model can emit
           thousands of bare closing tags. These are removed wherever they are not
           paired with an opening tag.
        2. Several tool calls in one turn: the harness only executes the last
           match, so collapsing to it is lossless and makes the stored text agree
           with what was actually run.

        Both have to be repaired rather than filtered on length, because if the
        stray tags reach SFT the student learns to emit them, and because length
        here is a consequence of the number of tool calls rather than of the
        degeneration.

        Returns (repaired_content, number_of_tags_removed).
    """
    k = 0
    o, cl = c.count('<tool_call>'), c.count('</tool_call>')
    if cl > o:
        holes = []
        c2 = TCRE.sub(lambda m: holes.append(m.group(0)) or f'\x00{len(holes)-1}\x00', c)
        c2 = c2.replace('</tool_call>', '')
        c = re.sub(r'\x00(\d+)\x00', lambda m: holes[int(m.group(1))], c2)
        k += cl - o
    if len(TCRE.findall(c)) > 1:
        last = TCRE.findall(c)[-1]
        th = THINK.search(c)
        if th:
            inner = TCRE.sub('', th.group(1))
            tail = TCRE.sub('', c[th.end():]).strip()
            c = ('<think>\n' + inner.strip() + '\n</think>' +
                 ('\n\n' + tail if tail else '') + '\n\n' + last)
        else:
            c = TCRE.sub('', c).strip() + '\n\n' + last
        k += 1
    return c, k


def merge_consecutive(ms):
    """Merge consecutive same-role messages.

        A turn that contains only <think> plus an answer without <answer> tags
        makes the harness ask again without inserting a user turn, producing two
        assistant messages in a row. They are merged into one: both <think> blocks
        and the first body become a single <think>, while the second body (carrying
        <answer> or <tool_call>) stays outside it. Consecutive user messages, which
        come from parallel tool returns, are concatenated.

        Returns (new_messages, merge_count).
    """
    out = []; n = 0
    for m in ms:
        m = {'role': m.get('role'), 'content': m.get('content') or ''}
        if m['role'] == 'assistant':
            m['content'], _k = fix_degenerate(m['content'])
        if out and out[-1]['role'] == m['role'] and m['role'] in ('assistant', 'user'):
            a, b = out[-1]['content'], m['content']
            if m['role'] == 'user':
                out[-1]['content'] = a.rstrip() + '\n\n' + b.lstrip(); n += 1; continue
            ta = THINK.findall(a); tb = THINK.findall(b)
            ra = THINK.sub('', a).strip(); rb = THINK.sub('', b)
            if '<tool_call>' in ra or '<answer>' in ra:
                out.append(m); continue
            if len(ta) != 1 or len(tb) != 1:
                _j = a.rstrip() + '\n' + b.lstrip()
                _thinks = [t.strip() for t in THINK.findall(_j) if t.strip()]
                _rest = THINK.sub('\n', _j)
                _mm = re.search(r'<tool_call>|<answer>', _rest)
                if _mm:
                    _pre, _tail = _rest[:_mm.start()], _rest[_mm.start():]
                else:
                    _pre, _tail = _rest, ''
                if _pre.strip():
                    _thinks.append(_pre.strip())
                if not _thinks:
                    out.append(m); continue
                out[-1]['content'] = ('<think>\n' + '\n\n'.join(_thinks) +
                                      '\n</think>\n\n' + _tail.lstrip())
                n += 1; continue
            think = ta[0].strip() + ('\n\n' + ra if ra else '') + '\n\n' + tb[0].strip()
            out[-1]['content'] = '<think>\n' + think + '\n</think>\n\n' + rb.lstrip(); n += 1
        else:
            out.append(m)
    return out, n


def check(r, corr, fname, allowed):
    """Return None when a trajectory is admissible, otherwise the rejection reason."""
    k = (r.get('question'), fname)
    if corr and not corr.get(k, False):
        return 'judged incorrect or unjudged'
    if r.get('termination') != 'answer':
        return f"termination={r.get('termination')}"
    ms = r.get('messages') or []
    roles = [m.get('role') for m in ms]
    if len(ms) < 3 or roles[0] != 'system' or roles[1] != 'user' or roles[-1] != 'assistant':
        return 'structure is not system/user/.../assistant'
    for i in range(1, len(roles) - 1):
        if roles[i] == roles[i + 1]:
            return 'roles do not alternate'
    last = ms[-1].get('content') or ''
    if not ANS.search(last):
        return 'final turn has no <answer>'
    for m in ms:
        c = m.get('content') or ''
        if 'vllm server error' in c:
            return 'contains a server error'
        if m.get('role') == 'user' and any(t in c for t in REJECT_MARKERS):
            return 'tool_call rejected by the harness'
        if m.get('role') == 'assistant':
            th = THINK.findall(c)
            if len(th) != 1 or c.count('<think>') != 1 or c.count('</think>') != 1:
                return 'assistant turn has no matching <think>'
            if c.count('<tool_call>') != c.count('</tool_call>'):
                return 'unbalanced tool_call tags'
            if '<tool_call>' in c:
                if '</tool_call>' not in c:
                    return 'unclosed tool_call'
                ok, nm, _ = normalize(TC.findall(c)[-1])
                if not ok:
                    return 'tool_call fails normalisation'
                if nm not in allowed:
                    return f'unknown tool ({nm})'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--judgment', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dataset', default=None, help='question jsonl, used to attach subject labels')
    ap.add_argument('--tools', default=','.join(DEFAULT_TOOLS),
                    help='comma-separated tools the rollouts were allowed to call; '
                         'use PythonInterpreter alone for Stage I')
    ap.add_argument('--keep-sampling-prompt', action='store_true',
                    help='leave sampling-only prompt blocks in place')
    ap.add_argument('--strip-block', action='append', default=[], metavar='FILE',
                    help='file holding a sampling-only system-prompt segment to '
                         'remove; repeat for several')
    a = ap.parse_args()
    jpath = a.judgment or os.path.join(a.results, 'judgment_results.json')
    corr = load_judgment(jpath)
    allowed = {x.strip() for x in a.tools.split(',') if x.strip()}
    if not allowed:
        raise SystemExit('--tools must name at least one tool')
    print(f'[to_sft] allowed tools: {sorted(allowed)}', flush=True)

    blks = [] if a.keep_sampling_prompt else sampling_prompt_blocks(a.strip_block)
    subj = {}
    if a.dataset and os.path.exists(a.dataset):
        for l in open(a.dataset):
            if l.strip():
                d = json.loads(l); subj[d['question'][:400]] = (d.get('meta') or {}).get('subject')

    stat = collections.Counter(); seen = set(); out = []
    chars = []; asst_chars = []; nturn = []
    for f in sorted(glob.glob(os.path.join(a.results, 'iter*.jsonl'))):
        fname = os.path.basename(f)
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l); stat['total'] += 1
            r['messages'], nm = merge_consecutive(r.get('messages') or [])
            if nm: stat['merged_same_role'] += 1
            why = check(r, corr, fname, allowed)
            if why:
                stat['dropped:' + why] += 1; continue
            q = (r.get('question') or '').strip()
            if q[:400] in seen:
                stat['dropped:duplicate question'] += 1; continue
            seen.add(q[:400])
            ms = [{'role': m['role'], 'content': m['content']} for m in r['messages']]
            for _b in blks:
                if _b in ms[0]['content']:
                    ms[0]['content'] = ms[0]['content'].replace(_b, '')
            out.append({'messages': ms, '_subject': subj.get(q[:400])})
            chars.append(sum(len(m['content']) for m in ms))
            asst_chars.append(sum(len(m['content']) for m in ms if m['role'] == 'assistant'))
            nturn.append(sum(1 for m in ms if m['role'] == 'assistant'))
            stat['kept'] += 1

    with open(a.out, 'w') as fo:
        for x in out:
            fo.write(json.dumps({'messages': x['messages']}, ensure_ascii=False) + '\n')
    print(f"kept {stat['kept']} / total {stat['total']}  -> {a.out}")
    for k, v in sorted(stat.items()):
        if k.startswith('dropped:'):
            print(f"  {k:32} {v}")
    if out:
        s = sorted(chars); m = len(s) // 2
        print(f"whole-trajectory chars: median {s[m]}  p90 {s[int(len(s)*.9)]}  max {max(s)}   assistant share {100*sum(asst_chars)/sum(chars):.1f}%   assistant turns median {sorted(nturn)[m]}")
        sub = collections.Counter(x['_subject'] for x in out)
        print("subjects:", dict(sub.most_common()))
        print(f"system prompt restored to the evaluation version: {not a.keep_sampling_prompt and bool(blks)}"
              f" ({len(blks)} sampling-only blocks stripped)")


if __name__ == '__main__':
    main()
