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

"""Closed-book solvability filter.

Answers each question with an LLM and no tools, then matches deterministically
against the reference answer. Items the model can already guess are dropped:
they measure recall of pretraining rather than retrieval, and a model that
guesses right learns nothing from the trajectory.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.

Usage:
    python closedbook_filter.py --probe probe.jsonl --key key.jsonl \
        --out cb.jsonl --model gpt-5.5
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'utils'))
from llm_client import LLMClient, run_batch
from score import correct, final_answer

PROMPT = """Answer the following question from your own knowledge only. You have no tools, no web access, no code execution. If you must estimate, give your best single estimate.
Think as much as you need, then end with the final answer alone on the last line in the form <answer>...</answer> (a number, identifier, or short name; no units unless asked).

{q}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probe', required=True, help='question jsonl (no answers)')
    ap.add_argument('--key', required=True, help='reference-answer jsonl')
    ap.add_argument('--out', required=True, help='output jsonl, one row per item')
    ap.add_argument('--model', default=None, help='model name (defaults to $OPENAI_MODEL)')
    ap.add_argument('--qpm', type=int, default=60, help='queries per minute')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--passes', type=int, default=3, help='retry passes over failed items')
    ap.add_argument('--max-tokens', type=int, default=16000)
    args = ap.parse_args()

    probe = [json.loads(l) for l in open(args.probe, encoding='utf-8')]
    key = {}
    for l in open(args.key, encoding='utf-8'):
        r = json.loads(l)
        key[r['id']] = r

    cli = LLMClient(model=args.model, qpm=args.qpm)
    print(f'[closedbook] model = {cli.model}', flush=True)

    def work(r):
        text = cli.chat([{'role': 'user', 'content': PROMPT.format(q=r['question'])}],
                        max_tokens=args.max_tokens, temperature=1.0)
        pred = final_answer(text)
        k = key[r['id']]
        return {'pred': pred, 'ok': correct(k, pred),
                'walk': k.get('walk', ''), 'gt': str(k['ground_truth'])}

    res, failed = run_batch(probe, key_fn=lambda r: r['id'], work_fn=work,
                            checkpoint=args.out + '.ckpt.jsonl',
                            workers=args.workers, passes=args.passes)

    n, ok = collections.Counter(), collections.Counter()
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in probe:
            if r['id'] not in res:
                continue
            x = res[r['id']]
            n[x['walk']] += 1
            ok[x['walk']] += bool(x['ok'])
            f.write(json.dumps({'id': r['id'], **x}, ensure_ascii=False) + '\n')

    print('\nguessable closed-book, per source:')
    for w in sorted(n):
        print(f'  {w:6} {ok[w]:4d}/{n[w]:4d} = {100 * ok[w] / n[w]:5.1f}%')
    total_n, total_ok = sum(n.values()), sum(ok.values())
    if total_n:
        print(f'  {"total":6} {total_ok:4d}/{total_n:4d} = {100 * total_ok / total_n:5.1f}%'
              '   <- drop these items')
    print(f'failed {len(failed)}  {dict(cli.stats)}')


if __name__ == '__main__':
    main()
