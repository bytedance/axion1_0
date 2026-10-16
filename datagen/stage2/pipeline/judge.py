#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Deterministic scoring of harness output against the reference answers.

Because a reference answer is produced by construction rather than by a model,
scoring needs no judge model: the comparison in score.py is exact, with a
tolerance for numeric items.

Reports per source and overall:

    pass@1    mean accuracy across the rollouts of each item
    best@k    fraction of items solved by at least one rollout
    rounds    mean number of harness turns

Items that every rollout got wrong are written to the report, since those are
either genuinely hard or defective and worth inspecting.

Usage:
    python judge.py --key key.jsonl --out report.json iter1.jsonl [iter2.jsonl ...]
"""
import argparse
import collections
import json

from score import correct, final_answer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rollouts', nargs='+', help='harness output jsonl (one per rollout)')
    ap.add_argument('--key', required=True, help='reference-answer jsonl')
    ap.add_argument('--out', default='judge_report.json', help='where to write the report')
    ap.add_argument('--probe', default=None,
                    help='question jsonl, used to match rollouts that carry no id')
    args = ap.parse_args()

    key = {}
    for l in open(args.key, encoding='utf-8'):
        r = json.loads(l)
        key[r['id']] = r

    # Rollouts produced before ids were stable are matched on question text.
    by_question = {v['question']: v for v in key.values() if 'question' in v}
    if args.probe:
        for l in open(args.probe, encoding='utf-8'):
            d = json.loads(l)
            if d['id'] in key:
                by_question[d['question']] = key[d['id']]

    outcomes = collections.defaultdict(list)
    rounds = collections.defaultdict(list)
    for f in args.rollouts:
        for l in open(f, encoding='utf-8'):
            rec = json.loads(l)
            if 'question' not in rec:
                continue
            k = key.get(rec.get('id')) or by_question.get(rec['question'])
            if not k:
                continue
            pred = final_answer(rec.get('prediction') or '')
            outcomes[k['id']].append(correct(k, pred))
            rounds[k.get('walk', '')].append(rec.get('num_rounds', 0))

    if not outcomes:
        raise SystemExit('no rollout matched the key file')

    by_walk = collections.defaultdict(list)
    for item_id, oks in outcomes.items():
        by_walk[key[item_id].get('walk', '')].append(oks)

    print(f"{'source':8} {'n':>5} {'pass@1':>8} {'best@k':>8} {'rounds':>8}")
    n_all = p1_all = bk_all = 0
    for w in sorted(by_walk):
        v = by_walk[w]
        p1 = sum(sum(x) / len(x) for x in v)
        bk = sum(any(x) for x in v)
        mean_rounds = sum(rounds[w]) / max(len(rounds[w]), 1)
        print(f'{w:8} {len(v):5d} {p1 / len(v):8.3f} {bk / len(v):8.3f} {mean_rounds:8.1f}')
        n_all += len(v)
        p1_all += p1
        bk_all += bk
    print(f'{"ALL":8} {n_all:5d} {p1_all / n_all:8.3f} {bk_all / n_all:8.3f}')

    all_wrong = [i for i, v in outcomes.items() if not any(v)]
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'per_id': dict(outcomes), 'all_wrong': all_wrong}, f, indent=1)
    print(f'\n{len(all_wrong)} items wrong in every rollout -> {args.out}')


if __name__ == '__main__':
    main()
