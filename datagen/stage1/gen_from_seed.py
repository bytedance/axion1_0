#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Turn seed problems into harder questions with audited reference answers.

For each seed, three prompts run in sequence:

    GEN     name the mathematical object the seed is about, then ask a deeper
            question about that same object whose solution has to pass through
            the seed's answer
    SOLVE   write out the complete derivation and a single final answer
    AUDIT   re-read that derivation without being shown the answer, and say
            whether it holds

Only an item the audit returns CORRECT on keeps its answer; UNSURE and
unparseable verdicts are rejected rather than kept.

A seed is material only: it never appears in the output.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.

Usage:
    python gen_from_seed.py --seed_bench seeds.jsonl --n 500 --out bank.jsonl \
        --model gpt-5.5
"""
import argparse
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts

GEN="""You are constructing a HARDER research-level exam question from a seed problem.

SEED PROBLEM:
{q}

SEED ANSWER: {a}

Identify the underlying MATHEMATICAL OBJECT in the seed, then pose a NEW question about that SAME object whose solution path must pass through the seed's answer as a load-bearing intermediate step.

RULES:
- Go DEEPER INTO THE OBJECT. Do NOT do arithmetic on the seed's numeric answer.
  BAD : "let N be the seed answer; find the sum of N's prime factors"
  GOOD: seed gives the Galois group; new question asks the degrees of all normal
        subextensions (group -> normal subgroup lattice -> indices)
- SELF-CONTAINED. No "look up", no external table or paper. All data stated.
- Do NOT mention the seed problem or its answer.
- THE FINAL ANSWER MUST BE OF THIS FORM: {form}
  Make it a SINGLE unambiguous object of that form. If a tuple, state the component
  order explicitly in the question.
- The question must be well-posed: every object you name must exist and be unique.

Output EXACTLY:
OBJECT: <one line>
QUESTION:
<the new question, self-contained>
SEED_QUANTITY: <a one-line NAME (no value) of the quantity in the new question's
derivation that equals the seed answer>
ANSWER_FORM: {form}"""

# Answer forms the new question may demand, with the mix used in our run.
# Every form is machine-checkable, which is what lets the audit be decisive.
FORMS = ['a tuple of 2-4 components',
         'a closed-form expression in the stated parameters',
         'a single exact fraction or rational number',
         'a finite set, listed in increasing order',
         'a single exact integer']
WEIGHTS = [.35, .30, .15, .10, .10]


def field(text, key, next_key):
    """Pull the block between two labels out of a structured model reply."""
    m = re.search(rf'{key}:\s*(.*?)\s*(?:{next_key}:|$)', text or '', re.S)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed_bench', required=True,
                    help='jsonl of seeds, each {"question": ..., "answer": ...}')
    ap.add_argument('--out', required=True, help='output jsonl')
    ap.add_argument('--n', type=int, default=200, help='how many seeds to use')
    ap.add_argument('--model', default=None, help='model name (defaults to $OPENAI_MODEL)')
    ap.add_argument('--seed', type=int, default=77,
                    help='sampling seed; change it when generating another batch, '
                         'or the same seeds are drawn again')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--qpm', type=int, default=60, help='queries per minute')
    ap.add_argument('--gen-max-tokens', type=int, default=6000)
    ap.add_argument('--solve-max-tokens', type=int, default=16000)
    ap.add_argument('--audit-max-tokens', type=int, default=8000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cli = prompts.client(args.model, qpm=args.qpm)

    seeds = []
    for line in open(args.seed_bench, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        q = r.get('question') or r.get('problem')
        a = r.get('answer')
        if q and a is not None:
            seeds.append({'question': q, 'answer': a})
    if not seeds:
        raise SystemExit(f'no usable seeds in {args.seed_bench} '
                         '(each line needs "question" and "answer")')
    rng.shuffle(seeds)
    picked = seeds[:args.n]
    print(f'[1/3] generating from {len(picked)} seeds', flush=True)

    def gen_one(i):
        form = rng.choices(FORMS, WEIGHTS)[0]
        text = cli.chat([{'role': 'user', 'content': GEN.format(
            q=picked[i]['question'][:3000], a=str(picked[i]['answer'])[:300], form=form)}],
            max_tokens=args.gen_max_tokens, temperature=1.0, tag=f'gen/{i}')
        return i, form, text

    generated = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(gen_one, i) for i in range(len(picked))]):
            try:
                i, form, text = fu.result()
            except Exception as e:
                print(f'   gen failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            generated[i] = (form, text)
            done += 1
            if done % 40 == 0:
                print(f'   {done}/{len(picked)}', flush=True)

    items = []
    for i, (form, text) in generated.items():
        q = field(text, 'QUESTION', 'SEED_QUANTITY')
        sq = field(text, 'SEED_QUANTITY', 'ANSWER_FORM')
        if q and sq:
            items.append({'seed_index': i, 'form': form, 'question': q, 'seed_quantity': sq})
    print(f'[2/3] {len(items)} well-formed, solving each once', flush=True)

    def solve_one(k):
        text = cli.chat([{'role': 'user', 'content': prompts.SOLVE_WITH_SEEDQ.format(
            q=items[k]['question'][:5000], sq=items[k]['seed_quantity'][:300])}],
            max_tokens=args.solve_max_tokens, temperature=1.0, tag=f'solve/{k}')
        m = re.search(r'FINAL:\s*(.+?)\s*$', text, re.S | re.M)
        final = m.group(1).strip().split('\n')[0].strip() if m else None
        m = re.search(r'SEED_Q_VALUE:\s*(.+?)\s*$', text, re.M)
        return k, final, (m.group(1).strip() if m else None), text

    solved = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(solve_one, k) for k in range(len(items))]):
            try:
                k, final, seed_value, text = fu.result()
            except Exception as e:
                print(f'   solve failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            solved[k] = (final, seed_value, text)
            done += 1
            if done % 80 == 0:
                print(f'   {done}/{len(items)}', flush=True)

    auditable = [k for k, (final, _, text) in solved.items()
                 if final and text and 'ill_posed' not in final.lower()]
    print(f'[3/3] auditing {len(auditable)} derivations, answer withheld', flush=True)

    def audit_one(k):
        text = cli.chat([{'role': 'user', 'content': prompts.AUDIT.format(
            q=items[k]['question'][:4000],
            sol=prompts.clip_for_audit(solved[k][2]))}],
            max_tokens=args.audit_max_tokens, temperature=1.0, tag=f'audit/{k}')
        return k, text

    audits = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(audit_one, k) for k in auditable]):
            try:
                k, text = fu.result()
            except Exception as e:
                print(f'   audit failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            audits[k] = text
            done += 1
            if done % 80 == 0:
                print(f'   {done}/{len(auditable)}', flush=True)

    stat = collections.Counter()
    kept = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for k, it in enumerate(items):
            final, seed_value, solution = solved.get(k, (None, None, None))
            audit = audits.get(k, '')
            m = re.search(r'VERDICT:\s*(CORRECT|INCORRECT|ILL_POSED|UNSURE)', audit or '', re.I)
            verdict = (m.group(1).upper() if m
                       else ('NO_SOLUTION' if k not in auditable else 'PARSE_FAIL'))
            mb = re.search(r'FIRST_BAD_STEP:\s*(.+)', audit or '')
            stat[verdict] += 1
            if verdict != 'CORRECT':
                continue
            kept += 1
            out.write(json.dumps({
                'question': it['question'],
                'ground_truth': final,
                'answer_form': it['form'],
                'seed_quantity': it['seed_quantity'],
                'seed_quantity_value': seed_value,
                'solution': solution,
                'audit_verdict': verdict,
                'audit_first_bad_step': (mb.group(1).strip()[:300] if mb else ''),
            }, ensure_ascii=False) + '\n')

    print(f'   audit verdicts: {dict(stat)}', flush=True)
    print(f'DONE  kept {kept}/{len(items)} = '
          f'{100 * kept / max(1, len(items)):.1f}%  -> {args.out}')


if __name__ == '__main__':
    main()
