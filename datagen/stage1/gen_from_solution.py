#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Rewrite questions your base model already solves, against the route it took.

The attacker is shown the question *and the solution the base model produced*,
and has to break that specific route while keeping the problem well-posed,
still solvable by a strong model, and self-contained.

Pass --traj-dir: a judgment file holds only the extracted final answer, which
is far too little for the rewrite to target an actual route.

A rewritten question invalidates the old answer, so each one is solved and
audited again, and kept only if the audit returns CORRECT.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.

Usage:
    python gen_from_solution.py --judgment judgment_results.json --traj-dir rollouts/ \
        --out attacked.jsonl --model gpt-5.5
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prompts

ATTACK = """You are the ATTACKER in an adversarial data-generation loop. Your job is to rewrite an
exam question so that a specific target model can no longer solve it — while keeping the question
completely legitimate.

Below is a question the TARGET MODEL solved correctly, together with the actual solution it produced.
Study how it won.

=== QUESTION ===
{q}

=== TARGET MODEL'S WINNING SOLUTION ===
{sol}

=== YOUR TASK ===
Rewrite the question so the target's route above no longer works.

HARD CONSTRAINTS — a rewrite violating any of these is worthless:
1. WELL-POSED. Every object you name must exist and be unique. The answer must be unique.
2. STILL SOLVABLE. A strong expert model must be able to solve it from first principles.
   You are making it harder, NOT unsolvable.
3. SELF-CONTAINED. No "look up", no reference to any paper/table/database. All data stated.
4. NO CHEAP TRICKS. These do not count as making it harder and are forbidden:
   - making the question merely longer or more verbose
   - burying the question in irrelevant detail
   - requiring an obscure named fact or a specific paper's notation
   - demanding an unusual output format, many decimal places, or many separate sub-answers
   - requiring more tool calls just to inflate the interaction count
5. THE DIFFICULTY MUST BE MATHEMATICAL. The target should fail because a reasoning step is
   genuinely harder, not because it got confused, ran out of budget, or misparsed anything.

USE THIS TACTIC: {tactic}

  A. COMPOSITION — chain one more genuine operation whose input is this question's answer
  B. GENERALIZE — replace a concrete value with a parameter; the answer becomes an expression
  C. REGIME — push a parameter to where the target's shortcut/enumeration stops working
  D. CASE SPLIT — choose parameters so the answer genuinely splits into cases the target will miss
  E. TRAP — ask for a quantity that a plausible-but-wrong reading would compute differently
     (state explicitly what the wrong quantity would be)

Output EXACTLY:

ATTACK_ON: <one line: which step of the target's solution you are breaking>
TACTIC: <the tactic letters you used>
NEW_QUESTION:
<the rewritten question, self-contained>
WHY_TARGET_FAILS: <one or two lines: the specific step the target will now get wrong>
TRAP_ANSWER: <the wrong answer the target will most likely give, or NONE>"""


# Mix of tactics. Composition combined with a trap was the strongest in our
# measurements; composition alone was close to useless. The rest of the budget
# is spread over the other tactics rather than pushed entirely into the best
# one, so a batch is not monotonous.
TACTIC_QUOTA = [('A+E', .40), ('E', .25), ('A+D', .10), ('B+E', .10),
                ('C', .10), ('D', .05)]


def field(text, key, next_key):
    """Pull the block between two labels out of a structured model reply."""
    m = re.search(rf'{key}:\s*(.*?)(?=\n{next_key}:|\Z)', text or '', re.S)
    return m.group(1).strip() if m else ''


def load_solutions(traj_dir):
    """Map question -> the longest assistant text the base model produced for it."""
    out = {}
    if not traj_dir:
        return out
    for fp in sorted(glob.glob(os.path.join(traj_dir, 'iter*.jsonl'))):
        for line in open(fp, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            text = '\n\n'.join(m['content'] for m in (j.get('messages') or [])
                                if m.get('role') == 'assistant'
                                and isinstance(m.get('content'), str))
            if len(text) > len(out.get(j.get('question', ''), '')):
                out[j['question']] = text
    return out


def clip_solution(text, head=4000, tail=8000, cap=12000):
    """Clip a solution for the attacker, keeping both ends.

    The opening states the approach taken and the closing carries the decisive
    steps, so a head-only clip would cut off the conclusion the attack targets.
    """
    return text if len(text) <= cap else (
        text[:head] + '\n\n...[middle omitted]...\n\n' + text[-tail:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--judgment', required=True,
                    help="the base model's judgment_results.json")
    ap.add_argument('--out', required=True, help='output jsonl')
    ap.add_argument('--traj-dir', default=None,
                    help="directory of the base model's rollouts (iter*.jsonl); "
                         'strongly recommended, see the note above')
    ap.add_argument('--model', default=None, help='model name (defaults to $OPENAI_MODEL)')
    ap.add_argument('--round', type=int, default=1, help='attack round, used in ids')
    ap.add_argument('--min-pass', type=float, default=1.0,
                    help='only attack questions whose pass rate is at least this')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--qpm', type=int, default=60, help='queries per minute')
    ap.add_argument('--attack-max-tokens', type=int, default=8000)
    ap.add_argument('--solve-max-tokens', type=int, default=16000)
    ap.add_argument('--audit-max-tokens', type=int, default=8000)
    args = ap.parse_args()

    cli = prompts.client(args.model, qpm=args.qpm)

    solutions = load_solutions(args.traj_dir)
    if solutions:
        lens = sorted(len(v) for v in solutions.values())
        print(f'[traj] {len(solutions)} solutions, median length '
              f'{lens[len(lens) // 2]}', flush=True)
    else:
        print('[traj] no --traj-dir given: falling back to the final answer only, '
              'which makes the rewrite much weaker', flush=True)

    d = json.load(open(args.judgment, encoding='utf-8'))
    records = d if isinstance(d, list) else d['results']
    by_question = collections.defaultdict(list)
    for x in records:
        by_question[x['question']].append(x)

    targets = []
    for q, recs in by_question.items():
        rate = sum(1 for r in recs if r['is_correct']) / len(recs)
        if rate >= args.min_pass:
            sol = solutions.get(q) or max(
                (r.get('predicted_answer') or '' for r in recs), key=len)
            targets.append((q, sol))
    print(f'[round {args.round}] {len(targets)} questions at pass rate '
          f'>= {args.min_pass}', flush=True)
    if not targets:
        raise SystemExit('nothing to attack')

    tactics = []
    for name, frac in TACTIC_QUOTA:
        tactics += [name] * max(1, round(frac * len(targets)))
    random.seed(1000 + args.round)
    random.shuffle(tactics)
    tactics = (tactics * 3)[:len(targets)]

    def attack_one(i):
        q, sol = targets[i]
        return i, cli.chat([{'role': 'user', 'content': ATTACK.format(
            q=q[:4000], sol=clip_solution(sol), tactic=tactics[i])}],
            max_tokens=args.attack_max_tokens, temperature=1.0, tag=f'attack/{i}')

    attacked = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(attack_one, i) for i in range(len(targets))]):
            try:
                i, text = fu.result()
            except Exception as e:
                print(f'   attack failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            attacked[i] = text
            done += 1
            if done % 20 == 0:
                print(f'   attacked {done}/{len(targets)}', flush=True)

    items = [(i, text, field(text, 'NEW_QUESTION', 'WHY_TARGET_FAILS'))
             for i, text in attacked.items()
             if field(text, 'NEW_QUESTION', 'WHY_TARGET_FAILS')]
    print(f'   rewritten {len(items)}/{len(targets)}, now solving each once', flush=True)

    def solve_one(k):
        text = cli.chat([{'role': 'user', 'content': prompts.SOLVE_BASE.format(
            q=items[k][2][:5000])}],
            max_tokens=args.solve_max_tokens, temperature=1.0, tag=f'solve/{k}')
        m = re.search(r'FINAL:\s*(.+?)\s*$', text, re.S | re.M)
        return k, (m.group(1).strip().split('\n')[0].strip() if m else None), text

    finals, solution_text = {}, {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(solve_one, k) for k in range(len(items))]):
            try:
                k, final, text = fu.result()
            except Exception as e:
                print(f'   solve failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            finals[k], solution_text[k] = final, text
            done += 1
            if done % 60 == 0:
                print(f'   solved {done}/{len(items)}', flush=True)

    print('   auditing, answer withheld', flush=True)

    def audit_one(k):
        return k, cli.chat([{'role': 'user', 'content': prompts.AUDIT.format(
            q=items[k][2][:4000],
            sol=prompts.clip_for_audit(solution_text.get(k) or ''))}],
            max_tokens=args.audit_max_tokens, temperature=1.0, tag=f'audit/{k}')

    audits = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(audit_one, k) for k in range(len(items))]):
            try:
                k, text = fu.result()
            except Exception as e:
                print(f'   audit failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                continue
            audits[k] = text
            done += 1
            if done % 60 == 0:
                print(f'   audited {done}/{len(items)}', flush=True)

    stat = collections.Counter()
    written = 0
    with open(args.out, 'w', encoding='utf-8') as out:
        for k, (i, text, new_q) in enumerate(items):
            final = finals.get(k)
            if not final or 'ill_posed' in final.lower():
                stat['ill_posed'] += 1
                continue
            m = re.search(r'VERDICT:\s*(CORRECT|INCORRECT|ILL_POSED|UNSURE)',
                          audits.get(k, '') or '', re.I)
            verdict = m.group(1).upper() if m else 'PARSE_FAIL'
            if verdict != 'CORRECT':
                stat[f'audit_{verdict.lower()}'] += 1
                continue
            stat['kept'] += 1
            written += 1
            out.write(json.dumps({
                'id': f'adv{args.round}_' +
                      hashlib.md5(new_q[:400].encode()).hexdigest()[:16],
                'question': new_q,
                'ground_truth': final,
                'answer_type': 'exactMatch',
                'solution': solution_text.get(k, ''),
                'meta': {
                    'round': args.round,
                    'tactic_assigned': tactics[i],
                    'tactic_used': field(text, 'TACTIC', 'NEW_QUESTION')[:120],
                    'attack_on': field(text, 'ATTACK_ON', 'TACTIC')[:250],
                    'why_target_fails': field(text, 'WHY_TARGET_FAILS', 'TRAP_ANSWER')[:350],
                    'trap_answer': field(text, 'TRAP_ANSWER', 'ZZZ')[:200],
                    'original_question': targets[i][0][:2500],
                },
            }, ensure_ascii=False) + '\n')

    print(f'\n{dict(stat)}')
    print(f'DONE  {written} questions -> {args.out}')


if __name__ == '__main__':
    main()
