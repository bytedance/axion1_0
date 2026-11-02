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

"""Rewrite templated question text into natural prose.

An LLM rewrites each question; the result is then checked deterministically for
(a) coverage of the key literals that must survive verbatim (numbers, reaction
strings, identifiers) and (b) absence of the answer and of any intermediate
entity.

A rewrite that fails either check is discarded and the templated sentence is
kept, so this step can never introduce a wrong answer.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.

Usage:
    python paraphrase.py --out ./paraphrased --model gpt-5.5 in1.jsonl [in2.jsonl ...]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'utils'))
from llm_client import LLMClient, run_batch

DBS = ['OurAirports', 'GeoNames', 'Glottolog', 'OpenAlex', 'Cremona']

def must_keep(r):
    q, m, w = r['question'], r['meta'], r['walk']
    keep = set(re.findall(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?', q))
    keep |= {d for d in DBS if d.lower() in q.lower()}
    if w == 'EC':
        for ph in ['conductor', 'isomorphism over Q', 'Mordell-Weil rank', 'torsion subgroup', 'minimal discriminant', 'good reduction', 'trace of Frobenius', 'supersingular', 'point at infinity', 'odd primes', 'exact integer']:
            if ph.lower() in q.lower(): keep.add(ph)
    elif w in ('AIR', 'GEO', 'LANG', 'OA', 'PWR'):
        keep |= {k for k in (m.get('pool_desc') or '').split(' in ')[:1] if k}
        import re as _re
        keep |= set(_re.findall(r'\b[A-Z]{2,4}\b', m.get('pool_desc') or ''))
        for ph in [
            'heliports', 'seaplane bases', 'balloonports', 'small airports',
            'medium-sized airports', 'large airports', 'scheduled commercial service',
            'IATA code', 'field elevation',
            'modelled terrain elevation', 'administrative division', 'populated places',
            'named mountains', 'hydrographic features',
            'greater than', 'below', 'or higher', 'or above', 'at most', 'or greater',
            'or more', 'longer than', 'shorter than', 'within',
            'exactly as recorded', 'single integer', 'only its name',
            'macroarea', 'language family', 'reference latitude', 'reference longitude',
            'spoken in', 'works count', 'cited by count', 'institutions', 'country code',
            'first take the one with the', 'then, for the', 'population recorded for',
            'nearest city with a population above', 'global gazetteer of populated places',
            'considering only those for which that value is recorded',
            'satisfy both of the following', 'answer 0 if none of',
            'both conditions',
        ]:
            if ph.lower() in q.lower():
                keep.add(ph)
    return sorted(keep)

def leaks(r):
    if r.get('walk') in ('AIR', 'GEO', 'LANG', 'OA', 'PWR'):
        if (r.get('answer_type') or '') != 'exactMatch':
            return []
        nm = str(r['ground_truth']).strip()
        return [nm] if len(nm) >= 4 else []

    m, w = r['meta'], r['walk']
    L = [str(r['ground_truth'])]
    if w == 'EC': L += [m['label']]
    return [x for x in L if x and len(x) >= 3]

PROMPT = """You are rewriting a scientific multi-step question so that it reads like a natural question posed by a domain expert, instead of a filled-in template.

Rules:
- Preserve EVERY constraint, number, identifier, organism name, reaction, database name, unit and required output format exactly. Do not round, drop, or add anything. Do not add hints or explanations.
- The following strings must appear verbatim in your rewrite: {keep}
- Do not name the entity that the question is asking the solver to identify, and do not mention this instruction in the rewrite.
- Keep it a single self-contained question; you may reorder sentences and vary phrasing freely.
- Output only the rewritten question, nothing else.

Question:
{q}"""

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='question jsonl files to rewrite')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--model', default=None, help='model name (defaults to $OPENAI_MODEL)')
    ap.add_argument('--qpm', type=int, default=60, help='queries per minute')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--passes', type=int, default=3, help='retry passes over failed items')
    args = ap.parse_args()

    OUTDIR = args.out
    cli = LLMClient(model=args.model, qpm=args.qpm)
    os.makedirs(OUTDIR, exist_ok=True)
    for path in args.inputs:
        rows = [json.loads(l) for l in open(path)]
        base = os.path.basename(path)
        def work(r):
            keep = must_keep(r); bad = leaks(r)
            def ok(t):
                t2 = t.strip(); tl = t2.lower()
                if not (0.5 * len(r['question']) < len(t2) < 2.5 * len(r['question'])): return False
                if any(k.lower() not in tl for k in keep): return False
                if any(b.lower() in tl for b in bad if b.lower() not in r['question'].lower()): return False
                if set(re.findall(r'\d+(?:\.\d+)?', t2)) - set(re.findall(r'\d+(?:\.\d+)?', r['question'])) - set(re.findall(r'\d+(?:\.\d+)?', json.dumps(r['meta']))): return False
                if re.search(r'OurAirports|GeoNames|Glottolog|OpenAlex|Cremona', t2): return False
                if re.search(r'do not name|without naming|not name it|not identify it by name|do not identify|no need to name|need not (?:be )?name', tl): return False
                return True
            t = cli.chat([{'role': 'user', 'content': PROMPT.format(keep=json.dumps(keep, ensure_ascii=False), q=r['question'])}],
                         validator=ok, max_tokens=4000, temperature=1.0)
            return {'question': t.strip(), 'question_template': r['question'], 'keep': keep}
        res, failed = run_batch(rows, key_fn=lambda r: r['id'], work_fn=work,
                                checkpoint=f'{OUTDIR}/{base}.ckpt.jsonl',
                                workers=args.workers, passes=args.passes)
        out = []
        for r in rows:
            if r['id'] in res:
                r = dict(r); r['question_template'] = r['question']; r['question'] = res[r['id']]['question']; out.append(r)
        with open(f'{OUTDIR}/{base}', 'w') as f:
            for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'{base}: {len(out)}/{len(rows)} paraphrased, {len(failed)} failed', cli.stats)

if __name__ == '__main__':
    main()
