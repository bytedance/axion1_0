#!/usr/bin/env python3
# Adapted for Axion publication: explicit model path, installed Swift package, 128K default.
"""Recompute total_tokens with teenx-swift's REAL training template (qwen3_5, train mode), the same
encode the trainer runs, so the packer's length budget matches what the model sees.
Usage (inside the training image): python tools_template_lengths.py IN.jsonl OUT.jsonl [--cap 131072] --model MODEL_DIR [--workers 16]
Writes every record with total_tokens = template length (visual_tokens=0), drops records > cap, prints stats."""
import sys, json, os, argparse, statistics
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--cap', type=int, default=131072)
ap.add_argument('--workers', type=int, default=16); ap.add_argument('--model', required=True)
a = ap.parse_args()
_tpl = None
def _build_template():
    from swift.template import get_template
    from swift.model import get_model_processor
    _, processor = get_model_processor(a.model, load_model=False, model_type='qwen3_5_moe')
    tpl = get_template(processor, template_type='qwen3_5', max_length=None, truncation_strategy='raise')
    tpl.set_mode('train')
    return tpl
def _init():
    global _tpl
    _tpl = _build_template()
def enc(line):
    d = json.loads(line)
    L = len(_tpl.encode({'messages': d['messages']})['input_ids'])
    return L, d.get('total_tokens')
if __name__ == '__main__':
    # fail fast in the parent (a failing pool initializer respawns forever and idles the job)
    _probe = _build_template(); _probe_len = len(_probe.encode({'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]})['input_ids'])
    print(f'[template-len] template probe ok: {_probe_len} tokens for a 2-turn toy chat', flush=True)
    lines = open(a.src).readlines()
    with Pool(a.workers, initializer=_init) as p: res = p.map(enc, lines, chunksize=8)
    kept = 0; ratios = []; lens = []
    with open(a.dst, 'w') as g:
        for line, (L, old) in zip(lines, res):
            lens.append(L)
            if old: ratios.append(L / old)
            if L <= a.cap:
                d = json.loads(line); d['total_tokens'] = L; d['visual_tokens'] = 0
                g.write(json.dumps(d, ensure_ascii=False) + '\n'); kept += 1
    print(f'[template-len] {a.src}: n={len(lines)} kept={kept} (cap {a.cap}) template_len mean={statistics.mean(lens):.0f} p99={sorted(lens)[int(0.99*len(lens))-1]} max={max(lens)} total={sum(lens)/1e6:.1f}M')
    if ratios: print(f'[template-len] template/old_total_tokens ratio mean={statistics.mean(ratios):.3f} min={min(ratios):.3f} max={max(ratios):.3f}')
