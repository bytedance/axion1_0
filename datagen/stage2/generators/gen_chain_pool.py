#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Build the UniProt anchor pool consumed by gen_chain.py.

Fetches reviewed enzymes carrying a catalytic-activity annotation for the
requested organisms and stores them as JSON. The pool holds candidate proteins
and their reaction strings only; it contains no answers. Uniqueness is asserted
against the live API at question-generation time.

The pool is created if absent, otherwise only missing organisms are appended
(existing ones are skipped and the file is backed up first).

Data: UniProt  https://www.uniprot.org/help/license   CC BY 4.0

Usage:
    # default: the eight species gen_chain.py supports
    python gen_chain_pool.py --out ./data/up_proteins.json

    # explicit organisms (NCBI taxonomy ids)
    python gen_chain_pool.py --out ./data/up_proteins.json --taxa 9606,10090
"""
import argparse
import json
import os
import shutil
import time
import urllib.parse as P
import urllib.request as U

DEFAULT_TAXA = [9606, 10090, 10116, 7955, 7227, 6239, 559292, 9913]

FIELDS = ('accession,protein_name,gene_primary,organism_name,organism_id,ec,'
          'cc_catalytic_activity,cc_subcellular_location,length')


def fetch(tax):
    out=[]; url='https://rest.uniprot.org/uniprotkb/search?query='+P.quote(f'organism_id:{tax} AND reviewed:true AND cc_catalytic_activity:*')+f'&size=500&format=json&fields={FIELDS}'
    while url:
        for k in range(4):
            try: r=U.urlopen(U.Request(url,headers={'User-Agent':'axion/1.0'}),timeout=120); d=json.loads(r.read().decode()); break
            except Exception as e:
                print('retry',tax,k,repr(e)[:100],flush=True); time.sleep(3*(k+1))
        else: raise RuntimeError('uniprot down')
        for e in d['results']:
            rxn=[]; ec=[]; loc=[]
            for c in e.get('comments',[]):
                if c.get('commentType')=='CATALYTIC ACTIVITY':
                    n=(c.get('reaction') or {}).get('name')
                    if n: rxn.append(n)
                    for x in (c.get('reaction') or {}).get('ecNumber',[]) if isinstance((c.get('reaction') or {}).get('ecNumber'),list) else []: ec.append(x)
                    en=(c.get('reaction') or {}).get('ecNumber')
                    if isinstance(en,str): ec.append(en)
                if c.get('commentType')=='SUBCELLULAR LOCATION':
                    for s in c.get('subcellularLocations',[]):
                        v=(s.get('location') or {}).get('value')
                        if v: loc.append(v)
            pd=e.get('proteinDescription',{}); name=((pd.get('recommendedName') or {}).get('fullName') or {}).get('value') or ((pd.get('submissionNames') or [{}])[0].get('fullName') or {}).get('value')
            for x in (pd.get('recommendedName') or {}).get('ecNumbers',[]): ec.append(x['value'])
            out.append({'acc':e['primaryAccession'],'name':name,'gene':(e.get('genes') or [{}])[0].get('geneName',{}).get('value'),
                        'org':e['organism']['scientificName'],'tax':e['organism']['taxonId'],'ec':sorted(set(ec)),'rxn':rxn,'loc':sorted(set(loc)),'length':e['sequence']['length']})
        link=r.headers.get('Link') or ''
        url=link.split('<')[1].split('>')[0] if 'rel="next"' in link else None
        time.sleep(0.3)
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='path of the anchor-pool json')
    ap.add_argument('--taxa', default='',
                    help='comma-separated NCBI taxonomy ids; defaults to eight species')
    args = ap.parse_args()

    taxa = [int(t) for t in args.taxa.split(',') if t.strip()] or DEFAULT_TAXA
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)

    if os.path.exists(args.out):
        pool = json.load(open(args.out))
        shutil.copy(args.out, args.out + '.bak_' + time.strftime('%m%d_%H%M'))
        print(f'[resume] {len(pool)} entries present, backup written')
    else:
        pool = []
        print('[new] creating pool')

    have = {p['tax'] for p in pool}
    for t in taxa:
        if t in have:
            print(f'skip {t}, already in pool', flush=True)
            continue
        rows = fetch(t)
        print(f'tax {t}  {rows[0]["org"] if rows else "?"}  {len(rows)} proteins', flush=True)
        pool.extend(rows)
        have.add(t)
        json.dump(pool, open(args.out, 'w'))
    print(f'DONE  {len(pool)} entries written to {args.out}')


if __name__ == '__main__':
    main()
