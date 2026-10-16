# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Cross-database multi-hop: UniProt reaction anchor -> Ensembl orthologue.

Construction pattern (one of three used in Stage II):
    pick a protein whose catalytic activity is unique within its organism
      -> hop to Ensembl for its orthologous gene
      -> read the target quantity straight from the database

Every hop asserts uniqueness through an API count. An anchor is kept only when
`cc_catalytic_activity:"<reaction>" AND organism_id:<tax>` matches exactly one
entry; otherwise the item is discarded. This is what makes the reference answer
unambiguous.

The `tree` field records the expected intermediate value of each hop, which
lets a judge localise where a failure happened and provides a process-level
reward signal during RL.

With --blind (default) the question text does not name any database, so a
solver has to work out which resources to consult.

Data
    UniProt   https://www.uniprot.org/help/license   CC BY 4.0
    Ensembl   project-generated data, usable without restriction per the
              official disclaimer

Prerequisite: the anchor pool produced by gen_chain_pool.py

Usage
    python gen_chain_pool.py --out ./data/up_proteins.json
    python gen_chain.py --pool ./data/up_proteins.json --n 60 --out x3.jsonl
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import time
import urllib.parse as P
import urllib.request as U

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--pool', required=True, help='anchor pool produced by gen_chain_pool.py')
ap.add_argument('--out', required=True, help='output jsonl (appended, so runs can be resumed)')
ap.add_argument('--n', type=int, default=60, help='number of questions to generate')
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--cache', default='./xdb_cache.json', help='cache file for API responses')
ap.add_argument('--blind', type=int, default=1, help='1 = question text does not name any database')
ap.add_argument('--seen', default='', help='comma-separated earlier outputs; their anchors are excluded')
args = ap.parse_args()

random.seed(args.seed)
WALK, N, BLIND = 'X3', args.n, args.blind == 1
OUT, CACHEF = args.out, args.cache
SEEN_FILES = [f for f in args.seen.split(',') if f]
CACHE_DIR = os.path.dirname(os.path.abspath(CACHEF)) or '.'
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE = json.load(open(CACHEF)) if os.path.exists(CACHEF) else {}
_ncall = 0
def http(url,data=None,hdr=None,tries=4):
    global _ncall
    k=url+('|'+data.decode() if data else '')
    if k in CACHE: return CACHE[k]
    for t in range(tries):
        try:
            r=U.urlopen(U.Request(url,data=data,headers=hdr or {'User-Agent':'axion/1.0'}),timeout=90)
            txt=r.read().decode(); CACHE[k]=txt; _ncall+=1
            if _ncall%50==0: json.dump(CACHE,open(CACHEF,'w'))
            return txt
        except Exception as e:
            code=getattr(e,'code',None)
            if code in (404,400): CACHE[k]=''; return ''
            if code==204: CACHE[k]=''; return ''
            time.sleep(2*(t+1))
    return None
def J(txt):
    try: return json.loads(txt) if txt else None
    except Exception: return None
def up_count(q):
    t=http('https://rest.uniprot.org/uniprotkb/search?query='+P.quote(q)+'&size=1&format=json&fields=accession')
    d=J(t); return None if d is None else int(d.get('results') is not None and len(d['results'])>0 and 1 or 0) if False else (None if d is None else (len(d['results']) if 'results' in d else None))
def up_total(q):
    t=http('https://rest.uniprot.org/uniprotkb/search?query='+P.quote(q)+'&size=500&format=json&fields=accession')
    d=J(t); return None if d is None else len(d.get('results',[]))
def up_xrefs(tax,db):
    fn = os.path.join(CACHE_DIR, f'up_xref_{db}_{tax}.json')
    if os.path.exists(fn): return json.load(open(fn))
    out={}; url='https://rest.uniprot.org/uniprotkb/search?query='+P.quote(f'organism_id:{tax} AND reviewed:true AND database:{db} AND cc_catalytic_activity:*')+f'&size=500&format=json&fields=accession,gene_primary,xref_{db}'
    while url:
        r=U.urlopen(U.Request(url,headers={'User-Agent':'axion/1.0'}),timeout=120); d=json.loads(r.read().decode())
        for e in d['results']:
            xr=collections.defaultdict(list)
            for x in e.get('uniProtKBCrossReferences',[]):
                props={p['key']:p['value'] for p in x.get('properties',[])}
                xr[x['database']].append((x['id'],props))
            out[e['primaryAccession']]={'gene':(e.get('genes') or [{}])[0].get('geneName',{}).get('value'),'xr':xr}
        link=r.headers.get('Link') or ''
        m=re.search(r'<([^>]+)>; rel="next"',link); url=m.group(1) if m else None
    json.dump(out,open(fn,'w')); return out
ENS='https://rest.ensembl.org'
SPECIES = {9606: 'homo_sapiens', 10090: 'mus_musculus', 10116: 'rattus_norvegicus',
           7955: 'danio_rerio', 7227: 'drosophila_melanogaster',
           6239: 'caenorhabditis_elegans', 559292: 'saccharomyces_cerevisiae',
           9913: 'bos_taurus'}

# UniProt files the Ensembl cross-reference under a per-division name, so the
# query term, the fields selector and the response key all have to match the
# division: vertebrates use "ensembl"/"Ensembl", invertebrates
# "ensemblmetazoa"/"EnsemblMetazoa", fungi "ensemblfungi"/"EnsemblFungi".
# Using the plain name for a non-vertebrate returns no cross-references at all,
# without any error.
DIVISION = {7227: ('ensemblmetazoa', 'EnsemblMetazoa'),
            6239: ('ensemblmetazoa', 'EnsemblMetazoa'),
            559292: ('ensemblfungi', 'EnsemblFungi')}

# Scientific names of the supported species, taken from a table rather than
# from the anchor pool: an orthologue's target species need not appear in the
# pool, so indexing the pool would raise for a partial pool.
SCIENTIFIC = {9606: 'Homo sapiens', 10090: 'Mus musculus',
              10116: 'Rattus norvegicus', 7955: 'Danio rerio',
              7227: 'Drosophila melanogaster', 6239: 'Caenorhabditis elegans',
              559292: 'Saccharomyces cerevisiae', 9913: 'Bos taurus'}


def division(tax):
    """(query/fields name, cross-reference key) of a taxon's Ensembl division."""
    return DIVISION.get(tax, ('ensembl', 'Ensembl'))
def ens(path):
    time.sleep(0.1); return J(http(ENS+path+('&' if '?' in path else '?')+'content-type=application/json',hdr={'Content-Type':'application/json','User-Agent':'axion/1.0'}))
PROT = json.load(open(args.pool))
ORG={p['tax']:p['org'] for p in PROT}
rxorg=collections.defaultdict(list)
for p in PROT:
    for r in p['rxn']: rxorg[(p['tax'],r)].append(p['acc'])
def okrxn(r): return 30<len(r)<300 and '=' in r and not r.lower().startswith(('random','hydrolysis of'))
def uniq_rxn(p):
    rs=[r for r in p['rxn'] if okrxn(r) and len(rxorg[(p['tax'],r)])==1]
    return random.choice(rs) if rs else None
def C1(r,tax): return up_total(f'cc_catalytic_activity:"{r}" AND organism_id:{tax} AND reviewed:true')
ORGSHORT=lambda o: re.sub(r'\s*\(.*$','',o)
def walk_X3(p,r,xr):
    if p['tax'] not in SPECIES: return None,'no_species'
    en=[(k,pr) for k,pr in xr.get(division(p['tax'])[1],[])]
    genes={pr.get('GeneId','').split('.')[0] for _,pr in en if pr.get('GeneId')}
    if len(genes)!=1: return None,'ens_gene_%d'%len(genes)
    gid=genes.pop(); sp=SPECIES[p['tax']]
    tgt_tax=random.choice([t for t in SPECIES if t!=p['tax']]); tsp=SPECIES[tgt_tax]
    h=ens(f'/homology/id/{sp}/{gid}?target_species={tsp};type=orthologues;format=condensed')
    hom=((h or {}).get('data') or [{}])[0].get('homologies') or []
    if len(hom)!=1 or hom[0].get('type')!='ortholog_one2one': return None,'orth_%d'%len(hom)
    tg=hom[0]['id']
    g=ens(f'/lookup/id/{tg}?expand=1')
    if not g or not g.get('Transcript'): return None,'lookup_fail'
    trs=g['Transcript']; canon=[t for t in trs if t.get('is_canonical')==1]
    if len(canon)!=1: return None,'canon'
    ct=ens(f'/lookup/id/{canon[0]["id"]}?expand=1')
    nex=len((ct or {}).get('Exon') or [])
    if not nex: return None,'exon'
    plen=max([((t.get('Translation') or {}).get('length') or 0) for t in trs if t.get('biotype')=='protein_coding'] or [0])
    if plen<50: return None,'plen'
    mode=random.choice(['plen','plen','nex'] if tgt_tax==9606 else ['plen'])
    if mode=='plen': ans=str(plen); fin='number'
    else: ans=str(nex); fin='number'
    tgt=ORGSHORT(SCIENTIFIC.get(tgt_tax) or ORG.get(tgt_tax) or '')
    if not tgt: return None,'no_target_name'
    if BLIND:
        head=f"Consider the enzyme from {ORGSHORT(p['org'])} that catalyzes the reaction: {r}. Take its one-to-one orthologous gene in {tgt}."
        tail=("What is the length, in amino acids, of the longest annotated protein isoform encoded by that gene?" if mode=='plen'
              else "How many exons does the MANE Select transcript of that gene have?")
        q=f"{head} {tail}"
    else:
        tail=("What is the length, in amino acids, of the longest protein-coding translation annotated for the orthologous gene in Ensembl?" if mode=='plen'
              else 'How many exons does the Ensembl canonical transcript of the orthologous gene have?')
        q=(f"Consider the reviewed (Swiss-Prot) UniProt entry from {ORGSHORT(p['org'])} whose annotated catalytic activity is the reaction: {r}. Follow its cross-reference to Ensembl, and use Ensembl Compara to find its one-to-one orthologue in {tgt}. {tail}")
    tree={'description':f'Sequential chain: enzyme -> Ensembl gene -> one2one orthologue in {tsp} -> {mode}','critical':True,'aggregation_strategy':'sequential','children':[
        {'description':f'Identify the enzyme: UniProt {p["acc"]} (gene {p["gene"]})','critical':False,'expected':p['acc']},
        {'description':f'Ensembl gene {gid}','critical':False,'expected':gid},
        {'description':f'one2one orthologue in {tsp}: {tg} ({g.get("display_name")})','critical':False,'expected':tg},
        {'description':f'Final answer ({mode}): {ans}','critical':True,'expected':ans}]}
    meta={'hop1':{'acc':p['acc'],'gene':p['gene'],'org':p['org'],'tax':p['tax'],'rxn':r},'hop2':{'ensembl_gene':gid},'hop3':{'target_species':tsp,'target_tax':tgt_tax,'ortholog':tg,'symbol':g.get('display_name')},'hop4':{'mode':mode,'n_transcripts':len(trs),'canonical':canon[0]['id'],'n_exons':nex,'plen':plen,'chr':g['seq_region_name'],'strand':g['strand']}}
    return dict(question=q,ground_truth=ans,answer_type=fin,tree=tree,meta=meta),'ok'
OLS='https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms?obo_id='

COFACTOR = {
 'CHEBI:30616','CHEBI:456216','CHEBI:456215',
 'CHEBI:57540','CHEBI:57945','CHEBI:58349','CHEBI:57783',
 'CHEBI:57287','CHEBI:57288',
 'CHEBI:59789','CHEBI:57856',
 'CHEBI:57692','CHEBI:58307',
 'CHEBI:43474','CHEBI:33019',
 'CHEBI:15377','CHEBI:15378','CHEBI:15379','CHEBI:16526',
}

WALKS = {'X3': ('ensembl', walk_X3)}
db, fn = WALKS[WALK]
taxa = [t for t in ORG if t in SPECIES]
random.shuffle(taxa)
cands = []
for t in taxa:
    xr = up_xrefs(t, division(t)[0])
    for p in PROT:
        if p['tax'] == t and p['acc'] in xr and p['rxn']:
            cands.append((p, xr[p['acc']]['xr']))
random.shuffle(cands)
print(WALK, 'candidates', len(cands), flush=True)

stat, out, seen = collections.Counter(), [], set()
if os.path.exists(OUT):
    for l in open(OUT):
        d = json.loads(l); out.append(d); seen.add(d['meta']['hop1']['acc'])
    print(f'[resume] {len(out)} questions already present', flush=True)
for sf in SEEN_FILES:
    for l in open(sf):
        try: seen.add(json.loads(l)['meta']['hop1']['acc'])
        except Exception: pass
print('anchors already used', len(seen), flush=True)

for p, xr in cands:
    if len(out) >= N: break
    if p['acc'] in seen: continue
    r = uniq_rxn(p)
    if not r:
        stat['no_uniq_rxn'] += 1; continue
    c = C1(r, p['tax'])
    if c != 1:
        stat['C1_%s' % c] += 1; continue
    try:
        res, why = fn(p, r, xr)
    except Exception as e:
        res, why = None, 'exc:' + type(e).__name__
        print('EXC', p['acc'], repr(e)[:200], flush=True)
    stat[why] += 1
    if res:
        res['id'] = 'xdb-%s-%s' % (WALK, hashlib.md5((p['acc'] + r).encode()).hexdigest()[:8])
        res['walk'] = WALK
        res['blind'] = BLIND
        out.append(res); seen.add(p['acc'])
        with open(OUT, 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(len(out), res['id'], res['ground_truth'], '|', res['question'][:120], flush=True)
    if sum(stat.values()) % 20 == 0:
        print(dict(stat), flush=True); json.dump(CACHE, open(CACHEF, 'w'))
json.dump(CACHE, open(CACHEF, 'w'))
print('DONE', WALK, len(out), 'rejections:', dict(stat))
