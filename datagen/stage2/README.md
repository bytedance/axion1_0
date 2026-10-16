# Stage II — Verifiable Tool-Use Data

The released Stage II set is 7,544 trajectories (~312M tokens), shipped with
the paper. **This page is how to generate more of the same kind** — from the
databases below, or from any other source you point a generator at.

A generator walks a public database: filter a pool, identify exactly one
target, compute a derived quantity. The reference answer therefore comes from
the construction rather than from a model.

The steps are:

1. **Generate questions** — reuse a generator below, or write one against the
   fields your own database happens to expose: every database supports its own
   kinds of question, so the generator is written to fit the data rather than
   the data forced into a template
2. **Rewrite the question text** so it reads naturally, and verify the rewrite
3. **Drop the closed-book-solvable items** that need no retrieval
4. **Sample an advanced model** in your own agent harness
5. **Score** against the reference answers
6. **Repair, compress and flatten** the surviving trajectories into SFT data

## Install

```bash
pip install openai numpy
export OPENAI_API_KEY=...            # rewriting and closed-book filtering
export OPENAI_BASE_URL=...           # OpenAI-compatible endpoint
```

## Generate questions

Three generators, each asking for a different kind of work. A real item from
each:

### `gen_single_row.py` — one database, pick a row by a property

> Consider all elliptic curves over Q of conductor 3420, up to isomorphism
> over Q. Take the one whose minimal discriminant has the largest absolute
> value. For that curve, what is the trace of Frobenius a_p at p = 1123?

Find the right table, single out one row, compute. The generator checks its
own arithmetic against reference values shipped with the database.

```bash
git clone https://github.com/JohnCremona/ecdata ./data/ecdata
python generators/gen_single_row.py --data ./data/ecdata/allcurves --n 60 --out ec.jsonl
```

> Cremona elliptic curve tables · Artistic-2.0 · one clone, then fully offline

### `gen_ranked_list.py` — one database, build a list and rank it

> Among all small airports in the Los Lagos region of Chile, which one has the
> lowest published field elevation?

No web page lists that, so the list has to be assembled before comparing. In
half the items the deciding attribute lives in a second dataset.

```bash
# one source
python generators/gen_ranked_list.py --src PWR --n 200 --out pwr.jsonl --cache ./data
# five unrelated sources at once
python generators/gen_ranked_list.py --src AIR,GEO,LANG,PWR,OA --n 800 \
    --out rank.jsonl --cache ./data
```

| `--src` | Dataset | License |
|--|--|--|
| `AIR` | OurAirports | Public Domain |
| `GEO` | GeoNames place names | CC BY 4.0 |
| `LANG` | Glottolog languages | CC BY 4.0 |
| `PWR` | WRI Global Power Plant | CC BY 4.0 |
| `OA` | OpenAlex institutions | CC0 1.0 |

Datasets download on demand. `--sourceless 1` removes every mention of a data
source from the question, which makes items harder.

### `gen_chain.py` — two databases, follow a chain

> Consider the enzyme from *Homo sapiens* that catalyzes the reaction
> `1,2-dihydroxy-5-(methylsulfanyl)pent-1-en-3-one + O2 = ...`. Take its
> one-to-one orthologous gene in *Danio rerio*. What is the length, in amino
> acids, of the longest protein isoform that gene encodes?

Three lookups across two databases, each with exactly one right answer
(Q9BV57 → ENSG00000182551 → ENSDARG00000020448). All three are recorded, so a
judge can tell *where* a solver went wrong, and RL has something to reward
besides the final number.

```bash
python generators/gen_chain_pool.py --out ./data/up_proteins.json
python generators/gen_chain.py --pool ./data/up_proteins.json --n 60 \
    --out x3.jsonl --cache ./data/xdb_cache.json
```

> UniProt CC BY 4.0 · Ensembl unrestricted per its disclaimer

This one queries both APIs live, so expect it to be slow — minutes per item on
a cold cache. Keep the default set of species when building the pool.

## Questions → training data

```bash
# 2. rewrite templated text into natural prose; verified for literal coverage
#    and no leakage, and discarded on failure
python pipeline/paraphrase.py --out ./paraphrased --model gpt-5.5 rank.jsonl

# 3. answer closed-book, drop whatever the model already guesses
python pipeline/closedbook_filter.py --probe probe.jsonl --key key.jsonl \
    --out cb.jsonl --model gpt-5.5

# 4. sample an advanced model in your own ReAct harness

# 5. score against the reference answers — no judge model needed
python pipeline/judge.py --key key.jsonl --out report.json iter1.jsonl iter2.jsonl

# 6. recover trajectories whose reasoning is sound but whose form is flawed
python pipeline/repair_traj.py iter1.jsonl --out ./repaired

# 7. bring over-long trajectories under the cutoff
python pipeline/compress_traj.py ./repaired/iter1.jsonl --out ./compressed \
    --cutoff 131072

# 8. flatten into multi-turn sharegpt messages
python ../utils/to_sft.py --results ./compressed --judgment judgment.json \
    --out sft.jsonl --tools PythonInterpreter,search,visit,google_scholar
```

`score.py` (matching rules) and `../utils/llm_client.py` (retrying,
checkpointed API client) are imported, not run directly.

### What step 4 needs

Sampling is the only step that calls outside services, and the harness is not
part of this repo. Ours read these from the environment:

| Tool | Variable | What it is |
|--|--|--|
| `search`, `google_scholar` | `SERPER_KEY_ID` | Serper API key |
| `visit` | `JINA_API_KEYS` | Jina Reader key(s), comma-separated, for fetching a page |
| `visit` | `SUMMARY_MODEL_NAME` | the model that condenses a fetched page against the current goal — a separate, usually smaller model than the one being sampled |
| `visit` | `SUMMARY_MODEL_KEYS` | `{"<model>": "<key>"}` when that model needs its own key; otherwise the global key is used |
| `PythonInterpreter` | `E2B_API_KEY`, or `SANDBOX_FUSION_ENDPOINT` / `SANDBOX_FUSION_ENDPOINTS` | a sandbox to execute code in |

`visit` is the expensive one: a page is fetched once and cached, but the
summary is recomputed per `(url, goal)` pair, so nearly every call hits the
summarisation model.

Compute the final token counts with your training framework's own chat
template, not from the raw message content — the tool-call template adds a
margin that differs between frameworks.

## Output format

```json
{"id": "ec-694bce86", "walk": "EC",
 "question": "Consider all elliptic curves over Q of conductor 3420 ... a_p for p = 1123?",
 "ground_truth": "-40", "answer_type": "exactMatch", "tolerance": null,
 "tree": {"id": "root", "type": "sequential", "children": [
    {"id": "identify", "critical": false, "expected": "3420c4", "desc": "Cremona 3420c4"},
    {"id": "compute",  "critical": true,  "expected": -40,       "desc": "ap {'p': 1123}"}]},
 "meta": {"N": 3420, "identify_by": "disc", "derive": "ap"}}
```

`answer_type` is `exactMatch` or `numeric` (the latter needs `tolerance`).
`tree` holds the expected value of each hop, which lets a judge tell "retrieved
the wrong object" from "computed it wrong" and supplies a process reward in RL.

## Writing a generator for another database

```python
for _ in range(max_tries):
    pool = [r for r in rows if cond_a(r) and cond_b(r)]       # two crossed conditions
    if not (POOL_LO <= len(pool) <= POOL_HI):
        continue                      # too small to discriminate, too large to assemble
    sel = [r for r in pool if r[attr] == target]
    if len(sel) != 1:
        continue                      # ambiguous -> discard, never "take the first"
    emit({'question': ..., 'ground_truth': str(derive(sel[0])),
          'tree': {...}, 'meta': {...}})
```

Four rules, each learned by breaking it:

- **Not unique → discard the item.** Never "take the first".
- **An extremum needs a clear margin** over the runner-up, or solvers whose
  source has slightly different values cannot decide.
- **Pick pools whose membership is reproducible.** We once defined pools as
  "airports with an ICAO code" — a real standard, but it is also issued to
  airstrips and heliports that no public list covers, so our pool and the
  solver's were different sets and the answers disagreed. An identifier being
  standard does not make the set it defines reproducible.
- **Validate the generator independently.** Its answers are the reference
  answers, so a bug in it cannot be caught downstream. Compare against values
  shipped with the database, or compute the quantity a second way.
