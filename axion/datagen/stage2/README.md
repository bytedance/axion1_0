# Stage II — Verifiable Tool-Use Data

The paper uses 7,544 Stage II trajectories (~312M tokens). This pipeline
constructs tool-dependent questions from public databases and verifies answers
against the source records.

A generator walks a public database: filter a pool, identify exactly one
target, compute a derived quantity. The reference answer therefore comes from
the construction rather than from a model.

The steps are:

1. **Generate questions** with source-specific filters, selection rules and calculations
2. **Rewrite the question text** so it reads naturally, and verify the rewrite
3. **Drop the closed-book-solvable items** that need no retrieval
4. **Sample an advanced model** in your own agent harness
5. **Score** against the reference answers
6. **Repair, compress and flatten** the surviving trajectories into SFT data

Run the commands below from `axion/datagen/stage2/`.

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

> Consider all confirmed exoplanets in systems with exactly 6 known planets.
> Among them, take the one with the largest orbital semi-major axis. For that
> planet, determine its mean density in g/cm^3, computed from its mass and
> radius. Use M_Earth = 5.9722e24 kg; R_Earth = 6371 km. Give the numerical
> value to three significant figures.

Find the right table, single out one row, compute. The pool constraint, the
field the row is singled out by, and the derived quantity are each drawn at
random, so one database yields many shapes of question. Where the catalogue
also tabulates the derived quantity, the generator checks its own arithmetic
against that column and discards the item on a mismatch.

```bash
python generators/gen_single_row.py --n 60 --out exo.jsonl --cache ./data/pscomppars.csv
```

The table is downloaded on first run; `--cache` keeps it for later runs.

> NASA Exoplanet Archive · US public domain · one download, then fully offline

### `gen_ranked_list.py` — one database, build a list and rank it

> Among all small airports in the Los Lagos region of Chile, which one has the
> lowest published field elevation?

The solver assembles the candidate list before comparing values. Some items
require a second dataset for the deciding attribute.

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

Use an external agent harness for sampling. Its retrieval and code-execution
tools need the following service configuration:

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
{"id": "exo-bf6422b2", "walk": "EXO",
 "question": "Consider all confirmed exoplanets in systems with exactly 6 known planets ...",
 "ground_truth": "0.264 g/cm^3", "answer_type": "numeric", "tolerance": 0.02,
 "tree": {"id": "root", "type": "sequential", "children": [
    {"id": "identify", "critical": false, "expected": "HD 34445 g",
     "desc": "Identify the planet: largest orbital semi-major axis in the pool"},
    {"id": "params",   "critical": false, "expected": {"pl_bmasse": 120.6, "pl_rade": 13.6},
     "desc": "Retrieve the required parameters"},
    {"id": "final",    "critical": true,  "expected": "0.264", "desc": "density value"}]},
 "meta": {"planet": "HD 34445 g", "id_field": "pl_orbsmax", "direction": "largest",
          "deriv": "density"}}
```

`answer_type` is `exactMatch` or `numeric` (the latter needs `tolerance`).
`tree` holds the expected value of each hop, which lets a judge tell "retrieved
the wrong object" from "computed it wrong" and supplies a process reward in RL.

## RL task selection

The [RL guide](../../training/rl/README.md#prepare-tasks) documents
`pipeline/prepare_rl.py`: explicit intermediate targets, reviewed exclusions,
and selection from four complete attempts at one checkpoint. It writes the
training parquet and its selection manifest. Legacy generator `tree` fields
must be mapped to reviewed intermediate targets; the selector does not infer
them from `critical` flags.

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

Generator requirements:

- **Not unique → discard the item.** Never "take the first".
- **An extremum needs a clear margin** over the runner-up, or solvers whose
  source has slightly different values cannot decide.
- **Use reproducible candidate pools.** Choose membership rules supported by a
  complete public source, so construction and solving use the same records.
- **Validate the generator independently.** Its answers are the reference
  answers, so a bug in it cannot be caught downstream. Compare against values
  shipped with the database, or compute the quantity a second way.
