# Evaluation

Five benchmarks: how to build each question set, and how to grade rollouts on it.

**No benchmark data is in this repository.** Each directory has a `prepare.py`
that downloads its source and converts it. The only exception is
`hle/subset_130_ids.txt`, a list of 130 item ids with no questions and no
answers, so the HLE subset used here can be rebuilt without redistributing a
gated dataset.

| | items | source | grade with |
|--|--|--|--|
| **FS-O** | 100 | [openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience), split `olympiad` | `frontierscience/judge.py --bench fs_o` |
| **FS-R** | 60 | [openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience), split `research` | `frontierscience/judge.py --bench fs_r` |
| **HiPhO** | 360, or 102 text-only | [SciYu/HiPhO](https://huggingface.co/datasets/SciYu/HiPhO) | `hipho/judge.py` |
| **HLE** | 130 text-only subset | [cais/hle](https://huggingface.co/datasets/cais/hle) (gated) | `hle/judge.py` |
| **SEAL-0** | 111 | [vtllms/sealqa](https://huggingface.co/datasets/vtllms/sealqa), config `seal_0` | `seal0/judge.py` |

Each judge's module docstring states exactly what it grades, which prompt it
uses and where that prompt came from.

## Install

The judges run in the repository environment, `../requirements.txt`.

HiPhO needs three more packages, installed into a separate prefix because the
vendored official grader requires a different antlr4 version than the training
stack. From this directory:

```bash
pip install --target .hipho_deps -r requirements.txt
export PYTHONPATH=$PWD/.hipho_deps        # only needed when grading HiPhO
```

## Configuring the APIs

### The judge

Every judge talks to one OpenAI-compatible endpoint through
`../datagen/utils/llm_client.py`:

```bash
export OPENAI_API_KEY=sk-...          # key used for grading
export OPENAI_MODEL=gpt-5.5           # judge model
```

That is the whole configuration for the OpenAI API. To use another
OpenAI-compatible server, add its base URL:

```bash
export OPENAI_BASE_URL=https://my-endpoint.example.com/v1
```

Two optional hooks exist for gateways that need more than a key:
`OPENAI_QUERY_JSON` adds query parameters and `OPENAI_HEADERS_JSON` adds
headers, both as JSON objects.

Use a key reserved for grading. If the same key also serves the agent being
evaluated, the two compete for one rate limit, and a throttled request can come
back as an empty response rather than an error.

Sampling parameters are command-line flags, not environment defaults:
`--temperature`, `--judge-max-tokens`, `--workers`, `--qpm`.
`python <judge>.py --help` lists them with their defaults, and every value used
is written into the report.

### The agent being evaluated

Rollouts are produced outside this repository, but the four benchmarks that
allow retrieval need these services, so the keys are listed here for
completeness. The harness reads them from the environment:

```bash
export SERPER_KEY_ID=...              # web search
export JINA_API_KEYS=...              # page fetch; comma-separated for several keys
export OPENAI_API_KEY=sk-...          # summarising fetched pages
```

and one Python sandbox, either hosted or self-run:

```bash
export PYTHON_SANDBOX_BACKEND=e2b
export E2B_API_KEY=...
# or
export PYTHON_SANDBOX_BACKEND=sandbox_fusion
export SANDBOX_FUSION_ENDPOINT=http://127.0.0.1:8080
```

Serve the model being evaluated on its own endpoint, separate from the judge:

```bash
vllm serve <model> --enable-prefix-caching --port 8000
```

## Rollout format

Every judge reads one directory holding one file per sampling round,
`iter1.jsonl` ... `iterN.jsonl`, one JSON object per line:

```json
{"filename": "fs_o_0007_abc",
 "question": "...",
 "prediction": "...",
 "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "termination": "answer"}
```

* `filename` -- item key, must match the prepared dataset. HLE uses `id` instead.
* `prediction` -- content of the final `<answer></answer>`, if the harness extracts it.
* `messages` -- the conversation.

At least one of `prediction` and `messages` must carry the answer. Which one a
judge reads is per benchmark and is set by `--answer-source` where it applies;
see `utils/rollouts.py`.

The inference harness is not part of this repository. Anything that writes this
format can be graded.

## FrontierScience (FS-O, FS-R)

```bash
cd frontierscience
python prepare.py                        # -> fs_o.jsonl (100), fs_r.jsonl (60)

python judge.py --bench fs_o --dataset fs_o.jsonl \
    --input-dir <rollout-dir> --rollouts 3
python judge.py --bench fs_r --dataset fs_r.jsonl \
    --input-dir <rollout-dir> --rollouts 3
```

With a local copy instead of a download: `python prepare.py --src <dir>`,
expecting `<dir>/olympiad/test.jsonl` and `<dir>/research/test.jsonl`.

Judge prompts are in `prompts/`, extracted verbatim from the paper's appendix.

Output, written next to the rollouts:

```
judge_fs_o_prediction_report.json
judge_fs_o_prediction_checkpoint.jsonl     per-item verdicts, resumable
```

FS-O reports accuracy. FS-R reports both the official threshold metric
(`mean_pct`) and the continuous rubric mean (`alt_mean_rubric_pct`); the
threshold is `--fsr-threshold`, default 7.0 of 10.

## HiPhO

```bash
cd hipho
python prepare.py                        # -> hipho.jsonl (360), hipho_to.jsonl (102 text-only)

python judge.py --dataset hipho.jsonl \
    --input-dir <rollout-dir> --rollouts 3 --judge-model gemini-2.5-flash
```

With a local clone: `python prepare.py --src /path/to/HiPhO/data`.

Grading runs the official VLMEvalKit implementation, vendored in
`official_impl/` -- see `official_impl/README.md` for the source and how to
refresh it. `--judge-model` selects the model that implementation calls;
the default is the one it uses upstream.

Output: `judge_hipho_official_report.json` plus a checkpoint. Quote
`leaderboard_points_mean`, the mean of the exams' raw point totals;
`score_rate` is also reported and is a different quantity.

## HLE

```bash
cd hle
export HF_TOKEN=hf_...                   # accept the terms at the dataset page first

python prepare.py                        # -> hle_full.jsonl, hle_text_only.jsonl,
                                         #    hle_text_only_130.jsonl
python judge.py --dataset hle_text_only_130.jsonl \
    --input-dir <rollout-dir> --rollouts 3
```

With the parquet already downloaded: `python prepare.py --parquet <file>`.

`hle_text_only_130.jsonl` is built from `subset_130_ids.txt` and is the set
reported here. Output: `judge_hle_report.json` with per-round accuracy and
Best@k, plus a checkpoint.

This grader uses the agent harness's own judge prompt, not the official HLE
grader; `judge.py`'s docstring lists the differences.

## SEAL-0

```bash
cd seal0
python prepare.py --current-year 2026    # -> seal0_<today>.jsonl

python judge.py --dataset seal0_<today>.jsonl \
    --input-dir <rollout-dir> --rollouts 3
```

`--current-year` goes into the prompt, so pass the year inference runs in.
`--prompt from_a1` (default) uses the Agents-A1 reference judge;
`--prompt official` uses the SealQA paper's appendix prompt. Both are in
`prompts/`.

Output: `judge_seal0_from_a1_report.json`, with per-round accuracy,
`Avg. Pass@N`, `Pass@N` and breakdowns by `freshness` and `effective_year`.

Reference answers in SEAL-0 change over time. Keep the prepared snapshot
alongside the rollouts graded against it; grading old rollouts against a fresh
download does not measure the same thing.
