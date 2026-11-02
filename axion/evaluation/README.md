# Evaluation

Dataset preparation and rollout grading for FrontierScience Olympiad,
FrontierScience Research, HiPhO, HLE-130 and SEAL-0.

Each benchmark has a `prepare.py` that downloads and converts its source
data. The repository includes only the HLE subset IDs in
`hle/subset_130_ids.txt`; benchmark questions and answers are downloaded separately.

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

From the repository root, install the evaluation dependencies in a separate
environment:

```bash
python -m pip install -e ".[evaluation]"
cd axion/evaluation
```

Each benchmark section below starts from `axion/evaluation/`. For module
execution, for example `python -m axion.evaluation.hle.judge`, pass explicit
dataset and output paths. The preparation scripts otherwise write next to
their source files.

HiPhO needs three more packages, installed into a separate prefix because the
vendored official grader requires a different antlr4 version than the training
stack. From this directory:

```bash
pip install --target .hipho_deps -r requirements.txt
export PYTHONPATH=$PWD/.hipho_deps        # only needed when grading HiPhO
```

## Serve a checkpoint

Use the Hugging Face directory produced by [SFT export](../training/sft/README.md#export).
It must contain the weights, model configuration and tokenizer, including its
chat template. Native Megatron checkpoints need conversion first.

Prepare a separate vLLM environment using the
[Qwen deployment guide](https://recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B).
For an eight-GPU server:

```bash
vllm serve /shared/sft-stage2-hf \
  --served-model-name axion-122b \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 8 --dtype bfloat16 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Choose tensor parallelism and context capacity for the available GPU memory.
For remote clients, bind to a reachable interface on the service network.
In another terminal, check the model ID and send a request:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"axion-122b","messages":[{"role":"user","content":"What is 2 + 3?"}],"max_tokens":1024}'
```

This checks model serving. Benchmark rollouts additionally require an agent
harness that runs tools and writes the format below. GPU serving and paper
benchmark results have not been revalidated for this release.

## Configuring the APIs

### The judge

Every judge talks to one OpenAI-compatible endpoint through
`../datagen/utils/llm_client.py`:

```bash
export OPENAI_API_KEY=sk-...          # key used for grading
export OPENAI_MODEL=gpt-5.5           # judge model
```

For another OpenAI-compatible service, set its base URL:

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

The external agent harness needs retrieval services for tool-enabled
evaluation. Configure the service credentials it uses:

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
export SANDBOX_FUSION_ENDPOINT=http://127.0.0.1:9000/run_code
```

To run that endpoint yourself, start the bundled
[sandbox server](../training/rl/tools/start_local_sandbox.sh)
on the same port:

```bash
(
  cd ../training/rl
  SANDBOX_RO_ROOT=/data SANDBOX_PORT=9000 \
    bash tools/start_local_sandbox.sh
)
```

Serve the evaluated model on a separate endpoint from the judge, as described
in [Serve a checkpoint](#serve-a-checkpoint). Tool calls returned by the model
must be executed by the harness and added to the next request as tool messages.

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
