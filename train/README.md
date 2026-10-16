# Training

Two stages, each with its own framework:

| Stage | Framework | Directory |
|---|---|---|
| SFT | LlamaFactory (Megatron backend) | `train/sft/` |
| RL  | verl (fully-async GRPO)         | `train/verl/`, recipes in `train/rl/` |

This page covers RL; SFT has its own README in `train/sft/`.

## RL

### Layout

```
train/rl/
  rl_122b.sh                           recipe: topology, parallelism, batch shape
  rl_35b.sh                            same, for the 35B model

train/verl/recipe/deepresearch/
  run_deepresearch_fully_async_megatron.sh    launcher
  agent_loop/                          multi-turn rollout
  tools/                               search, visit, scholar, python, sandbox server
  obj_task_eval/  openended_task_eval/  citation_task_eval/   graders
  reward.py  memory.py                 reward and context compaction
  config/                              tools.yaml, trainer config, node endpoint files
  scripts/                             search/scholar services, cache maintenance
```

A recipe only exports environment variables. The launcher reads them and starts
Ray, the rollout engines and the trainer.

### Requirements

- A conda environment with the aligned Megatron + verl runtime stack. For MoE
  models it additionally needs Apex, DeepEP/HybridEP, `causal-conv1d` and
  `flash-linear-attention`.
- GPUs on every node, plus one or more nodes serving the judge and summarizer
  models over an OpenAI-compatible API (vLLM).
- A Serper API key (search, scholar) and a Jina Reader key (visit).

### Credentials

Pass keys through the environment, or put them in a file you source. Never put
them on a command line and never commit them.

```bash
# ~/.axion_keys   (chmod 600)
export SERPER_KEY_ID=...        # serper.dev
export JINA_API_KEYS=...        # jina.ai; comma-separate several keys
```

### Input data

A parquet file with one row per prompt:

| Column | Contents |
|---|---|
| `data_source` | which generator produced the item |
| `prompt` | `[{"role": "user", "content": ...}]` |
| `reward_model` | `{"ground_truth": {"ref_answer": ..., "answer": ..., "task_id": ..., "type": "obj"}}` |
| `extra_info` | `{"answer_type": ..., "index": ..., "question": ...}` |
| `rl_task_category` | `objective` or `openended` |

`datagen/` produces the questions; see its README for the conversion.

### Services

Start these before training.

**Sandbox** — the endpoint behind `PythonInterpreter`. Run it on every node that
hosts rollout workers; it writes its endpoint into
`config/python_nodes.conf`, which the python tool hot-reloads.

```bash
cd train/verl
export SANDBOX_RO_ROOT=/data,/models     # remounted read-only inside the sandbox
bash recipe/deepresearch/tools/start_local_sandbox.sh
bash recipe/deepresearch/tools/verify_sandbox.sh      # optional check
```

`SANDBOX_RO_ROOT` lists the shared-storage mounts to remount read-only for
executed code; the script refuses to start without it. Set
`SANDBOX_PORT`, `SANDBOX_WORKERS` or `SANDBOX_PYTHON` to override the port
(9000), the concurrency (64) or the interpreter.

The server accepts loopback only. For a multi-node run, name the networks the
other trainer nodes come from and the address to advertise to them:

```bash
export SANDBOX_ALLOW=10.0.0.0/8          # added to loopback, not swapped for it
export ADVERTISE_HOST=10.1.2.3           # omit to let the script detect it
```

**Search and scholar** — one instance each, on the head node.

```bash
bash recipe/deepresearch/scripts/run_search_service.sh
bash recipe/deepresearch/scripts/run_scholar_service.sh
```

**Judge and summarizer** — serve them with vLLM and point the recipe at them:

```bash
export JUDGE_MODEL_NAME=...        JUDGE_BASE_URLS=http://<host>:8000/v1
export SUMMARY_MODEL_NAME=...      SUMMARY_BASE_URLS=http://<host>:8001/v1
```

Comma-separate several base URLs to spread load across engines.

### Launch

Run the same three commands on every node:

```bash
cd train/verl

export MODEL_PATH=/path/to/sft/checkpoint
export DATASET=/path/to/prompts.parquet
export RAY_DATA_HOME=/path/for/checkpoints/and/rollouts
source ~/.axion_keys

source ../rl/rl_122b.sh                 # or rl_35b.sh
bash recipe/deepresearch/run_deepresearch_fully_async_megatron.sh
```

For multi-node, start Ray first and set `RAY_ADDRESS=auto` (or
`<head_ip>:6379`); the default `local` runs single-node.

Sourcing the recipe prints the topology it will use:

```
[rl_122b] topology 5+2 nodes; rollout TP4; trainer TP2 PP2 EP8; 50 steps, save every 5 versions
```

Override any of it from the environment, for example
`ROLLOUT_NNODES=3 TRAINER_NNODES=1 source ../rl/rl_122b.sh`.

### Checkpoints

Written under `$RAY_DATA_HOME` as `savefreq_step_<parameter_version>`. A
parameter version spans `ASYNC_TRIGGER_PARAMETER_SYNC_STEP` training steps, so
with the default of 2, `savefreq_step_5` is training step 10.

`latest_checkpointed_iteration.txt` holds the newest version. On shared storage
read it with `dd iflag=direct` so you do not get a cached copy.

### Cache maintenance

Search, scholar and visit results are cached in sqlite shards. After a run,
merge them back into the shared pool and rebuild the semantic index:

```bash
python recipe/deepresearch/scripts/merge_serper_cache.py --pool <pool> --sources <run-dir>...
python recipe/deepresearch/scripts/merge_visit_cache.py  --pool <pool> --nodes   <node-dir>...
python recipe/deepresearch/scripts/build_faiss_pool.py   --pool <pool>
```

Two concurrent runs must not share a cache directory: give each its own
`DATASET_TAG`.
