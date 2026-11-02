# Reinforcement learning

This package contains Axion's agent loop, reward calculation, context
condensation, tool services and configuration. The backend is vendored separately
in [`third_party/verl`](../../../third_party/verl/README.md).

The complete launch configuration and matching runtime for the paper's final RL
runs have not been recovered. The source launcher is retained for development;
its defaults are not a validated reproduction recipe. SFT environment
instructions do not apply to RL, and fused forward code alone does not establish
a working 122B training configuration.

## Source layout

| Location | Purpose |
|---|---|
| `agent_loop/` | Multi-turn research trajectories and partial rollouts |
| `reward.py`, `reward_manager.py`, `reward_loop_manager.py` | Reward evaluation and trainer integration |
| `session_algos.py`, `memory.py` | Session training and context condensation |
| `obj_task_eval/`, `openended_task_eval/`, `citation_task_eval/` | Objective, rubric and citation evaluation |
| `tools/` | Search, scholar, webpage and Python tools |
| `config/` | Agent, trainer and service configuration |
| `scripts/` | Retrieval services and FAISS index construction |
| `run_deepresearch_fully_async_megatron.sh` | Source launcher for the fully asynchronous verl trainer |

The launcher continues to call
`verl.experimental.fully_async_policy.fully_async_main`. Axion modules use the
`axion.training.rl` namespace. The retained `deepresearch_main_ppo.py` is part of
the source implementation; it is not the fully asynchronous launcher's entrypoint.

## Development setup

Use a separately prepared RL environment containing compatible PyTorch, CUDA,
TransformerEngine, Megatron, Ray and rollout-engine packages. Backend dependency
metadata is in `third_party/verl`; it is not a lock of the original training
runtime. Install the source packages without replacing that environment:

```bash
# From the repository root, in the prepared RL environment.
pip install --no-deps -e third_party/verl
pip install --no-deps -e .
```

Run the source shell scripts from a checkout. The launcher sets the repository
root and RL directory on `PYTHONPATH`, including for Ray workers. The same source,
model, data and service configuration must be available to all nodes.

Training and validation parquet files are external inputs selected by
`TRAIN_FILE` and `VAL_FILE`. `MODEL_PATH` selects the initial model. No training
data or model checkpoints are bundled here.

## Tool services

Configure service endpoints in `config/search_nodes.conf`,
`config/scholar_nodes.conf`, `config/python_nodes.conf` and
`config/eval_llm_nodes.conf`. Tool settings are in `config/tools.yaml`.
Start retrieval services from the repository root:

```bash
bash axion/training/rl/scripts/run_search_service.sh
bash axion/training/rl/scripts/run_scholar_service.sh
```

The scripts accept `SEARCH_SERVICE_PORT` / `SCHOLAR_SERVICE_PORT` and
`SEARCH_SERVICE_CONFIG` / `SCHOLAR_SERVICE_CONFIG`. FAISS builders are in
`scripts/init_faiss_search.sh` and `scripts/init_faiss_scholar.sh`; model and index
locations must be configured for the deployment. Python execution uses
`PYTHON_SANDBOX_BACKEND` (`sandbox_fusion` or `e2b`) and the corresponding endpoint
or credentials.

Keep API credentials in the environment or the local, untracked
`.secrets/deepresearch_api_keys.env` under `QUEST_ROOT` (the checkout root by
default). Search uses `SERPER_KEY_ID`; webpage retrieval uses `JINA_API_KEYS`.
Reward evaluation, citation evaluation, rubric evaluation, page summarization
and memory condensation have separate environment settings in the launcher:
`EVAL_LLM_*`, `CITATION_EVAL_LLM_*`, `OPENENDED_EVAL_LLM_*`, `VISIT_SUMMARY_*` and
`MEMORY_*`. Configure each service explicitly for the deployment.

## Provenance

Derived from the DeepResearch recipe in QUEST. Modified files retain their
change notices; original license headers remain in place. This reorganization
changes module and resource paths while preserving training logic, prompts and
configuration values. See [third-party notices](../../../THIRD_PARTY.md).
