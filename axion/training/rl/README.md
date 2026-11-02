# Reinforcement learning

One RL recipe continues from the **Qwen3.5-122B-A10B Stage II checkpoint**.
It implements Sections 3.3 and 5.2 and Table 5 of the
[Axion report](https://axion1-0.github.io/paper/axion.pdf).
Axion code is in this package; the asynchronous Megatron/vLLM backend is in
[`third_party/verl`](../../../third_party/verl/README.md).

## Training recipe

| Setting | Value |
|---|---|
| Algorithm | Asynchronous GRPO, full-parameter BF16 training |
| Rollouts per prompt | 8, temperature 1.0 |
| Optimizer batch | 24 prompt groups with nonzero reward variance |
| Learning rate | 2e-6, constant after 5 warmup updates |
| Adam | Betas (0.9, 0.999), weight decay 0.1, gradient clipping 1.0 |
| PPO clipping | Lower 0.2, upper 0.28 |
| KL loss / entropy | 0.001 / 0 |
| Session limits | 12,000 prompt tokens, 49,152 response tokens |
| Weight synchronization | Every 2 optimizer updates |
| Staleness limit | 11 published parameter versions |
| Rollout / trainer GPUs | 40 (TP4) / 16 (TP2, EP8), with CPU optimizer offload |

Advantages are normalized over the eight original trajectories before context
condensation sessions are expanded. Each session inherits its trajectory's
reward and advantage; loss is normalized per session. TITO retains the exact
token IDs and policy masks. R3 records MoE routes as `uint8` during generation
and replays them in the actor. Groups containing stale trajectories or failed
reward evaluations are excluded. Initial prompts are filtered using the complete
system-plus-question template. If a condensed continuation would exceed the
prompt limit, the rollout ends at its current valid session; it does not truncate
the question or create an over-limit training session.

The reward follows Equations 4–5: a correct final answer receives 1; otherwise
it is `max(0, 0.3 * P - 0.1 * (1 - A))`, where `A` indicates a submitted final
answer and `P` is the fraction of consecutive verified intermediate targets.
Numerical tolerances and targets are explicit task data. Tool observations and
reasoning drafts do not count as submitted answers.

## Environment

Use a separate RL environment with CUDA, PyTorch, Megatron-Core,
TransformerEngine, mbridge, vLLM, Ray and verl's Python dependencies. The runtime
must support Qwen3.5 MoE, GDN and routing replay. The launch preflight checks
these APIs and requires **vLLM >= 0.22.0** for hybrid-attention route capture.
Install this checkout's packages into that prepared environment:

```bash
python -m pip install --no-deps -e third_party/verl
python -m pip install -e '.[training,rl-data]'
```

Do not install the vendored `verl[vllm]` extra: its upstream version range
predates the R3 support required here. The launcher uses the legacy mbridge
backend and records installed versions and capability checks in the run output.
These checks do not establish a complete compatible environment or validate a
GPU run. The original final RL image has not been recovered.

The current BSHD training path does not support the vendored fused LM-head
projection, so that fusion is disabled. Memory use must be checked on the target
hardware. SFT environment instructions do not apply to RL.

## Prepare tasks

The selector takes canonical tasks with `id`, `question`, `reference` and
ordered intermediate `hops`. Numeric tolerances must be explicit, including
zero for exact matching. See the schema and examples in
[`prepare_rl.py`](../../datagen/stage2/pipeline/prepare_rl.py), also available
through `python -m axion.datagen.stage2.pipeline.prepare_rl --help`.

First exclude held-out and defective questions. Both manifests contain
`task_ids` and `questions` lists; include reviewed near-duplicate exclusions.
Then sample **four complete trajectories per remaining task from one checkpoint**
using your agent harness. Give the model only the question, without references
or intermediate targets. Record the checkpoint identity and all role-tagged
messages in each attempt.

```bash
python -m axion.datagen.stage2.pipeline.prepare_rl candidates \
  --tasks tasks.jsonl --heldout heldout.json --defects defects.json \
  --out candidates.jsonl

# Collect four attempts per candidate into attempts.jsonl, then score them.
# Set S2_JUDGE_API_KEY in the environment for the OpenAI-compatible judge.
python -m axion.datagen.stage2.pipeline.prepare_rl select \
  --tasks candidates.jsonl --attempts attempts.jsonl \
  --heldout heldout.json --defects defects.json \
  --model-checkpoint CHECKPOINT_ID \
  --judge-address http://JUDGE_HOST:PORT/v1 --judge-model Qwen3.8-27B \
  --out train.parquet
```

Selection requires **at least one correct final answer and nonzero reward
variance**. The generated `train.selection.json` records scores, exclusions and
input hashes; keep it beside `train.parquet`. Training verifies the parquet hash
and selection rule. These four offline attempts are separate from the eight
rollouts per prompt used during RL.

## Configure and launch

Copy [`config/deployment.example.yaml`](config/deployment.example.yaml) to a
local deployment file and fill its null values. The paper does not specify the
condensation threshold, turn limit or per-turn tool/response limits, so these
are required inputs. Model, data, output, source and environment paths must be
accessible on every Ray node.

Deploy search, scholarly retrieval and Sandbox Fusion services, plus
OpenAI-compatible endpoints for the paper's Qwen3.8-27B judge and Qwen3.5-9B
summarization/condensation models. Service replicas run outside the training and
rollout pools. Set the three API-key variables named in the deployment file;
credentials are passed to workers without being saved in the resolved config.
Webpage retrieval also uses `JINA_API_KEYS` or `JINA_API_KEY`.

Retrieval service helpers are in `scripts/run_search_service.sh` and
`scripts/run_scholar_service.sh`; configure their indexes and upstream retrieval
credentials before starting them. Point the deployment URLs to the running
services. The launcher uses these URLs rather than bundled localhost defaults.

Start the Ray cluster in the prepared environment, then run from the repository
root. Set `RL_UPDATES` to the desired positive, even number of optimizer updates;
the paper does not report the final count.

```bash
bash axion/training/rl/train.sh \
  --model /shared/checkpoints/stage2-hf \
  --train-data /shared/data/train.parquet \
  --deployment /shared/config/deployment.yaml \
  --output /shared/runs/axion-122b-rl \
  --updates "${RL_UPDATES:?Set the optimizer-update count}" \
  --ray-address auto
```

Add `--dry-run` to inspect the resolved configuration without starting workers
or contacting services. GPU counts and parallelism can be adjusted in the
deployment file; the algorithm settings remain in [`config/paper.yaml`](config/paper.yaml).
RL uses context parallelism 1 and microbatch 1 for the current per-session
BSHD path. Each microbatch pads only its own session, avoiding batch-wide
padding of short sessions to the longest trajectory.

`--output` must be empty. This launcher starts from Stage II; exact resumption
of the asynchronous queue and in-flight generations is not supported. It saves
checkpoints, the resolved config and `run_provenance.json`. The optional
`--rollout-attempts` bounds attempted prompt groups; its default is
`updates * 24 * 100`, an execution guard rather than a paper hyperparameter.
Exhausting it before the requested updates raises an error.

## Validation and provenance

CPU tests cover reward and selection rules, trajectory-to-session conversion,
padding masks, update/version accounting, routing alignment and real worker
config construction. They do not validate distributed training, memory fit or
benchmark reproduction. Full GPU validation remains required before claiming
reproduction of the paper's results. To run the CPU checks in a separate test
environment, install `.[test,rl-test]` and the vendored backend with `--no-deps`,
then run `python -m pytest -q` from the repository root.

Source revisions and the recovered reward implementation are recorded in
[`provenance.json`](provenance.json). The implementation follows the paper;
it is not a recovered copy of the final RL launch configuration. SFT is unchanged.
