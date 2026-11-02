# Training

Axion trains in three stages: **SFT I → SFT II → RL**. Swift with
Megatron-FSDP runs SFT; verl runs asynchronous RL.

| Stage | Code | Backend | Guide |
|---|---|---|---|
| SFT I and II | `axion/training/sft/` | `third_party/swift/` | [Setup, launch and export](sft/README.md) |
| RL | `axion/training/rl/` | `third_party/verl/` | [Tasks, services and launch](rl/README.md) |

The reference SFT configuration uses Qwen3.5-122B-A10B with **32 GPUs, 128K
context, global batch 8 and microbatch 1**. The launcher also accepts compatible
8- and 16-GPU layouts. Stage II loads the Stage I native checkpoint and starts
a new optimizer and schedule.

RL uses asynchronous GRPO with four-attempt task selection, session training,
TITO and routing replay. Trainer and rollout workers use separate GPU pools,
which can share a host when it has enough resources.

Use separate environments for SFT, RL and inference. Each guide lists its
runtime requirements, launch commands and validation limits. The root package
contains Axion code; the backends keep their `swift` and `verl` import names.
