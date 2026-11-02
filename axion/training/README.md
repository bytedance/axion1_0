# Training

Axion trains in three successive stages: **SFT I → SFT II → RL**.
Swift with Megatron-FSDP implements both SFT stages. verl implements asynchronous
RL. Both backends are needed for the complete pipeline; they are not alternative
training recipes.

| Stage | Axion code | Backend | Guide |
|---|---|---|---|
| SFT I and II | `axion/training/sft/` | `third_party/swift/` | [Setup, training and export](sft/README.md) |
| RL | `axion/training/rl/` | `third_party/verl/` | [Agent training source and status](rl/README.md) |

The single released SFT recipe is for Qwen3.5-122B-A10B: **32 GPUs, 128K context,
global batch 8, microbatch 1**. Stage II initializes from the Stage I native
checkpoint with a fresh optimizer and schedule. The SFT guide records the
runtime patches, input fingerprints and differences between the saved arguments
and the paper.

RL source includes agent interaction, rewards, context condensation and tool
services. The final RL launch configuration and matching runtime have not been
recovered, so this release does not provide a validated RL reproduction command.

Use separate environments for SFT, RL, and data/evaluation. The root package
contains Axion code and resources; the backend distributions retain their
`swift` and `verl` import names and separate dependencies.
