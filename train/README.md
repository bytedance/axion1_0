# Training

Axion follows two stages of supervised fine-tuning, followed by reinforcement
learning. The released SFT entrypoint uses ms-swift with Megatron-FSDP.

| Component | Location | Guide |
|---|---|---|
| Two-stage SFT | `train/sft/` | [Setup, training and export](sft/README.md) |
| SFT framework | `train/swift/` | [Source and adaptations](swift/README.md) |
| RL framework and tools | `train/verl/` | [Source status](rl/README.md) |

The 122B SFT recipe uses 32 GPUs, 128K
context, global batch 8 and microbatch 1. The same configuration is used for
both stages; Stage II initializes from the Stage I native checkpoint with a
fresh optimizer and schedule. Its runtime patches, input fingerprints and
known environment limits are recorded in the SFT guide.

RL source is included. Reproduction instructions for the paper's final RL
runs are not yet available; see the [RL source status](rl/README.md).

Use separate environments for SFT, RL, and data/evaluation. The root `axion`
package does not install training frameworks or change their dependencies.
