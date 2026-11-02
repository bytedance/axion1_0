# Training

Axion trains in three successive stages: **SFT I → SFT II → RL**.
Swift with Megatron-FSDP implements both SFT stages. verl implements asynchronous
RL. Both backends are needed for the complete pipeline; they are not alternative
training recipes.

| Stage | Axion code | Backend | Guide |
|---|---|---|---|
| SFT I and II | `axion/training/sft/` | `third_party/swift/` | [Setup, training and export](sft/README.md) |
| RL | `axion/training/rl/` | `third_party/verl/` | [Paper recipe, data selection and launch](rl/README.md) |

The single released SFT recipe is for Qwen3.5-122B-A10B: **32 GPUs, 128K context,
global batch 8, microbatch 1**. Stage II initializes from the Stage I native
checkpoint with a fresh optimizer and schedule. The SFT guide records the
runtime patches, input fingerprints and differences between the saved arguments
and the paper.

The RL guide provides one paper-based asynchronous GRPO recipe, including
four-attempt task selection, session training, TITO and R3. The original final
RL runtime has not been recovered; distributed GPU training and benchmark
reproduction remain unverified. Unreported deployment settings are explicit
launch inputs.

Use separate environments for SFT, RL, and data/evaluation. The root package
contains Axion code and resources; the backend distributions retain their
`swift` and `verl` import names and separate dependencies.
