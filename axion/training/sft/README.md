# SFT

The reference two-stage Qwen3.5-122B-A10B configuration uses **32 GPUs (4 × 8),
131072-token packing, TP1 / PP1 / EP8 / CP4, microbatch 1, global batch 8**.
Both stages use full BF16 SFT, three epochs, constant LR 3e-6, and assistant-only
loss across all turns. Stage1 saves every 160 steps; stage2 saves at its final
step (interval 1000000). Optimizer and RNG states are saved; retention is 100.
The reference exports are stage1 iteration 795 and stage2 iteration 894.

## Environment

Use a dedicated **Python 3.12** environment with a compatible GPU stack already
installed: PyTorch, CUDA, TransformerEngine, FlashAttention and FLA. The complete
versions of that original stack were not recovered. The available records show
Megatron-core 0.17.1, mcore-bridge 1.4.3 and Transformers 5.10.4 at export; the
requirements file pins these observed versions, not a complete environment lock.
It also includes the source-declared Qwen3.5 loader dependencies
`qwen-vl-utils>=0.0.14` and `decord`; their original versions were not recovered.

From the repository root, on every training node:

```bash
pip install -r third_party/swift/requirements.txt -r axion/training/sft/requirements-runtime.txt
pip install --no-deps -e third_party/swift
pip install -e ".[training]"
python -m axion.training.sft.runtime --apply
```

The launcher checks the installed Swift files against [swift_files.json](swift_files.json).

The last command applies the original GDN CP backport to the two recognized
Megatron-core files in this environment. It checks file hashes, is idempotent,
and requires a writable environment. Do not install
the MCore `training` extra: its Transformers constraint conflicts with the
recorded export stack. The verified 0.17.1 wheel already contains the needed
`megatron.training.checkpointing` module.

Prepare CUDA libraries and NCCL networking for your cluster on every node. The
launcher retains `MS_SWIFT_SKIP_TE_PRELOAD=1`, the original conditional CUDA 13
cuBLAS preload (`CUBLAS_LIB_DIR` can specify its directory), and CUDA connections
1 before Swift initializes FSDP and changes it to 8. It clears the original
profiling/cache variables. Set NCCL interfaces and RDMA options for your cluster;
explicit `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` values are preserved.

Swift includes the paired CP/FSDP guard change, eight-thread synchronous DCP
writer and single-process export fix. **Lifting the Swift guard alone is not
sufficient**; the launcher checks the full MCore backport before training/export.
The imported development snapshot is not proven identical to the original mutable
training image. [provenance.json](provenance.json) records this distinction.

## Data

Each JSONL row needs `messages`, `total_tokens` measured by the actual Swift
training template, and `visual_tokens: 0`. Data must already be filtered to 128K.
Run the length encoder after `to_sft` to provide the metadata used for packing
and length filtering.

For new inputs, use the training-template encoder with the same base model:

```bash
python -m axion.training.sft.tools_template_lengths raw.jsonl annotated.jsonl \
  --model /path/to/Qwen3.5-122B-A10B --cap 131072 --workers 16
```

The reference inputs are not redistributed. Their hashes and counts are in
[provenance.json](provenance.json): stage1 has 9840 conversations and 277397139
template tokens; stage2 has 7544 conversations and 312161019 template tokens.
Use the corresponding annotated inputs to reproduce those runs. The launcher
validates the metadata; the length encoder computes the token counts.

## Train

The default layout is four nodes with eight GPUs each. Run the commands on
every node, setting `RANK` to 0, 1, 2 or 3 and `HEAD` to a reachable head-node
address. All ranks need the same model, data and shared output paths. Allow
traffic to the rendezvous port (29500 by default) and configure NCCL networking.
A native checkpoint can occupy about 1.1 TB; allow space for retained checkpoints.

```bash
python -m axion.training.sft stage1 \
  --model /path/to/Qwen3.5-122B-A10B \
  --dataset /path/to/stage1-annotated.jsonl --output /shared/sft-stage1 \
  --master-addr "$HEAD" --node-rank "$RANK"

python -m axion.training.sft stage2 \
  --model /path/to/Qwen3.5-122B-A10B \
  --dataset /path/to/stage2-annotated.jsonl --output /shared/sft-stage2 \
  --mcore-model /shared/sft-stage1/checkpoint-795 \
  --master-addr "$HEAD" --node-rank "$RANK"
```

For one eight-GPU host, add `--nnodes 1 --nproc-per-node 8`. The node rank
and rendezvous address default to 0 and localhost in this layout:

```bash
python -m axion.training.sft stage1 \
  --model /path/to/Qwen3.5-122B-A10B \
  --dataset /path/to/stage1-annotated.jsonl --output /shared/sft-stage1 \
  --nnodes 1 --nproc-per-node 8
```

Use the same topology options for Stage II. For two eight-GPU hosts, use
`--nnodes 2 --nproc-per-node 8` with ranks 0 and 1 and a shared `--master-addr`.

| Layout | Attention data parallelism | Gradient accumulation | Global batch |
|---|---:|---:|---:|
| 1 × 8 GPUs | 2 | 4 | 8 |
| 2 × 8 GPUs | 4 | 2 | 8 |
| 4 × 8 GPUs | 8 | 1 | 8 |

TP1, PP1, EP8, CP4, microbatch 1 and the learning settings are unchanged.
Smaller layouts need more memory per GPU for model and optimizer shards.
Memory capacity for a 122B model at 128K context must be checked on the target
hardware; topology validation alone does not establish that it fits.
`CUDA_VISIBLE_DEVICES` may select the local devices; it must provide enough
distinct entries for `--nproc-per-node`. Any inherited `NNODES`,
`NPROC_PER_NODE` and `WORLD_SIZE` must match the selected layout.

Add `--dry-run` to print the complete command/config without loading the model,
patching an environment or launching workers. `common.yaml` is shared by both
stages. Stage2 accepts a native `fsdp_dtensor` checkpoint, **not an HF export**;
`finetune=true` loads weights and starts a new optimizer/schedule. This entrypoint
does not resume an interrupted stage's optimizer.

## Export

Use the prepared SFT environment, one GPU and enough host RAM for the full model
(the original exporter estimated about 245 GB just for BF16 parameters). Export
outside the training directory:

```bash
python -m axion.training.sft export \
  --model /path/to/Qwen3.5-122B-A10B \
  --mcore-model /shared/sft-stage2/checkpoint-894 --output /shared/sft-stage2-hf
```

This uses DCP-direct export with `MS_SWIFT_EXPORT_NO_FSDP=1`, resets parallelism
and sequence parallelism, and preserves the checkpoint vocabulary.
The explicit `--model` takes precedence over a model path stored in the native
checkpoint and over inherited `BASE_MODEL`, so checkpoints can move between hosts.

Export fails if model parameters are missing or if the saved safetensors
shards or index are incomplete. Serve the resulting directory using the
[inference guide](../../evaluation/README.md#serve-a-checkpoint).

## Validation

Tests cover launch configuration, topology constraints, runtime patches,
checkpoint validation and package integrity. GPU training, export numerical
parity and paper benchmark results have not been revalidated for this release.
Source and environment records are in [provenance.json](provenance.json).
