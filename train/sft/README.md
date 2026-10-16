# SFT

LlamaFactory with the Megatron (mca) backend.

Training runs in two stages: stage 1 from the base model on the stage-1
trajectories, stage 2 continuing from the stage-1 checkpoint on the stage-2
trajectories. The RL recipe starts from the stage-2 checkpoint.

```
train/sft/
  LlamaFactory/      vendored upstream, with one patch (see below)
  sft_stage1.yaml    stage 1 config
  sft_stage2.yaml    stage 2 config
  check_data.py      verifies the data survives the pipeline
  data/              dataset_info.json; put the data here
```

## Install

```bash
pip install -r requirements.txt            # from the repository root
pip install -e train/sft/LlamaFactory
```

The environment is newer than the versions LlamaFactory declares, so set
`DISABLE_VERSION_CHECK=1` when running it.

### The one patch

`LlamaFactory/src/llamafactory/train/mca/workflow.py` adds
`_pad_to_tp_multiple_wrapper`, which pads each collated batch up to a multiple
of the tensor-parallel size.

It is needed because the multimodal collator appends a placeholder image to a
batch that contains none, which this text-only data does on every step. The
extra visual tokens make a batch that already fills `cutoff_len` overflow to an
odd length, and `sequence_parallel` then fails an assertion that the sequence
divides evenly across ranks. Without the patch, training crashes as soon as a
batch is long enough.

## Prepare the data

Put the two trajectory sets at `train/sft/data/stage1.jsonl` and
`train/sft/data/stage2.jsonl`, one conversation per line:

```json
{"messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}
```

Each line must hold exactly one conversation, with a single leading system
message and strict user/assistant alternation after it. LlamaFactory drops a
row that has a second system message mid-conversation, or two messages with
the same role in a row.

`datagen/` writes this format. The names `stage1` and `stage2` are registered
in `train/sft/data/dataset_info.json`; change them there and in the configs if
you use different filenames.

Then verify that nothing is dropped:

```bash
python train/sft/check_data.py \
    --model /path/to/base/model \
    --data-dir train/sft/data \
    --datasets stage1 stage2
```

It compares the number of conversations the pipeline yields against the number
of lines in the input and exits non-zero on any mismatch. A conversation lost
to an invalid role tag, or to being longer than `cutoff_len`, is otherwise
reported only as a warning that is easy to miss in the training log.

## Train

```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli train train/sft/sft_stage1.yaml
DISABLE_VERSION_CHECK=1 llamafactory-cli train train/sft/sft_stage2.yaml
```

Set `model_name_or_path`, `dataset` and `output_dir` in each config first. For
stage 2, `model_name_or_path` is a stage-1 checkpoint directory in the mca
format, not the `_hf` export.

### Scaling

`expert_model_parallel_size` must divide `DP * TP`, where
`DP = world_size / (tensor_model_parallel_size * pipeline_model_parallel_size)`.
With the shipped `TP=2, PP=1, EP=8`, the world size must be a multiple of 8.

Three settings are not free to change:

- `sequence_parallel: true` is required. This model has a large vocabulary and
  cross-entropy materialises the logits in fp32; at this sequence length they
  do not fit on one device.
- Keep `tensor_model_parallel_size` at 2. `num_query_groups` is 2, so a larger
  TP takes the compensation path in megatron-core's attention, which re-slices
  the query heads but not the output gate. The gate then has more heads than
  the attention output and the reshape in `_apply_output_gate` raises.
- `context_parallel` must stay off. Most layers are gated_delta_net, which
  requires `context_parallel_size == 1`.

Together those cap `cutoff_len`: the sequence can only be split along TP, TP
cannot exceed 2, and the fp32 logits grow linearly with the sequence. 98304 is
the longest that fits. Adding GPUs does not help, because that raises DP and
leaves the per-device sequence length unchanged.

`optimizer_cpu_offload` is off. The expert optimizer state shards across
`DP * TP / EP` ranks, which is enough here. Turn it on if it does not fit; it
costs throughput.

### Two settings that silently change what is trained

| Setting | Keep | If changed |
|---|---|---|
| `enable_thinking` | `true` (default) | `false` strips every `<think>` from the data |
| `mask_history` | `false` (default) | `true` trains only the last turn of each conversation |

### Packing

Stage 2 sets `packing: true`; stage 1 does not. With expert parallelism the
collator pads every batch to `cutoff_len`, so without packing most of each
batch is padding.

Use plain packing, not `neat_packing` — the argument parser rejects
`neat_packing` on transformers >= 4.53, and it is not needed here because
`position_ids` restart at each sample, which is what flash-attention uses to
keep samples from attending across the boundary.

Packing produces fewer, longer sequences, so a given number of epochs runs
fewer optimizer steps than an unpacked run. Raise
`gradient_accumulation_steps` if you want the step count to match.
