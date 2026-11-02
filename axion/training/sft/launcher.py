#!/usr/bin/env python3
"""Launch the recorded two-stage SFT recipe on a compatible GPU topology, or export its checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import sys

import yaml

HERE = Path(__file__).resolve().parent
NNODES, GPUS_PER_NODE = 4, 8
MODEL_SHAPE = {'hidden_size': 3072, 'num_hidden_layers': 48, 'num_experts': 256,
               'num_experts_per_tok': 8, 'linear_num_key_heads': 16, 'linear_num_value_heads': 64,
               'num_attention_heads': 32, 'num_key_value_heads': 2}


def validate_model(directory):
    path = Path(directory).expanduser().resolve()
    if not (path / 'config.json').is_file():
        raise ValueError('--model must be a local Hugging Face Qwen3.5-122B base-model directory.')
    config = json.loads((path / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5_moe' or any(
            config.get('text_config', {}).get(key) != value for key, value in MODEL_SHAPE.items()):
        raise ValueError('--model does not match the recorded Qwen3.5-122B-A10B architecture.')
    return path


def validate_native_checkpoint(directory):
    path = Path(directory).expanduser().resolve()
    required = [path / 'args.json', path / 'latest_checkpointed_iteration.txt']
    if not all(item.is_file() for item in required):
        raise ValueError('--mcore-model must be a native Megatron checkpoint-N directory, not an HF export.')
    try:
        iteration = int(required[1].read_text().strip())
    except ValueError as exc:
        raise ValueError('The native checkpoint iteration tracker is invalid.') from exc
    if iteration < 1 or not (path / f'iter_{iteration:07d}' / '.metadata').is_file():
        raise ValueError('The native checkpoint is missing its iteration directory or DCP .metadata.')
    config = json.loads(required[0].read_text())
    expected = {'tensor_model_parallel_size': 1, 'pipeline_model_parallel_size': 1,
                'expert_model_parallel_size': 8, 'context_parallel_size': 4, 'ckpt_format': 'fsdp_dtensor'}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('The native checkpoint must use the recorded TP1/PP1/EP8/CP4 fsdp_dtensor format.')
    return path


def validate_dataset(filename):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise ValueError('--dataset must point to the annotated JSONL input.')
    count = 0
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                tokens = row.get('total_tokens')
                valid = (isinstance(row.get('messages'), list) and bool(row['messages'])
                         and type(tokens) is int and 0 < tokens <= 131072
                         and row.get('visual_tokens') == 0)
            except (ValueError, AttributeError):
                valid = False
            if not valid:
                raise ValueError(f'Invalid training-template length metadata at JSONL line {number}; '
                                 'run tools_template_lengths.py and use its filtered output.')
            count += 1
    if count == 0:
        raise ValueError('Training data is empty; no valid length metadata was found.')
    return path


def as_flags(config):
    flags = []
    for key, value in config.items():
        if value is None:
            continue
        flags.append('--' + key)
        if isinstance(value, list):
            flags.extend(str(item) for item in value)
        else:
            flags.append(str(value).lower() if isinstance(value, bool) else str(value))
    return flags


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def safetensors_parameters(filename):
    """Validate tensor byte ranges from the header and file size, without reading weights."""
    filename = Path(filename)
    try:
        size = filename.stat().st_size
        with filename.open('rb') as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError('missing header length')
            header_size = struct.unpack('<Q', prefix)[0]
            if not 0 < header_size <= min(size - 8, 100_000_000):
                raise ValueError('invalid or truncated header length')
            header = json.loads(handle.read(header_size), object_pairs_hook=unique_json_object)
        if not isinstance(header, dict):
            raise ValueError('header must be an object')
        # Standard byte-aligned safetensors dtypes. The released exporter writes BF16.
        dtype_bytes = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
                       'F8_E8M0': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
                       'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
        parameters, ranges = {}, []
        for key, value in header.items():
            if key == '__metadata__':
                if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
                    raise ValueError('invalid metadata')
                continue
            if not isinstance(value, dict):
                raise ValueError(f'invalid tensor metadata for {key}')
            shape, dtype, offsets = value.get('shape'), value.get('dtype'), value.get('data_offsets')
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError(f'invalid shape for {key}')
            if not isinstance(dtype, str) or dtype not in dtype_bytes:
                raise ValueError(f'unsupported dtype for {key}: {dtype}')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int or offset < 0 for offset in offsets)):
                raise ValueError(f'invalid offsets for {key}')
            count = 1
            for dimension in shape:
                count *= dimension
            start, end = offsets
            if end - start != count * dtype_bytes[dtype]:
                raise ValueError(f'tensor shape/dtype does not match byte range for {key}')
            parameters[key] = count
            ranges.append((start, end))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError('overlapping tensor ranges or a gap in the payload')
            cursor = end
        if cursor != size - 8 - header_size:
            raise ValueError('truncated payload or trailing bytes')
        return parameters
    except (ValueError, UnicodeError) as error:
        raise ValueError(f'Invalid safetensors file {filename.name}: {error}') from error


def verify_export(directory):
    """Check complete shards and their HF index, then retain the recorded parameter-count check."""
    directory = Path(directory)
    files = sorted(directory.glob('*.safetensors'))
    parameters, weight_map = 0, {}
    for filename in files:
        tensors = safetensors_parameters(filename)
        for key, count in tensors.items():
            if key in weight_map:
                raise ValueError(f'Duplicate exported tensor across safetensors shards: {key}')
            weight_map[key] = filename.name
            parameters += count
    index_path = directory / 'model.safetensors.index.json'
    if len(files) > 1 and not index_path.is_file():
        raise ValueError('HF export is missing its safetensors shard index.')
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(), object_pairs_hook=unique_json_object)
        except ValueError as error:
            raise ValueError(f'Invalid HF safetensors shard index: {error}') from error
        if not isinstance(index, dict) or index.get('weight_map') != weight_map:
            raise ValueError('HF safetensors shard index does not match the exported tensors and files.')
    if parameters <= 120_000_000_000 or not all(
            (directory / name).is_file() for name in ['config.json', 'tokenizer.json']):
        raise ValueError('HF export is incomplete: expected more than 120B parameters plus config/tokenizer.')
    return {'safetensors_shards': len(files), 'parameters': parameters}


def visible_devices(processes):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(str(i) for i in range(processes)))
    selected = [device.strip() for device in devices.split(',')]
    if (len(selected) < processes or len(set(selected)) != len(selected)
            or any(not device or device.startswith('-') for device in selected)):
        raise ValueError(f'CUDA_VISIBLE_DEVICES must select at least {processes} distinct GPU devices.')
    return devices


def training_topology(args, config):
    if args.nnodes < 1 or args.nproc_per_node < 1:
        raise ValueError('--nnodes and --nproc-per-node must be positive integers.')
    world_size = args.nnodes * args.nproc_per_node
    # MCore 0.17.1 uses separate attention (TP/CP/DP/PP) and expert (ETP/EP/DP/PP)
    # rank generators. CP is not multiplied into the expert group's world size.
    attention_size = (config['tensor_model_parallel_size'] * config['pipeline_model_parallel_size']
                      * config['context_parallel_size'])
    expert_size = (config['expert_tensor_parallel_size'] * config['expert_model_parallel_size']
                   * config['pipeline_model_parallel_size'])
    if world_size % attention_size or world_size % expert_size:
        raise ValueError(f'{world_size} GPUs cannot form the recorded TP1/PP1/EP8/CP4 groups; '
                         'use 8, 16, or 32 GPUs.')
    data_parallel_size = world_size // attention_size
    micro_batch = config['micro_batch_size'] * data_parallel_size
    if config['global_batch_size'] % micro_batch:
        raise ValueError(f'{world_size} GPUs cannot preserve global batch 8 with microbatch 1 and CP4; '
                         'use 8, 16, or 32 GPUs.')
    for key, expected in {'NNODES': args.nnodes, 'NPROC_PER_NODE': args.nproc_per_node,
                          'WORLD_SIZE': world_size}.items():
        if key in os.environ and os.environ[key] != str(expected):
            raise ValueError(f'Selected topology has {world_size} GPUs; {key} must equal {expected}, '
                             f"found {os.environ[key]!r}.")
    node_rank = args.node_rank
    if node_rank is None and args.nnodes == 1:
        node_rank = 0
    if node_rank is None or not 0 <= node_rank < args.nnodes:
        raise ValueError(f'--node-rank must be between 0 and {args.nnodes - 1}.')
    master_addr = args.master_addr or ('localhost' if args.nnodes == 1 else None)
    if not master_addr or not master_addr.strip():
        raise ValueError('--master-addr (or MASTER_ADDR) is required for a multi-node run.')
    return {'world_size': world_size, 'data_parallel_size': data_parallel_size,
            'expert_data_parallel_size': world_size // expert_size,
            'gradient_accumulation_steps': config['global_batch_size'] // micro_batch,
            'node_rank': node_rank, 'master_addr': master_addr}


def resolve_swift_package(verify=True):
    if __package__:
        from .backend import resolve_swift_package as resolve
    else:
        from backend import resolve_swift_package as resolve
    return resolve(verify=verify)


def build_plan(args):
    model = validate_model(args.model)
    checkpoint = validate_native_checkpoint(args.mcore_model) if args.mcore_model else None
    output = Path(args.output).expanduser().resolve()
    environment = {'USE_MCORE_GDN': '1', 'SWIFT_NONTHINK_LIST_CONTENT': '0',
                   'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'MS_SWIFT_SKIP_TE_PRELOAD': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
                   'OMP_NUM_THREADS': '14', 'NVTE_FP8_PADDING': '1',
                   'AFM_DCP_SAVE_THREADS': '8', 'AFM_DCP_SYNC_FILES': '1',
                   'HTTP_POOL_SIZE': os.environ.get('HTTP_POOL_SIZE', '128')}
    # Retain the source launcher's conditional CUDA 13 workaround without loading TE early.
    cublas = Path(os.environ.get('CUBLAS_LIB_DIR', '/usr/local/cuda/targets/x86_64-linux/lib'))
    if (cublas / 'libcublasLt.so.13').is_file():
        preload = f"{cublas / 'libcublasLt.so.13'}:{cublas / 'libcublas.so.13'}"
        if os.environ.get('LD_PRELOAD'):
            preload += ':' + os.environ['LD_PRELOAD']
        environment['LD_PRELOAD'] = preload
    if args.stage == 'export':
        if checkpoint is None:
            raise ValueError('export requires --mcore-model with a native checkpoint.')
        if output == checkpoint.parent or checkpoint.parent in output.parents:
            raise ValueError('Export output must be outside the native checkpoint training directory.')
        environment.update({'MS_SWIFT_EXPORT_NO_FSDP': '1', 'BASE_MODEL': str(model),
                            'CUDA_VISIBLE_DEVICES': visible_devices(1),
                            'NPROC_PER_NODE': '1', 'NNODES': '1', 'NODE_RANK': '0',
                            'MASTER_ADDR': 'localhost', 'MASTER_PORT': str(args.master_port)})
        config = {'model': str(model), 'mcore_model': str(checkpoint), 'to_hf': True,
                  'output_dir': str(output), 'torch_dtype': 'bfloat16', 'exist_ok': False}
        command = [sys.executable, '-m', 'swift.cli._megatron.main', 'export', *as_flags(config)]
        return {'stage': args.stage, 'world_size': 1, 'config': config, 'command': command, 'environment': environment}
    config = yaml.safe_load((HERE / 'common.yaml').read_text())
    topology = training_topology(args, config)
    if not args.dataset:
        raise ValueError('--dataset is required for training.')
    dataset = validate_dataset(args.dataset)
    if args.stage == 'stage2' and checkpoint is None:
        raise ValueError('stage2 requires --mcore-model with the native stage-1 checkpoint-795.')
    if args.stage == 'stage1' and checkpoint is not None:
        raise ValueError('stage1 starts from the base model; --mcore-model is only used by stage2 or export.')
    interval = 160 if args.stage == 'stage1' else 1000000
    config.update({'model': str(model), 'dataset': [str(dataset)], 'output_dir': str(output),
                   'tensorboard_dir': str(output / 'runs'), 'save_steps': interval, 'eval_steps': interval})
    if checkpoint is not None:
        config['mcore_model'] = str(checkpoint)
    environment.update({'NNODES': str(args.nnodes), 'NPROC_PER_NODE': str(args.nproc_per_node),
                        'NODE_RANK': str(topology['node_rank']),
                        'CUDA_VISIBLE_DEVICES': visible_devices(args.nproc_per_node),
                        'MASTER_ADDR': topology['master_addr'], 'MASTER_PORT': str(args.master_port),
                        'MS_SWIFT_EXPORT_NO_FSDP': '0'})
    entry = resolve_swift_package(verify=False) / 'cli' / '_megatron' / 'sft.py'
    command = [sys.executable, '-m', 'torch.distributed.run', f'--nnodes={args.nnodes}',
               f'--nproc_per_node={args.nproc_per_node}',
               f"--node_rank={topology['node_rank']}", f"--master_addr={topology['master_addr']}",
               f'--master_port={args.master_port}', str(entry), *as_flags(config)]
    return {'stage': args.stage, **topology, 'config': config, 'command': command, 'environment': environment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['stage1', 'stage2', 'export'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mcore-model')
    parser.add_argument('--nnodes', type=int, default=NNODES, help='training node count (default: 4)')
    parser.add_argument('--nproc-per-node', type=int, default=GPUS_PER_NODE,
                        help='GPU workers per training node (default: 8)')
    parser.add_argument('--node-rank', type=int, default=os.environ.get('NODE_RANK'))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR'))
    parser.add_argument('--master-port', type=int, default=os.environ.get('MASTER_PORT', '29500'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if not 1 <= args.master_port <= 65535:
            raise ValueError('--master-port must be between 1 and 65535.')
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        if __package__:
            from .runtime import verify_runtime
        else:
            from runtime import verify_runtime
        verify_runtime()
        resolve_swift_package()
        environment = os.environ.copy()
        environment.update(plan['environment'])
        # Retain the source launcher's cache and profiling cleanup. Network interfaces
        # are configured by each deployment; the old scheduler's eth0/GID defaults are omitted.
        for key in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                    'TORCH_NCCL_TRACE_BUFFER_SIZE', 'TORCH_FR_BUFFER_SIZE', 'TORCH_NCCL_ENABLE_TIMING',
                    'TORCH_NCCL_ENABLE_MONITORING', 'TORCH_NCCL_DUMP_ON_TIMEOUT', 'TORCH_NCCL_DESYNC_DEBUG',
                    'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'TORCH_NCCL_DEBUG_INFO_PIPE_FILE',
                    'TORCH_NCCL_DEBUG_INFO_TEMP_FILE', 'TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC',
                    'TORCH_NCCL_COORD_CHECK_MILSEC', 'NSYS_GPU_METRICS', 'NSYS_GPU_METRICS_DEVICES'):
            environment.pop(key, None)
        result = subprocess.run(plan['command'], env=environment)
        if result.returncode == 0 and args.stage == 'export':
            print(json.dumps(verify_export(plan['config']['output_dir']), indent=2))
        raise SystemExit(result.returncode)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
