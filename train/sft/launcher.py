#!/usr/bin/env python3
"""Launch the recorded two-stage 32-GPU SFT configuration or export its checkpoint."""
import argparse
import importlib.util
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


def verify_export(directory):
    """Retain the recovered exporter's safetensors parameter-count sanity check."""
    directory = Path(directory)
    files = sorted(directory.glob('*.safetensors'))
    parameters = 0
    for filename in files:
        with filename.open('rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        for key, value in header.items():
            if key == '__metadata__':
                continue
            count = 1
            for dimension in value['shape']:
                count *= dimension
            parameters += count
    if parameters <= 120_000_000_000 or not all(
            (directory / name).is_file() for name in ['config.json', 'tokenizer.json']):
        raise ValueError('HF export is incomplete: expected more than 120B parameters plus config/tokenizer.')
    return {'safetensors_shards': len(files), 'parameters': parameters}


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
                            'CUDA_VISIBLE_DEVICES': '0',
                            'NPROC_PER_NODE': '1', 'NNODES': '1', 'NODE_RANK': '0',
                            'MASTER_ADDR': 'localhost', 'MASTER_PORT': str(args.master_port)})
        config = {'model': str(model), 'mcore_model': str(checkpoint), 'to_hf': True,
                  'output_dir': str(output), 'torch_dtype': 'bfloat16', 'exist_ok': False}
        command = [sys.executable, '-m', 'swift.cli._megatron.main', 'export', *as_flags(config)]
        return {'stage': args.stage, 'world_size': 1, 'config': config, 'command': command, 'environment': environment}
    for key, expected in {'NNODES': NNODES, 'NPROC_PER_NODE': GPUS_PER_NODE, 'WORLD_SIZE': 32}.items():
        if key in os.environ and os.environ[key] != str(expected):
            raise ValueError(f'This recipe requires 32 GPUs (4 nodes x 8); {key} must equal {expected}.')
    if args.node_rank is None or not 0 <= args.node_rank < NNODES:
        raise ValueError('--node-rank must be 0, 1, 2, or 3 for the 32-GPU run.')
    if not args.master_addr:
        raise ValueError('--master-addr (or MASTER_ADDR) is required for the four-node run.')
    if not args.dataset:
        raise ValueError('--dataset is required for training.')
    dataset = validate_dataset(args.dataset)
    if args.stage == 'stage2' and checkpoint is None:
        raise ValueError('stage2 requires --mcore-model with the native stage-1 checkpoint-795.')
    if args.stage == 'stage1' and checkpoint is not None:
        raise ValueError('stage1 starts from the base model; --mcore-model is only used by stage2 or export.')
    config = yaml.safe_load((HERE / 'common.yaml').read_text())
    interval = 160 if args.stage == 'stage1' else 1000000
    config.update({'model': str(model), 'dataset': [str(dataset)], 'output_dir': str(output),
                   'tensorboard_dir': str(output / 'runs'), 'save_steps': interval, 'eval_steps': interval})
    if checkpoint is not None:
        config['mcore_model'] = str(checkpoint)
    environment.update({'NNODES': '4', 'NPROC_PER_NODE': '8', 'NODE_RANK': str(args.node_rank),
                        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'),
                        'MASTER_ADDR': args.master_addr, 'MASTER_PORT': str(args.master_port),
                        'MS_SWIFT_EXPORT_NO_FSDP': '0'})
    entry = HERE.parent / 'swift' / 'swift' / 'cli' / '_megatron' / 'sft.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=4', '--nproc_per_node=8',
               f'--node_rank={args.node_rank}', f'--master_addr={args.master_addr}',
               f'--master_port={args.master_port}', str(entry), *as_flags(config)]
    return {'stage': args.stage, 'world_size': 32, 'config': config, 'command': command, 'environment': environment}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['stage1', 'stage2', 'export'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mcore-model')
    parser.add_argument('--node-rank', type=int, default=os.environ.get('NODE_RANK'))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR'))
    parser.add_argument('--master-port', type=int, default=int(os.environ.get('MASTER_PORT', '29500')))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if not 1 <= args.master_port <= 65535:
            raise ValueError('--master-port must be between 1 and 65535.')
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return
        from runtime import verify_runtime
        verify_runtime()
        expected = HERE.parent / 'swift' / 'swift' / '__init__.py'
        spec = importlib.util.find_spec('swift')
        if spec is None or Path(spec.origin).resolve() != expected.resolve():
            raise RuntimeError('Install this repository\'s Swift with pip install --no-deps -e train/swift.')
        environment = os.environ.copy()
        environment.update(plan['environment'])
        # Retain the source launcher's cache and profiling cleanup. Network interfaces
        # are configured by each deployment; the old scheduler's eth0/GID defaults are omitted.
        for key in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'NCCL_IB_HCA',
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
