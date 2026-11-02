"""Launch the Axion paper RL recipe from a Stage II 122B checkpoint."""
import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def backend_directory():
    spec = importlib.util.find_spec('verl')
    if spec and spec.origin:
        return Path(spec.origin).resolve().parent
    source = HERE.parents[2] / 'third_party/verl/verl'
    if (source / '__init__.py').is_file():
        return source
    raise ValueError('Install this repository\'s third_party/verl backend.')


def read_deployment(path):
    import yaml
    try:
        result = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as error:
        raise ValueError('deployment must contain valid YAML') from error
    if not isinstance(result, dict):
        raise ValueError('deployment must be a YAML mapping')
    limits = {'context_threshold': (12000, 61151), 'max_assistant_turns': (1, 100000),
              'max_turn_response_length': (1, 49152), 'max_tool_response_length': (1, 61152)}
    for key, (low, high) in limits.items():
        value = result.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'deployment.{key} must be explicitly set to an integer in [{low}, {high}].')
    for role, model in [('judge', 'Qwen3.8-27B'), ('summarizer', 'Qwen3.5-9B'), ('condenser', 'Qwen3.5-9B')]:
        service = result.get(role, {})
        if not isinstance(service, dict):
            raise ValueError(f'deployment.{role} must be a YAML mapping.')
        if not service.get('base_url') or service.get('model') != model:
            raise ValueError(f'deployment.{role} requires base_url and the paper model {model}.')
        env_name = service.get('api_key_env')
        if not isinstance(env_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env_name):
            raise ValueError(f'deployment.{role}.api_key_env must name its credential environment variable.')
    urls = {key: result.get(key) for key in ('search_url', 'scholar_url', 'python_url')}
    urls.update({role + '.base_url': result[role]['base_url'] for role in ('judge', 'summarizer', 'condenser')})
    for key, url in urls.items():
        try:
            if not isinstance(url, str) or any(char.isspace() for char in url):
                raise ValueError('URL must be a string without whitespace')
            parts = urlsplit(url)
            if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password
                    or parts.query or parts.fragment or parts.port == 0):
                raise ValueError('invalid service URL')
        except ValueError as error:
            raise ValueError(f'deployment.{key} must be an HTTP(S) URL with a valid port and no '
                             'embedded credentials, query string or fragment.') from error
    defaults = {'rollout_nodes': 5, 'trainer_nodes': 2, 'gpus_per_node': 8,
                'rollout_tensor_parallel_size': 4, 'trainer_tensor_parallel_size': 2,
                'trainer_expert_parallel_size': 8}
    for key, default in defaults.items():
        value = result.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f'deployment.{key} must be a positive integer.')
        result[key] = value
    trainer_gpus = result['trainer_nodes'] * result['gpus_per_node']
    for key in ('trainer_tensor_parallel_size', 'trainer_expert_parallel_size'):
        if trainer_gpus % result[key]:
            raise ValueError(f'deployment.{key} must divide the {trainer_gpus} trainer GPUs.')
    if 256 % result['trainer_expert_parallel_size']:
        raise ValueError('deployment.trainer_expert_parallel_size must divide the model\'s 256 experts.')
    rollout_gpus = result['rollout_nodes'] * result['gpus_per_node']
    rollout_tp = result['rollout_tensor_parallel_size']
    if rollout_gpus % rollout_tp:
        raise ValueError('deployment.rollout_tensor_parallel_size must divide the total rollout GPUs.')
    if rollout_tp % min(result['gpus_per_node'], rollout_tp):
        raise ValueError('deployment.rollout_tensor_parallel_size must use whole replica nodes when '
                         'it exceeds gpus_per_node.')
    return result


def validate_cluster_resources(config, node_resources=None):
    """Reject impossible Ray placements before loading models or creating output.

    This checks current capacity, not a reservation: other Ray jobs can still
    consume resources after the check. Logical pools may share a physical host.
    """
    if node_resources is None:
        import ray
        alive = {node['NodeID'] for node in ray.nodes() if node['Alive']}
        node_resources = {node: resources for node, resources in
                          ray._private.state.available_resources_per_node().items() if node in alive}
    capacities = tuple(sorted((math.floor(resources.get('GPU', 0)),
                               math.floor(resources.get('CPU', 0)), False)
                              for resources in node_resources.values()))
    trainer = config.trainer
    rollout = config.rollout
    tp = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    replica_node_gpus = min(rollout.n_gpus_per_node, tp)
    rollout_gpus = rollout.nnodes * rollout.n_gpus_per_node
    # Trainer groups use STRICT_PACK with 3 CPU per GPU. Standalone rollout
    # groups use PACK with 2 CPU per GPU, but vLLM groups their workers by
    # host; check that each replica node can fit on one physical host.
    groups = [(trainer.n_gpus_per_node, 3 * trainer.n_gpus_per_node, False)] * trainer.nnodes
    groups += [(replica_node_gpus, 2 * replica_node_gpus, True)] * (rollout_gpus // replica_node_gpus)
    gpu_required = sum(gpus for gpus, _, _ in groups)
    # TaskRunner=1, FullyAsyncTrainer=10, FullyAsyncRollouter=10, MessageQueue=2.
    cpu_required = sum(cpus for _, cpus, _ in groups) + 23
    if (sum(gpus for gpus, _, _ in capacities) < gpu_required
            or sum(cpus for _, cpus, _ in capacities) < cpu_required):
        raise ValueError(f'Ray cluster needs {gpu_required} free GPUs and at least {cpu_required} free CPUs '
                         'for both disjoint pools and their coordinators.')
    # The queue starts after rollout servers. Model that ordering so its 2 CPU
    # can provide the temporary scheduling CPU used by default Ray actors.
    groups += [(0, 10, False), (0, 10, False), (0, 1, False)]
    groups.sort(reverse=True)

    @lru_cache(maxsize=4096)
    def fits(index, nodes):
        if index == len(groups):
            # vLLM HTTP server actors have hard affinity to their GPU host and
            # need 1 scheduling CPU outside the GPU placement group. They use
            # 0 lifetime CPU; MessageQueue subsequently reserves 2 CPU.
            return all(not has_rollout or cpus >= 1 for _, cpus, has_rollout in nodes) and any(
                cpus >= 2 for _, cpus, _ in nodes)
        gpus, cpus, is_rollout = groups[index]
        tried = set()
        for pos, (available_gpus, available_cpus, has_rollout) in enumerate(nodes):
            if nodes[pos] in tried or available_gpus < gpus or available_cpus < cpus:
                continue
            tried.add(nodes[pos])
            remaining = list(nodes)
            remaining[pos] = (available_gpus - gpus, available_cpus - cpus, has_rollout or is_rollout)
            if fits(index + 1, tuple(sorted(remaining))):
                return True
        return False

    if not fits(0, capacities):
        raise ValueError('Ray free resources cannot fit the trainer and rollout groups on their hosts: '
                         f'trainer groups need {trainer.n_gpus_per_node} GPUs and '
                         f'{3 * trainer.n_gpus_per_node} CPUs each; rollout groups need '
                         f'{replica_node_gpus} GPUs and {2 * replica_node_gpus} CPUs each. '
                         'Allow a scheduling CPU on each rollout host and CPU capacity for the '
                         '10/10/1/2-CPU coordinators. Adjust the deployment or free node resources.')


def build_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    if args.updates <= 0 or args.updates % 2:
        raise ValueError('--updates must be a positive even number; checkpoints follow two-update weight synchronization.')
    if args.rollout_attempts is not None and args.rollout_attempts < args.updates * 24:
        raise ValueError('--rollout-attempts must allow at least 24 accepted prompt groups per update.')
    model = Path(args.model).expanduser().resolve()
    shape = json.loads((model / 'config.json').read_text())
    text = shape.get('text_config', {})
    if shape.get('model_type') != 'qwen3_5_moe' or any(text.get(k) != v for k, v in {
            'hidden_size': 3072, 'num_hidden_layers': 48, 'num_experts': 256, 'num_experts_per_tok': 8}.items()):
        raise ValueError('--model must be a Qwen3.5-122B-A10B Stage II Hugging Face checkpoint.')
    data = Path(args.train_data).expanduser().resolve()
    if not data.is_file() or data.suffix != '.parquet':
        raise ValueError('--train-data must be the selected RL parquet produced by prepare_rl.')
    deployment = read_deployment(args.deployment)
    search = json.dumps(['file://' + str(backend_directory() / 'trainer/config')])
    with initialize_config_dir(version_base=None, config_dir=str(HERE / 'config')):
        config = compose(config_name='paper', overrides=['hydra.searchpath=' + search])
    with open_dict(config):
        config.actor_rollout_ref.model.path = str(model)
        config.actor_rollout_ref.model.tokenizer_path = str(model)
        config.data.train_files = [str(data)]
        config.trainer.default_local_dir = str(Path(args.output).expanduser().resolve())
        config.trainer.total_training_steps = args.updates
        config.actor_rollout_ref.actor.optim.total_training_steps = args.updates
        config.actor_rollout_ref.actor.optim.lr_decay_steps = args.updates
        # The explicit attempt budget bounds rejection-heavy data; it is not an optimizer-step count.
        config.rollout.total_rollout_steps = args.rollout_attempts or args.updates * 24 * 100
        config.custom_reward_function.path = str(HERE / 'reward.py')
        multi = config.actor_rollout_ref.rollout.multi_turn
        for key in ('max_assistant_turns', 'max_tool_response_length'):
            multi[key] = deployment[key]
        multi.max_user_turns = deployment['max_assistant_turns']
        multi.tool_config_path = str(HERE / 'config/tools.yaml')
        config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(HERE / 'config/agent_loop_config.yaml')
        for role in ('rollout', 'trainer'):
            config[role].nnodes = deployment.get(role + '_nodes', 5 if role == 'rollout' else 2)
            config[role].n_gpus_per_node = deployment.get('gpus_per_node', 8)
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = deployment.get('rollout_tensor_parallel_size', 4)
        for role in ('actor', 'ref'):
            engine = config.actor_rollout_ref[role].megatron
            engine.tensor_model_parallel_size = deployment.get('trainer_tensor_parallel_size', 2)
            engine.expert_model_parallel_size = deployment.get('trainer_expert_parallel_size', 8)
    return config, deployment


def worker_environment(deployment):
    # Values stay outside the printed/saved Hydra configuration.
    env = {key: os.environ[key] for key in ('SERPER_KEY_ID', 'JINA_API_KEY', 'JINA_API_KEYS',
           'HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HF_HOME', 'HF_HUB_CACHE') if key in os.environ}
    # Connecting to an existing Ray cluster does not inherit the launch shell.
    # Preserve explicit communication settings without inventing new defaults.
    env.update({key: value for key, value in os.environ.items()
                if key.startswith(('NCCL_', 'TORCH_NCCL_')) or key == 'CUDA_DEVICE_MAX_CONNECTIONS'})
    env.update({'MEMORY_ENABLED': '1', 'MEMORY_TIER1': '0',
                'FORCE_FINAL_ANSWER': '0', 'SOFTEN_ANSWER_PROMPT': '0',
                'CONTEXT_THRESHOLD': str(deployment['context_threshold']),
                'MAX_TURN_RESPONSE_LENGTH': str(deployment['max_turn_response_length']),
                'SEARCH_SERVICE_URL': deployment['search_url'], 'SCHOLAR_SERVICE_URL': deployment['scholar_url'],
                'PYTHON_SANDBOX_BACKEND': 'sandbox_fusion', 'SANDBOX_FUSION_ENDPOINT': deployment['python_url'],
                'PYTHONPATH': os.pathsep.join([str(HERE.parents[2]), str(HERE), os.environ.get('PYTHONPATH', '')])})
    for role, prefixes in [('judge', ['EVAL_LLM', 'CITATION_EVAL_LLM', 'OPENENDED_EVAL_LLM']),
                           ('summarizer', ['VISIT_SUMMARY']), ('condenser', ['MEMORY'])]:
        service = deployment[role]
        secret = os.environ.get(service['api_key_env'])
        if not secret:
            raise ValueError(f'Missing service credential environment variable {service["api_key_env"]}.')
        for prefix in prefixes:
            env.update({prefix + '_API_BASE': service['base_url'], prefix + '_MODEL_NAME': service['model'],
                        prefix + '_API_KEY': secret, prefix + '_PROVIDER': 'openai',
                        prefix + '_FALLBACK_MODEL_NAME': '', prefix + '_FALLBACK_API_KEY': '',
                        prefix + '_FALLBACK_API_BASE': '', prefix + '_AZURE_ENDPOINT': '',
                        prefix + '_FALLBACK_AZURE_ENDPOINT': '', prefix + '_LOCAL_FALLBACK_MODEL_NAME': ''})
    judge = deployment['judge']
    env.update({'OPENAI_BASE_URL': judge['base_url'], 'API_BASE': judge['base_url'],
                'OPENAI_API_KEY': os.environ[judge['api_key_env']], 'API_KEY': os.environ[judge['api_key_env']],
                'AZURE_OPENAI_ENDPOINT': '', 'S2_JUDGE_BASE_URL': judge['base_url'],
                'S2_JUDGE_MODEL': judge['model'], 'S2_JUDGE_API_KEY': os.environ[judge['api_key_env']]})
    for name in ('SEARCH', 'SCHOLAR', 'PYTHON'):
        env[name + '_NODES_CONF'] = ''
    return env


def validate_selected_data(path):
    path = Path(path)
    manifest_path = path.with_suffix('.selection.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('attempts_per_task') != 4 or manifest.get('criterion') != 'at_least_one_correct_and_nonzero_reward_variance':
        raise ValueError('RL selection manifest does not record the paper\'s four-attempt selection rule.')
    if manifest.get('selected_count', 0) < 1:
        raise ValueError('RL selection contains no usable tasks.')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest.get('parquet_sha256'):
        raise ValueError('RL parquet does not match its selection manifest.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'train-data', 'output', 'deployment'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--updates', required=True, type=int)
    parser.add_argument('--rollout-attempts', type=int, help='Maximum attempted prompt groups before an explicit shortfall error.')
    parser.add_argument('--ray-address', default='auto')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        from omegaconf import OmegaConf
        config, deployment = build_config(args)
        if args.dry_run:
            print(json.dumps({'config': OmegaConf.to_container(config, resolve=True),
                              'deployment': deployment, 'validation': 'Configuration only; no workers or services started.'}, indent=2))
            return
        output = Path(config.trainer.default_local_dir)
        if output.exists() and any(output.iterdir()):
            raise ValueError('--output must be empty; this recipe starts a new run from Stage II.')
        manifest = validate_selected_data(config.data.train_files[0])
        from .runtime import verify_runtime
        runtime = verify_runtime(args.model, vanilla_mbridge=True)
        env = worker_environment(deployment)
        os.environ.update(env)
        import ray
        from verl.trainer.main_ppo import get_ppo_ray_runtime_env
        defaults = get_ppo_ray_runtime_env(config)
        runtime_env = OmegaConf.to_container(OmegaConf.merge(defaults, {'env_vars': env}), resolve=True)
        ray.init(address=args.ray_address, runtime_env=runtime_env)
        validate_cluster_resources(config)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
        (output / 'run_provenance.json').write_text(json.dumps({'runtime': runtime, 'selection': manifest,
            'source': 'Paper-based implementation; original final RL runtime not recovered.'}, indent=2) + '\n')
        from verl.experimental.fully_async_policy.fully_async_main import main as train
        train.__wrapped__(config)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
