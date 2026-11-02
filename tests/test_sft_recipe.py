import hashlib
import ast
import difflib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Literal, Optional
import unittest
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('AXION_SFT_TEST_ROOT', Path(__file__).resolve().parents[1] / 'axion/training/sft'))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='axion-sft-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.model = self.base / 'model'
        self.model.mkdir()
        shape = dict(hidden_size=3072, num_hidden_layers=48, num_experts=256,
                     num_experts_per_tok=8, linear_num_key_heads=16, linear_num_value_heads=64,
                     num_attention_heads=32, num_key_value_heads=2)
        (self.model / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5_moe', 'text_config': shape}))
        self.data = self.base / 'data.jsonl'
        self.data.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'hi'},
                            {'role': 'assistant', 'content': 'hello'}], 'total_tokens': 20, 'visual_tokens': 0}) + '\n')
        self.checkpoint = self.base / 'checkpoint-795'
        self.checkpoint.mkdir()
        (self.checkpoint / 'args.json').write_text(json.dumps({'ckpt_format': 'fsdp_dtensor',
                    'tensor_model_parallel_size': 1, 'expert_model_parallel_size': 8,
                    'context_parallel_size': 4, 'pipeline_model_parallel_size': 1}))
        (self.checkpoint / 'latest_checkpointed_iteration.txt').write_text('795\n')
        iteration = self.checkpoint / 'iter_0000795'
        iteration.mkdir()
        (iteration / '.metadata').write_bytes(b'fixture')

    def run_launch(self, stage='stage1', more=(), env=None):
        command = ['bash', str(ROOT / 'train.sh'), stage, '--model', str(self.model),
                   '--dataset', str(self.data), '--output', str(self.base / 'output'),
                   '--master-addr', 'trainer-head.example', '--node-rank', '0', '--dry-run', *more]
        clean_env = {k: v for k, v in os.environ.items() if k not in {'NNODES', 'NPROC_PER_NODE', 'WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'CUDA_VISIBLE_DEVICES'}}
        clean_env.update(env or {})
        return subprocess.run(command, text=True, capture_output=True, env=clean_env)

    def plan(self, stage='stage1', more=()):
        result = self.run_launch(stage, more)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_selected_topology_and_two_stage_configuration(self):
        stage1 = self.plan()
        stage2 = self.plan('stage2', ('--mcore-model', str(self.checkpoint)))
        expected = {
            'context_parallel_size': 4, 'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1, 'expert_model_parallel_size': 8,
            'expert_tensor_parallel_size': 1, 'micro_batch_size': 1,
            'global_batch_size': 8, 'max_length': 131072, 'packing_length': 131072,
            'packing': True, 'padding_free': True, 'use_megatron_fsdp': True,
            'ckpt_format': 'fsdp_dtensor', 'save_total_limit': 100,
            'finetune': True, 'num_train_epochs': 3, 'lr': 3e-6, 'min_lr': 3e-6,
            'lr_decay_style': 'constant', 'lr_warmup_iters': 0,
            'loss_scale': 'default', 'add_non_thinking_prefix': True,
            'no_save_optim': False, 'no_save_rng': False,
        }
        for plan in (stage1, stage2):
            self.assertEqual(plan['world_size'], 32)
            self.assertIn('--nnodes=4', plan['command'])
            self.assertIn('--nproc_per_node=8', plan['command'])
            for key, value in expected.items():
                self.assertEqual(plan['config'][key], value, key)
        self.assertEqual(stage1['config']['save_steps'], 160)
        self.assertEqual(stage1['config']['eval_steps'], 160)
        self.assertEqual(stage2['config']['save_steps'], 1000000)
        self.assertEqual(stage2['config']['eval_steps'], 1000000)
        changes = {key for key in stage1['config'].keys() | stage2['config'].keys()
                   if stage1['config'].get(key) != stage2['config'].get(key)}
        self.assertEqual(changes, {'save_steps', 'eval_steps', 'mcore_model'})
        self.assertFalse((self.base / 'output').exists(), 'dry-run created output')

    def test_stage2_requires_native_checkpoint_and_never_resumes_optimizer(self):
        missing = self.run_launch('stage2')
        self.assertNotEqual(missing.returncode, 0)
        hf_only = self.run_launch('stage2', ('--mcore-model', str(self.model)))
        self.assertNotEqual(hf_only.returncode, 0)
        self.assertIn('native', hf_only.stderr.lower())
        plan = self.plan('stage2', ('--mcore-model', str(self.checkpoint)))
        self.assertEqual(plan['config']['mcore_model'], str(self.checkpoint))
        self.assertTrue(plan['config']['finetune'])
        self.assertEqual(plan['config']['save_steps'], 1000000)

    def test_wrong_world_size_or_rank_is_rejected(self):
        for env in ({'NNODES': '8'}, {'NPROC_PER_NODE': '4'}, {'WORLD_SIZE': '64'}):
            result = self.run_launch(env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('32', result.stderr)
        self.assertNotEqual(self.run_launch(more=('--node-rank', '4')).returncode, 0)

    def test_smaller_topologies_preserve_learning_recipe(self):
        reference = self.plan()
        for nodes, world, dp, expert_dp, accumulation in [(1, 8, 2, 1, 4), (2, 16, 4, 2, 2)]:
            with self.subTest(nodes=nodes):
                plan = self.plan(more=('--nnodes', str(nodes), '--nproc-per-node', '8'))
                self.assertEqual(plan['config'], reference['config'])
                self.assertEqual(plan['world_size'], world)
                self.assertEqual(plan['data_parallel_size'], dp)
                self.assertEqual(plan['expert_data_parallel_size'], expert_dp)
                self.assertEqual(plan['gradient_accumulation_steps'], accumulation)
                self.assertIn(f'--nnodes={nodes}', plan['command'])
                self.assertEqual(plan['environment']['NNODES'], str(nodes))
                stage2 = self.plan('stage2', ('--nnodes', str(nodes), '--mcore-model', str(self.checkpoint)))
                self.assertEqual(stage2['world_size'], world)
                self.assertEqual(stage2['config']['global_batch_size'], 8)

    def test_single_node_defaults_rank_and_master_only_for_single_node(self):
        command = [sys.executable, str(ROOT / 'launcher.py'), 'stage1', '--model', str(self.model),
                   '--dataset', str(self.data), '--output', str(self.base / 'output'), '--dry-run']
        env = {k: v for k, v in os.environ.items() if k not in {
            'NNODES', 'NPROC_PER_NODE', 'WORLD_SIZE', 'NODE_RANK', 'MASTER_ADDR', 'CUDA_VISIBLE_DEVICES'}}
        result = subprocess.run([*command, '--nnodes', '1'], text=True, capture_output=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['environment']['NODE_RANK'], '0')
        self.assertEqual(plan['environment']['MASTER_ADDR'], 'localhost')
        result = subprocess.run(command, text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('node-rank', result.stderr)
        result = subprocess.run([*command, '--node-rank', '0'], text=True, capture_output=True, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('master-addr', result.stderr)

    def test_invalid_topologies_fail_before_workers(self):
        cases = [('--nnodes', '0'), ('--nproc-per-node', '0'),
                 ('--nnodes', '1', '--nproc-per-node', '4'), ('--nnodes', '3'),
                 ('--nnodes', '8'), ('--nnodes', '1', '--node-rank', '1')]
        for flags in cases:
            with self.subTest(flags=flags):
                result = self.run_launch(more=flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('unrecognized arguments', result.stderr)

    def test_selected_topology_checks_inherited_environment(self):
        plan = self.run_launch(more=('--nnodes', '1'), env={
            'NNODES': '1', 'NPROC_PER_NODE': '8', 'WORLD_SIZE': '8'})
        self.assertEqual(plan.returncode, 0, plan.stderr)
        for env in ({'NNODES': '4'}, {'NPROC_PER_NODE': '4'}, {'WORLD_SIZE': '32'}):
            result = self.run_launch(more=('--nnodes', '1'), env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(next(iter(env)), result.stderr)

    def test_visible_gpu_selection_is_preserved_and_validated(self):
        selected = '7,6,5,4,3,2,1,0'
        result = self.run_launch(more=('--nnodes', '1'), env={'CUDA_VISIBLE_DEVICES': selected})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['environment']['CUDA_VISIBLE_DEVICES'], selected)
        for selected in ('0,1', '', '-1', '0,0,1,2,3,4,5,6'):
            result = self.run_launch(more=('--nnodes', '1'), env={'CUDA_VISIBLE_DEVICES': selected})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CUDA_VISIBLE_DEVICES', result.stderr)
        result = self.run_launch(more=('--nnodes', '2', '--nproc-per-node', '4'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['environment']['CUDA_VISIBLE_DEVICES'], '0,1,2,3')

    def test_deployment_network_selection_reaches_workers(self):
        spec = importlib.util.spec_from_file_location('launcher_network_test', ROOT / 'launcher.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        runtime = ModuleType('runtime')
        runtime.verify_runtime = lambda: None
        command = ['launcher', 'stage1', '--model', str(self.model), '--dataset', str(self.data),
                   '--output', str(self.base / 'output'), '--master-addr', 'head', '--node-rank', '0']
        env = {k: v for k, v in os.environ.items() if k not in {
            'NNODES', 'NPROC_PER_NODE', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES'}}
        env.update({'NCCL_IB_HCA': '=mlx5_0:1', 'NCCL_SOCKET_IFNAME': 'ib0'})
        observed = {}
        def launch_child(command, env):
            observed.update(env)
            return SimpleNamespace(returncode=0)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, 'argv', command), \
             mock.patch.dict(sys.modules, {'runtime': runtime}), \
             mock.patch.object(launcher, 'resolve_swift_package', return_value=ROOT), \
             mock.patch.object(launcher.subprocess, 'run', side_effect=launch_child):
            with self.assertRaises(SystemExit) as result:
                launcher.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(observed.get('NCCL_IB_HCA'), '=mlx5_0:1')
        self.assertEqual(observed.get('NCCL_SOCKET_IFNAME'), 'ib0')

    def test_missing_or_overlength_metadata_is_rejected(self):
        for row in ({'messages': []}, {'messages': [], 'total_tokens': 131073, 'visual_tokens': 0}):
            self.data.write_text(json.dumps(row) + '\n')
            result = self.run_launch()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('metadata', result.stderr.lower())

    def test_selected_runtime_environment_is_not_changed_by_inherited_flags(self):
        result = self.run_launch(env={'USE_MCORE_GDN': '0', 'SWIFT_NONTHINK_LIST_CONTENT': '1',
                                           'MS_SWIFT_SKIP_TE_PRELOAD': '0', 'CUDA_DEVICE_MAX_CONNECTIONS': '8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        env = json.loads(result.stdout)['environment']
        self.assertEqual(env['USE_MCORE_GDN'], '1')
        self.assertEqual(env['SWIFT_NONTHINK_LIST_CONTENT'], '0')
        self.assertEqual(env['MS_SWIFT_SKIP_TE_PRELOAD'], '1')
        self.assertEqual(env['CUDA_DEVICE_MAX_CONNECTIONS'], '1')

    def test_export_uses_direct_single_process_path_outside_training_directory(self):
        command = ['bash', str(ROOT / 'train.sh'), 'export', '--model', str(self.model),
                   '--mcore-model', str(self.checkpoint), '--output', str(self.base / 'export-inside-training'),
                   '--dry-run']
        invalid = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)
        with tempfile.TemporaryDirectory(prefix='axion-export-test-') as directory:
            command[command.index('--output') + 1] = str(Path(directory) / 'hf')
            inherited = {**os.environ, 'BASE_MODEL': '/unavailable/old-model'}
            result = subprocess.run(command, text=True, capture_output=True, env=inherited)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['world_size'], 1)
            self.assertEqual(plan['environment']['MS_SWIFT_EXPORT_NO_FSDP'], '1')
            self.assertEqual(plan['environment']['NPROC_PER_NODE'], '1')
            self.assertEqual(plan['environment']['NNODES'], '1')
            self.assertTrue(plan['config']['to_hf'])
            self.assertEqual(plan['environment']['BASE_MODEL'], str(self.model))


class RuntimePatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='axion-mcore-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('runtime_under_test', ROOT / 'runtime.py')
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)
        # Exercise the real strict patch algorithm offline. Tiny valid Python modules
        # replace wheel contents; full release-file SHA pairs remain in provenance.
        originals = {
            'megatron/core/ssm/gated_delta_net.py': 'def context_parallel():\n    return False\n',
            'megatron/core/transformer/transformer_config.py': 'context_parallel_size = 1\n',
        }
        replacements = {
            'megatron/core/ssm/gated_delta_net.py': 'def context_parallel():\n    return True\n',
            'megatron/core/transformer/transformer_config.py': 'context_parallel_size = 4\n',
        }
        sections, hashes = [], {}
        for name, before in originals.items():
            after = replacements[name]
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(before)
            sections.append(f'diff --git a/{name} b/{name}\n' + ''.join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f'a/{name}', tofile=f'b/{name}')))
            hashes[name] = tuple(hashlib.sha256(text.encode()).hexdigest() for text in (before, after))
        fixture_patch = self.root / 'fixture.diff'
        fixture_patch.write_text(''.join(sections))
        replacement = mock.patch.multiple(self.runtime, FILES=hashes, PATCH=fixture_patch,
                                          PATCH_SHA256=hashlib.sha256(fixture_patch.read_bytes()).hexdigest())
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_bundled_patch_matches_provenance(self):
        patch = ROOT / 'patches/mcore_v0.17.1_gdn_cp_pr2642.diff'
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(),
                         '3aa5a6266dd543fa50dd86e370bee25b38737801873736bacaef9ed5789d6cf0')

    def test_strict_patch_application_and_idempotency(self):
        self.assertFalse(self.runtime.check_mcore_patch(self.root))
        self.runtime.apply_mcore_patch(self.root)
        self.assertTrue(self.runtime.check_mcore_patch(self.root))
        before = {str(p): p.read_bytes() for p in self.root.rglob('*.py')}
        self.runtime.apply_mcore_patch(self.root)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*.py')})

    def test_unknown_source_is_rejected_before_any_changes(self):
        bad = self.root / 'megatron/core/transformer/transformer_config.py'
        bad.write_text(bad.read_text() + '\n# unexpected local edit\n')
        before = {str(p): p.read_bytes() for p in self.root.rglob('*.py')}
        with self.assertRaisesRegex(RuntimeError, 'unrecognized'):
            self.runtime.apply_mcore_patch(self.root)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*.py')})


class ExportCheckTests(unittest.TestCase):
    def test_header_count_detects_incomplete_export(self):
        spec = importlib.util.spec_from_file_location('launcher_under_test', ROOT / 'launcher.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='axion-export-header-') as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'tokenizer.json').write_text('{}')
            def write_header(count):
                header = json.dumps({'weight': {'shape': [count], 'dtype': 'BF16', 'data_offsets': [0, 0]}}).encode()
                (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header)
            write_header(60000000000)
            with self.assertRaises(ValueError):
                module.verify_export(root)
            write_header(122000000000)
            with self.assertRaisesRegex(ValueError, 'safetensors'):
                module.verify_export(root)


class ExportStructureTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('export_structure_test', ROOT / 'launcher.py')
        self.launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launcher)
        tmp = tempfile.TemporaryDirectory(prefix='axion-export-structure-')
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def shard(self, header, payload=b'', name='model.safetensors'):
        data = json.dumps(header).encode() if isinstance(header, dict) else header
        path = self.root / name
        path.write_bytes(struct.pack('<Q', len(data)) + data + payload)
        return path

    def test_reads_valid_small_tensors_and_zero_length_tensors_without_payload_load(self):
        path = self.shard({
            'matrix': {'shape': [2, 3], 'dtype': 'BF16', 'data_offsets': [0, 12]},
            'scalar': {'shape': [], 'dtype': 'F32', 'data_offsets': [12, 16]},
            'empty': {'shape': [0], 'dtype': 'F32', 'data_offsets': [16, 16]},
        }, bytes(16))
        self.assertEqual(self.launcher.safetensors_parameters(path), {'matrix': 6, 'scalar': 1, 'empty': 0})

    def test_rejects_truncated_overlapping_and_invalid_tensor_headers(self):
        cases = [
            ({'w': {'shape': [2], 'dtype': 'BF16', 'data_offsets': [0, 4]}}, bytes(2)),
            ({'w': {'shape': [1], 'dtype': 'BF16', 'data_offsets': [0, 2]},
              'v': {'shape': [1], 'dtype': 'BF16', 'data_offsets': [0, 2]}}, bytes(2)),
            ({'w': {'shape': [-1], 'dtype': 'BF16', 'data_offsets': [0, 0]}}, b''),
            ({'w': {'shape': [True], 'dtype': 'BF16', 'data_offsets': [0, 2]}}, bytes(2)),
            ({'w': {'shape': [1], 'dtype': 'INVALID', 'data_offsets': [0, 1]}}, bytes(1)),
            ({'w': {'shape': [1], 'dtype': 'BF16', 'data_offsets': [0, 4]}}, bytes(4)),
            ({'w': {'shape': [1], 'dtype': 'BF16', 'data_offsets': [2, 4]}}, bytes(4)),
            (b'{"w":{"shape":[0],"dtype":"BF16","data_offsets":[0,0]},"w":{"shape":[0],"dtype":"BF16","data_offsets":[0,0]}}', b''),
        ]
        for header, payload in cases:
            with self.subTest(header=header):
                with self.assertRaisesRegex(ValueError, 'safetensors'):
                    self.launcher.safetensors_parameters(self.shard(header, payload))
        for data in (b'', struct.pack('<Q', 1000) + b'{}'):
            path = self.root / 'short.safetensors'; path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'safetensors'):
                self.launcher.safetensors_parameters(path)

    def test_valid_shard_index_and_global_count_are_checked(self):
        (self.root / 'config.json').write_text('{}')
        (self.root / 'tokenizer.json').write_text('{}')
        for name in ('one.safetensors', 'two.safetensors'):
            (self.root / name).touch()
        index = self.root / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'a': 'one.safetensors', 'b': 'two.safetensors'}}))
        shards = [{'a': 61000000000}, {'b': 61000000000}]
        with mock.patch.object(self.launcher, 'safetensors_parameters', side_effect=shards):
            self.assertEqual(self.launcher.verify_export(self.root)['parameters'], 122000000000)
        for weight_map in ({'a': 'one.safetensors'}, {'a': 'two.safetensors', 'b': 'one.safetensors'}):
            index.write_text(json.dumps({'weight_map': weight_map}))
            with mock.patch.object(self.launcher, 'safetensors_parameters', side_effect=shards):
                with self.assertRaisesRegex(ValueError, 'index'):
                    self.launcher.verify_export(self.root)
        with mock.patch.object(self.launcher, 'safetensors_parameters', side_effect=[{'a': 61000000000}] * 2):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                self.launcher.verify_export(self.root)


class DirectExportWeightTests(unittest.TestCase):
    def setUp(self):
        filename = Path(__file__).resolve().parents[1] / 'third_party/swift/swift/megatron/utils/megatron_lm_utils.py'
        function = next(node for node in ast.parse(filename.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'dcp_direct_load_backbone')
        self.namespace = {'logger': SimpleNamespace(info=lambda *a: None, warning=lambda *a: None),
                          'torch': SimpleNamespace(empty=lambda *a, **k: object())}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(filename), 'exec'), self.namespace)

    def load(self, parameter_names, checkpoint_names, extra_missing=()):
        # GPU packages are unavailable in CPU publication tests. Exercise the real
        # loader, replacing only checkpoint I/O and the model's state-dict interface.
        parameters = dict(parameter_names)
        dcp = ModuleType('torch.distributed.checkpoint')
        metadata = {'model.module.' + name: SimpleNamespace(size=(1,), properties=SimpleNamespace(dtype='bf16'))
                    for name in checkpoint_names}
        dcp.FileSystemReader = lambda path: SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=metadata))
        dcp.load = lambda *a, **k: None
        dist = ModuleType('torch.distributed'); dist.checkpoint = dcp
        torch_module = ModuleType('torch'); torch_module.distributed = dist
        def load_state_dict(state, strict):
            return SimpleNamespace(missing_keys=[name for name in parameters if name not in state] + list(extra_missing),
                                   unexpected_keys=[name for name in state if name not in parameters])
        model = SimpleNamespace(load_state_dict=load_state_dict,
                                named_parameters=lambda remove_duplicate=True: list(parameters.items()))
        with mock.patch.dict(sys.modules, {'torch': torch_module, 'torch.distributed': dist,
                                          'torch.distributed.checkpoint': dcp}):
            return self.namespace['dcp_direct_load_backbone'](model, '/fixture')

    def test_missing_actual_backbone_parameter_aborts_export(self):
        for name in ('decoder.layers.0.weight', 'vision_model.weight'):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                self.load([(name, object())], [])

    def test_loaded_tied_parameter_alias_and_extra_state_are_allowed(self):
        shared = object()
        result = self.load([('embedding.weight', shared), ('output.weight', shared)], ['embedding.weight'],
                           extra_missing=['decoder._extra_state', 'rotary.nonpersistent_buffer'])
        self.assertEqual(result[0], 1)

    def test_all_distinct_parameters_load(self):
        self.assertEqual(self.load([('weight', object())], ['weight']), (1, [], []))


class ExportArgumentTests(unittest.TestCase):
    def test_explicit_model_survives_checkpoint_config_before_model_initialization(self):
        filename = Path(__file__).resolve().parents[1] / 'third_party/swift/swift/megatron/arguments/export_args.py'
        tree = ast.parse(filename.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'MegatronExportArguments')
        cls.decorator_list = []
        class Base:
            def _init_megatron_args(self):
                self.model_seen_by_base = self.model
        namespace = {
            'MegatronBaseArguments': Base,
            'MegatronArguments': SimpleNamespace(load_mcore_checkpoint_args=lambda path: {
                'model': '/unavailable/checkpoint-model', 'context_parallel_size': 4}),
            'HfConfigFactory': SimpleNamespace(to_torch_dtype=lambda dtype: dtype),
            'Literal': Literal, 'Optional': Optional,
        }
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(filename), 'exec'), namespace)
        for explicit, expected in [('/portable/base-model', '/portable/base-model'),
                                   (None, '/unavailable/checkpoint-model')]:
            args = namespace['MegatronExportArguments']()
            args._init_output_dir = lambda: None
            args.model, args.ckpt_dir, args.mcore_adapter = explicit, None, None
            args.mcore_model = '/portable/checkpoint-795'
            args._init_megatron_args()
            self.assertEqual(args.model, expected)
            self.assertEqual(args.model_seen_by_base, expected)
            self.assertEqual(args.context_parallel_size, 4)
            self.assertEqual(args.mcore_model, '/portable/checkpoint-795')


class DependencyPreflightTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('runtime_dependencies_under_test', ROOT / 'runtime.py')
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)

    def test_missing_qwen_loader_dependency_is_reported_before_model_load(self):
        def version(package):
            if package == 'qwen-vl-utils':
                raise self.runtime.importlib.metadata.PackageNotFoundError(package)
            return '0.6.0'
        with mock.patch.object(self.runtime.importlib.metadata, 'version', side_effect=version):
            with self.assertRaisesRegex(RuntimeError, 'qwen-vl-utils'):
                self.runtime.verify_model_dependencies()

    def test_old_qwen_utils_version_is_rejected(self):
        with mock.patch.object(self.runtime.importlib.metadata, 'version', return_value='0.0.13'):
            with self.assertRaisesRegex(RuntimeError, '0.0.14'):
                self.runtime.verify_model_dependencies()

    def test_loader_import_failure_is_reported(self):
        with mock.patch.object(self.runtime.importlib.metadata, 'version', return_value='0.6.0'), \
             mock.patch.object(self.runtime.importlib, 'import_module', side_effect=ImportError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'qwen_vl_utils'):
                self.runtime.verify_model_dependencies()

if __name__ == '__main__':
    unittest.main()
