# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import json
import numpy as np
import os
import shutil
from contextlib import nullcontext
from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from functools import partial
from modelscope.hub.utils.utils import get_cache_dir
from typing import Dict, List, Literal, Optional, Tuple, Union

from swift.hub import get_hub
from swift.utils import (disable_safe_ddp_context_use_barrier, get_dist_setting, get_logger, get_seed,
                         safe_ddp_context, use_hf_hub)
from .dataset_meta import DATASET_TYPE, BaseDatasetLoader
from .dataset_syntax import DatasetSyntax
from .preprocessor import RowPreprocessor
from .register import DATASET_MAPPING, DatasetMeta, SubsetDataset

logger = get_logger()


def _is_local_json_array_file(dataset_path: str) -> bool:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        while True:
            ch = f.read(1)
            if not ch:
                return False
            if not ch.isspace():
                return ch == '['


def _load_local_json_file(dataset_path: str) -> HfDataset:
    # orjson parses ~2x faster than stdlib json on large files; fall back if absent.
    try:
        import orjson
        _loads = orjson.loads
    except ImportError:
        _loads = json.loads
    with open(dataset_path, 'rb') as f:
        if _is_local_json_array_file(dataset_path):
            return HfDataset.from_list(_loads(f.read()))
        rows = [_loads(line) for line in f if line.strip()]
        return HfDataset.from_list(rows)


def _make_preprocessor_cache_key(dataset_path: str, preprocessor) -> str:
    stat = os.stat(dataset_path)
    preprocessor_id = (
        type(preprocessor).__name__ + ':v' + str(getattr(preprocessor, 'preprocessor_version', '1')))
    raw = json.dumps({
        'path': os.path.abspath(dataset_path),
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'preprocessor': preprocessor_id,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _distributed_prebuild_caches(datasets, resolve_loader, *, num_proc, streaming,
                                 preprocessed_dataset_cache_dir):
    """Opt-in (MS_SWIFT_DISTRIBUTED_PREBUILD): build the per-dataset preprocessed caches
    distributed across the local ranks instead of one rank building all while the rest idle.

    Each local rank builds the datasets it owns by strided assignment (i % world == rank),
    via the atomic claim in `_load_dataset_path`, so different local ranks build different
    datasets concurrently (~ceil(n/world) rounds). The cache is node-local, so this
    distributes the work within each node. Each builder's num_proc is scaled down to
    ~cpu_count / local_world_size so N concurrent maps don't re-create the fork/CPU contention
    that made every-rank-builds slow. The subsequent main loop memory-maps each cache,
    waiting (via the same claim path) for any that a peer rank is still building.
    """
    if not _env_flag('MS_SWIFT_DISTRIBUTED_PREBUILD'):
        return
    if streaming or not preprocessed_dataset_cache_dir or len(datasets) < 2:
        return
    _, local_rank, _, local_world_size = get_dist_setting()
    local_rank = max(local_rank, 0)
    local_world_size = max(local_world_size, 1)
    if local_world_size < 2:
        return
    try:
        cpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu = os.cpu_count() or 1
    per_builder = max(1, min(num_proc or 1, max(1, cpu // local_world_size)))
    n = len(datasets)
    # Strided ownership: each local rank builds the datasets it owns (i % world == rank),
    # so the work splits evenly and the owners build concurrently (~ceil(n/world) rounds).
    # A rotated full traversal instead converges all ranks onto one builder for the tail.
    mine = [i for i in range(n) if i % local_world_size == local_rank]
    logger.info(f'distributed pre-build: local_rank={local_rank}/{local_world_size} datasets={n} '
                f'building={mine} per_builder_num_proc={per_builder}')
    for i in mine:
        try:
            loader, dataset_syntax, dataset_meta, use_hf = resolve_loader(datasets[i], num_proc_override=per_builder)
            loader.load(dataset_syntax, dataset_meta, use_hf=use_hf)  # build+cache; result discarded
        except Exception as e:  # never abort training; the main loop will (re)build if needed
            logger.warning(f'distributed pre-build of {datasets[i]} skipped: {e}')


def _with_dataset_cg_id(dataset: DATASET_TYPE, cg_id: int) -> DATASET_TYPE:
    if dataset is None or not _env_flag('MS_SWIFT_ASSIGN_DATASET_CG_IDS'):
        return dataset
    if not isinstance(dataset, HfDataset):
        logger.warning(f'MS_SWIFT_ASSIGN_DATASET_CG_IDS only supports HfDataset, got {type(dataset).__name__}')
        return dataset
    column_names = set(getattr(dataset, 'column_names', []) or [])
    if getattr(dataset, '_indices', None) is not None and ({'cg_ids', '__cg_id'} - column_names):
        dataset = dataset.flatten_indices(keep_in_memory=True)
        column_names = set(getattr(dataset, 'column_names', []) or [])
    values = [int(cg_id)] * len(dataset)
    if 'cg_ids' not in column_names:
        dataset = dataset.add_column('cg_ids', values)
    if '__cg_id' not in column_names:
        dataset = dataset.add_column('__cg_id', values)
    logger.info(f'Assigned cg_id={cg_id} to dataset rows: {len(dataset)}')
    return dataset


class DatasetLoader(BaseDatasetLoader):

    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
        disable_auto_column_mapping: bool = False,
        preprocessed_dataset_cache_dir: Optional[str] = None,
    ):
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = streaming
        self.hub_token = hub_token
        self.strict = strict
        self.download_mode = download_mode
        self.columns = columns
        self.remove_unused_columns = remove_unused_columns
        self.disable_auto_column_mapping = disable_auto_column_mapping
        self.preprocessed_dataset_cache_dir = preprocessed_dataset_cache_dir

    def _raw_load_dataset_file(self, dataset_path: str) -> HfDataset:
        ext = os.path.splitext(dataset_path)[1].lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': self.streaming, 'num_proc': self.num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        if file_type == 'json' and _is_local_json_array_file(dataset_path):
            logger.info(f'Loading local JSON array file: {dataset_path}')
            return _load_local_json_file(dataset_path)
        kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
        try:
            return hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
        except StopIteration:
            if file_type == 'json':
                logger.warning(f'HF datasets JSON loader failed for {dataset_path}; '
                               'falling back to local JSON parser.')
                return _load_local_json_file(dataset_path)
            raise

    def _finish_preprocess(self, dataset: HfDataset, dataset_meta: DatasetMeta) -> HfDataset:
        """Column rename + preprocess map + column cleanup on an already-loaded dataset."""
        if self.columns:
            dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
        dataset = dataset_meta.preprocess_func(
            dataset,
            num_proc=self.num_proc,
            load_from_cache_file=self.load_from_cache_file,
            strict=self.strict,
            enable_auto_mapping=not self.disable_auto_column_mapping)
        if self.remove_unused_columns:
            dataset = RowPreprocessor.remove_useless_columns(dataset)
        return dataset

    def _build_preprocessed_dataset(self, dataset_path: str, dataset_meta: DatasetMeta) -> HfDataset:
        """Raw-load + preprocess a single dataset file (no DDP gating / no cache I/O)."""
        return self._finish_preprocess(self._raw_load_dataset_file(dataset_path), dataset_meta)

    def _save_preprocessed_cache(self, dataset: HfDataset, cache_path: str) -> None:
        logger.info(f'Saving preprocessed dataset to disk cache: {cache_path}')
        # Write to a temp dir and atomically rename so a crash never leaves a partial
        # cache and a stray concurrent writer can't collide.
        _tmp_path = f'{cache_path}.tmp.{os.getpid()}'
        shutil.rmtree(_tmp_path, ignore_errors=True)
        # Shard the write across processes to speed first-build of large datasets. Only
        # when the dataset is memory-mapped (backed by Arrow cache files): num_proc save
        # on an in-memory dataset pickles the whole table to every worker and is far
        # slower than a single process.
        _save_num_proc = max(1, min(self.num_proc, 16))
        _save_kwargs = ({
            'num_proc': _save_num_proc
        } if _save_num_proc > 1 and getattr(dataset, 'cache_files', None) else {})
        dataset.save_to_disk(_tmp_path, **_save_kwargs)
        try:
            os.rename(_tmp_path, cache_path)
        except OSError:  # another writer won the race; discard our temp copy
            shutil.rmtree(_tmp_path, ignore_errors=True)

    def _load_dataset_path(
        self,
        dataset_path: str,
        dataset_meta: DatasetMeta,
    ) -> HfDataset:
        _use_cache = (self.preprocessed_dataset_cache_dir and not self.streaming and os.path.isfile(dataset_path))
        if not _use_cache:
            # No cache dir: every rank builds in place (original behavior). Order only the
            # raw load across ranks to avoid HF disk-cache write races; the map runs
            # concurrently as before.
            with safe_ddp_context(None, True):
                dataset = self._raw_load_dataset_file(dataset_path)
            return self._finish_preprocess(dataset, dataset_meta)

        from datasets import load_from_disk as _load_from_disk
        os.makedirs(self.preprocessed_dataset_cache_dir, exist_ok=True)
        _cache_key = _make_preprocessor_cache_key(dataset_path, dataset_meta.preprocess_func)
        _cache_path = os.path.join(self.preprocessed_dataset_cache_dir, _cache_key)
        # IMPORTANT: do NOT early-return on `os.path.isdir(_cache_path)` here. A barrier-free
        # fast-path return issues ZERO distributed barriers, whereas the cold-build path below
        # enters safe_ddp_context (2 barriers/rank). Those two branches MUST never be taken by
        # different ranks of the same job. Under robust-training node replacement this is exactly
        # what happens: surviving nodes keep their node-local cache (hit -> fast path -> 0
        # barriers) while a fresh replacement node misses (-> build path -> 2 barriers). The
        # barrier-count mismatch is a collective deadlock -- survivors race ahead into the next
        # NCCL collective while the replacement sits in safe_ddp_context, and the whole job hangs
        # until the watchdog fires (observed: robust run stuck ~1h then SIGTERM). The mtime in
        # _make_preprocessor_cache_key makes it worse -- a re-downloaded dataset gets a fresh key
        # so the replacement can never match the survivors' cache. Route ALL ranks through the
        # SAME coordinated block below; a cache hit simply skips the build inside it, so the
        # per-rank barrier count is identical regardless of per-node cache state.

        # Cold build. Avoid every rank on the node redundantly running the heavy
        # load+preprocess map (e.g. 8 local ranks x num_proc workers, each forking the
        # heavyweight training process and thrashing CPU/memory -> the map slows
        # many-fold under contention).
        if _env_flag('MS_SWIFT_DISTRIBUTED_PREBUILD'):
            # Barrier-free atomic claim (own isdir short-circuit): exactly one rank builds each
            # cache, the rest wait on the filesystem claim then memory-map it. No collective
            # barrier -> inherently safe under asymmetric per-node cache state, and safe to
            # interleave with the rank-distributed pre-pass (different ranks build different
            # datasets at once).
            return self._claim_build_or_wait(dataset_path, dataset_meta, _cache_path)
        # Default: gate so only the (local) master builds; followers wait on the barrier,
        # re-check `isdir`, skip the build, and memory-map the result. The build is run by a
        # single rank, so the DDP barriers inside the preprocess map (RowPreprocessor uses
        # safe_ddp_context) have no peers here -- the followers are parked on this outer
        # barrier, not inside the map -- and would mismatch the barrier count and deadlock.
        # Disable them for the duration of the (sole) build; the outer gating barrier still
        # serializes master-first.
        with safe_ddp_context(None, True):
            if not os.path.isdir(_cache_path):
                with disable_safe_ddp_context_use_barrier():
                    dataset = self._build_preprocessed_dataset(dataset_path, dataset_meta)
                    if isinstance(dataset, HfDataset):
                        self._save_preprocessed_cache(dataset, _cache_path)
        if os.path.isdir(_cache_path):
            logger.info(f'Loading preprocessed dataset from disk cache: {_cache_path}')
            return _load_from_disk(_cache_path)
        # Fallback (non-HfDataset / save failed): all ranks reach this together and build in
        # lockstep, so the map's barriers align normally -- leave them enabled.
        return self._build_preprocessed_dataset(dataset_path, dataset_meta)

    def _claim_build_or_wait(self,
                             dataset_path: str,
                             dataset_meta: DatasetMeta,
                             cache_path: str,
                             *,
                             timeout: float = 7200.0,
                             interval: float = 2.0) -> HfDataset:
        """Build the cache iff we win the atomic claim; otherwise wait for the winner.

        Retry loop so an abandoned claim (the builder crashed/failed and released it)
        can be picked up by a waiter instead of hanging until `timeout`.
        """
        import time
        from datasets import load_from_disk as _load_from_disk
        claim_path = cache_path + '.claim'
        waited = 0.0
        while True:
            if os.path.isdir(cache_path):
                return _load_from_disk(cache_path)
            try:
                os.makedirs(claim_path)  # atomic on POSIX: exactly one rank wins
            except FileExistsError:
                # another rank owns the build; wait for the cache. If that rank fails it
                # removes the claim -> we loop, re-claim, and build it ourselves.
                time.sleep(interval)
                waited += interval
                if waited >= timeout:
                    raise TimeoutError(f'Timed out waiting for preprocessed cache: {cache_path}')
                continue
            # we own the claim. This build is single-rank, so the DDP barriers inside the
            # preprocess map (RowPreprocessor uses safe_ddp_context) have no peers -- other
            # ranks are off building/awaiting other datasets -- and would deadlock. Disable
            # them: with a sole builder there is no cross-rank cache race to coordinate.
            try:
                with disable_safe_ddp_context_use_barrier():
                    dataset = self._build_preprocessed_dataset(dataset_path, dataset_meta)
                    if isinstance(dataset, HfDataset):
                        self._save_preprocessed_cache(dataset, cache_path)
                if not isinstance(dataset, HfDataset):
                    return dataset  # can't cache a non-map dataset; use it directly
            except BaseException:
                shutil.rmtree(claim_path, ignore_errors=True)  # release so another rank can retry
                raise
            return _load_from_disk(cache_path)

    def _load_repo_dataset(
        self,
        dataset_id: str,
        subset: SubsetDataset,
        *,
        use_hf: Optional[bool] = None,
        revision: Optional[str] = None,
    ) -> HfDataset:
        datasets = []
        if os.path.isdir(dataset_id):
            retry = 1
            load_context = nullcontext
            use_hf = True
            dataset_str = f'Use local folder, dataset_dir: {dataset_id}'
            # The dataset downloaded from modelscope will have an additional dataset_infos.json file.
            with safe_ddp_context('dataset_infos_rename'):
                dataset_infos_path = os.path.join(dataset_id, 'dataset_infos.json')
                if os.path.isfile(dataset_infos_path):
                    os.rename(dataset_infos_path, f'{dataset_infos_path}_bak')
        elif dataset_id.startswith('/'):
            raise ValueError(f'The local path does not exist, dataset_id: `{dataset_id}`. '
                             f'os.path.exists(dataset_id): {os.path.exists(dataset_id)}')
        else:
            retry = 3
            load_context = partial(safe_ddp_context, hash_id=dataset_id, use_barrier=True)
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)
            else:
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id)
        logger.info(dataset_str)
        hub = get_hub(use_hf)
        for split in subset.split:
            i = 1
            with load_context():
                while True:
                    try:
                        dataset = hub.load_dataset(
                            dataset_id,
                            subset.subset,
                            split,
                            streaming=self.streaming,
                            revision=revision,
                            download_mode=self.download_mode,
                            hub_token=self.hub_token,
                            num_proc=self.num_proc)
                    except Exception as e:
                        if i == retry:
                            raise
                        i += 1
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset.subset},'
                                     f'split={split} with error: {e}')
                    else:
                        break
            if hasattr(dataset, '_hf_ds'):
                dataset = dataset._hf_ds
                if self.streaming and isinstance(dataset, HfDataset):
                    dataset = dataset.to_iterable_dataset()
            if self.columns:
                dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)
            dataset = subset.preprocess_func(
                dataset,
                num_proc=self.num_proc,
                load_from_cache_file=self.load_from_cache_file,
                strict=self.strict,
                enable_auto_mapping=not self.disable_auto_column_mapping)
            if self.remove_unused_columns:
                dataset = RowPreprocessor.remove_useless_columns(dataset)
            datasets.append(dataset)
        return self.concat_datasets(datasets)

    @staticmethod
    def _select_subsets(subsets: List[str], dataset_meta: DatasetMeta) -> List[SubsetDataset]:
        subset_mapping = {subset.name: subset for subset in dataset_meta.subsets}
        subset_names = list(subset_mapping.keys())
        if not subsets:
            if len(subset_names) <= 1:
                subsets = subset_names
            elif 'default' in subset_names:
                subsets = ['default']
            else:
                raise ValueError(f'Please provide subsets. available subsets: {subset_names}')
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:
            subsets = [subset_name for subset_name in subset_names if not subset_mapping[subset_name].is_weak_subset]

        subsets = [
            subset_mapping[subset_name] if subset_name in subset_mapping else SubsetDataset(subset=subset_name)
            for subset_name in subsets
        ]
        return [subset.set_default(dataset_meta) for subset in subsets]

    def load(
        self,
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        use_hf: Optional[bool] = None,
    ) -> HfDataset:
        if dataset_syntax.dataset_type == 'path':
            dataset = self._load_dataset_path(
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
            )
        else:
            subsets: List[SubsetDataset] = self._select_subsets(dataset_syntax.subsets, dataset_meta)
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision
            datasets = []
            for subset in subsets:
                dataset = self._load_repo_dataset(
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    revision=revision,
                )
                datasets.append(dataset)
            dataset = self.concat_datasets(datasets)
        return dataset


def init_self_cognition_preprocessor(
    dataset_meta: Optional[DatasetMeta],
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> None:
    from .dataset.llm import SelfCognitionPreprocessor
    if dataset_meta is None or model_name is None and model_author is None:
        return
    kwargs = {}
    # zh, en
    for key in ['name', 'author']:
        val = locals()[f'model_{key}']
        if isinstance(val, str):
            val = [val]
        if val is not None and val[0] is not None and (len(val) == 1 or val[1] is None):
            val = (val[0], val[0])
        kwargs[key] = val

    preprocess_funcs = [dataset_meta.preprocess_func]
    preprocess_funcs += [subset.preprocess_func for subset in dataset_meta.subsets if isinstance(subset, SubsetDataset)]
    for preprocess_func in preprocess_funcs:
        if isinstance(preprocess_func, SelfCognitionPreprocessor):
            preprocess_func.set_name_author(**kwargs)
    logger.info_once(f"SelfCognitionPreprocessor has been successfully configured with name: {kwargs['name']}, "
                     f"author: {kwargs['author']}.")


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,  # columns_mapping
    remove_unused_columns: bool = True,
    disable_auto_column_mapping: bool = False,
    preprocessed_dataset_cache_dir: Optional[str] = None,
    # self-cognition
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # zh, en
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """Load and preprocess datasets.

    This function provides a unified interface to load datasets from various sources (HuggingFace,
    ModelScope, or local paths), with support for splitting, shuffling, streaming, and interleaving
    multiple datasets. It also handles self-cognition dataset preprocessing for model training.

    Args:
        datasets: Single dataset name or list of dataset names to load. Can use special syntax
            for advanced configurations (e.g., 'dataset_name#1000' for sampling).
        split_dataset_ratio: Ratio for splitting dataset into train/validation sets.
            Value between 0 and 1. If 0, no validation split is created. Default: 0.
        seed: Random seed for reproducibility. Can be an integer or numpy RandomState object.
            If None, results will be non-deterministic. Default: 42.
        num_proc: Number of processes to use for dataset preprocessing. Set to None for
            streaming mode. Default: 1.
        load_from_cache_file: Whether to load preprocessed data from cache if available.
            Default: True.
        shuffle: Whether to shuffle the dataset(s) after loading. Default: False.
        streaming: Enable streaming mode for large datasets that don't fit in memory.
            When True, num_proc is automatically set to None. Default: False.
        interleave_prob: Probability weights for interleaving multiple datasets. Must have
            same length as datasets list. If None, datasets are concatenated instead. Default: None.
        stopping_strategy: Strategy when interleaving datasets of different lengths:
            - 'first_exhausted': Stop when shortest dataset is exhausted
            - 'all_exhausted': Continue until all datasets are exhausted
            Default: 'first_exhausted'.
        shuffle_buffer_size: Buffer size for shuffling in streaming mode. Larger values
            provide better randomization but use more memory. Default: 1000.
        use_hf: Force using HuggingFace Hub (True) or ModelScope (False). If None,
            it is controlled by the environment variable `USE_HF`, which defaults to '0'.
            Default: None.
        hub_token: Authentication token for accessing private datasets on the hub. Default: None.
        strict: If True, raise exceptions when encountering malformed data rows.
            If False, skip invalid rows with warnings. Default: False.
        download_mode: How to handle existing cached datasets:
            - 'reuse_dataset_if_exists': Use cached version if available
            - 'force_redownload': Always download fresh copy
            Default: 'reuse_dataset_if_exists'.
        columns: Manual column name mapping for datasets. Dictionary mapping source column
            names to target column names (e.g., {'text': 'content'}). Default: None.
        remove_unused_columns: Whether to remove columns not used in preprocessing.
            Helps reduce memory usage. Default: True.
        disable_auto_column_mapping: By default, column names in the dataset are automatically
            mapped. This parameter disables that behavior
            (the `columns` parameter remains effective), defaulting to `False`.
        model_name: Model name for self-cognition task preprocessing. Can be a tuple of
            (Chinese_name, English_name) or list of names. Default: None.
        model_author: Model author for self-cognition task preprocessing. Can be a tuple of
            (Chinese_author, English_author) or list of authors. Default: None.

    Returns:
        A tuple of (train_dataset, val_dataset):
            - train_dataset: The training dataset
            - val_dataset: The validation dataset if split_dataset_ratio > 0, otherwise None

    Examples:
        >>> # Load single dataset
        >>> train_ds, val_ds = load_dataset('AI-ModelScope/alpaca-gpt4-data-zh', split_dataset_ratio=0.1)

        >>> # Load multiple datasets
        >>> train_ds, _ = load_dataset(
        ...     ['AI-ModelScope/alpaca-gpt4-data-zh#500', 'swift/self-cognition#500'],
        ...     model_name=('我的模型', 'MyModel'),
        ...     model_author=('作者', 'Author')
        ... )
    """
    init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    if streaming:
        num_proc = None
    train_datasets = []
    val_datasets = []

    use_hf_default = use_hf
    if use_hf_default is None:
        use_hf_default = True if use_hf_hub() else False

    def _resolve_loader(dataset, *, num_proc_override=None):
        dataset_syntax = DatasetSyntax.parse(dataset)
        _use_hf = dataset_syntax.use_hf or use_hf_default
        # compat dataset_name
        if dataset_syntax.dataset in DATASET_MAPPING:
            dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]
            if dataset_syntax.use_hf is None and dataset_meta.dataset_path is not None:
                dataset_syntax.dataset = dataset_meta.dataset_path
                dataset_syntax.dataset_type = 'path'
            else:
                dataset_syntax.dataset = dataset_meta.hf_dataset_id if _use_hf else dataset_meta.ms_dataset_id
        else:
            dataset_meta = dataset_syntax.get_dataset_meta(_use_hf)
        loader = dataset_meta.loader(
            num_proc=num_proc if num_proc_override is None else num_proc_override,
            load_from_cache_file=load_from_cache_file,
            streaming=streaming,
            hub_token=hub_token,
            strict=strict,
            download_mode=download_mode,
            columns=columns,  # columns_mapping
            remove_unused_columns=remove_unused_columns,
            disable_auto_column_mapping=disable_auto_column_mapping,
            preprocessed_dataset_cache_dir=preprocessed_dataset_cache_dir,
        )
        return loader, dataset_syntax, dataset_meta, _use_hf

    _distributed_prebuild_caches(
        datasets,
        _resolve_loader,
        num_proc=num_proc,
        streaming=streaming,
        preprocessed_dataset_cache_dir=preprocessed_dataset_cache_dir)

    for dataset_idx, dataset in enumerate(datasets):
        loader, dataset_syntax, dataset_meta, use_hf = _resolve_loader(dataset)
        train_dataset = loader.load(dataset_syntax, dataset_meta, use_hf=use_hf)
        train_dataset, val_dataset = loader.post_process(
            train_dataset,
            dataset_sample=dataset_syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            streaming=streaming,
            shuffle=shuffle,
            random_state=seed,
        )
        train_dataset = _with_dataset_cg_id(train_dataset, dataset_idx)
        val_dataset = _with_dataset_cg_id(val_dataset, dataset_idx)
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)

    if interleave_prob is None:
        train_datasets = loader.concat_datasets(train_datasets)
        val_datasets = loader.concat_datasets(val_datasets)
    else:
        train_datasets = loader.interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)
        val_datasets = loader.interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)

    if shuffle:
        if train_datasets:
            train_datasets = loader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
        if val_datasets:
            val_datasets = loader.shuffle_dataset(val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
    return train_datasets, val_datasets
