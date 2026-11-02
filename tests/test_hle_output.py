"""Keep benchmark downloads outside an installed, possibly read-only package."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(os.environ.get('AXION_REPO', Path(__file__).resolve().parents[1]))


def test_hle_download_and_outputs_follow_out_dir(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'hle_prepare', ROOT / 'axion/evaluation/hle/prepare.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    out = tmp_path / 'new' / 'hle'
    ids = (ROOT / 'axion/evaluation/hle/subset_130_ids.txt').read_text().splitlines()
    ids += [f'fixture-{i}' for i in range(2000 - len(ids))]
    rows = [{'id': i, 'question': 'Fixture question', 'answer': '42',
             'rationale': 'Do not export this field'} for i in ids]
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        assert Path(kwargs['local_dir']) == out / '_hle_raw'
        return str(out / '_hle_raw' / 'test.parquet')

    class Frame:
        columns = list(rows[0])
        def __len__(self):
            return len(rows)
        def iterrows(self):
            return ((i, types.SimpleNamespace(to_dict=lambda row=row: row))
                    for i, row in enumerate(rows))

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, 'pandas', types.SimpleNamespace(read_parquet=lambda path: Frame()))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--out-dir', str(out)])
    assert prepare.main() == 0
    assert len(downloads) == 1
    subset = [json.loads(x) for x in (out / 'hle_text_only_130.jsonl').read_text().splitlines()]
    assert len(subset) == 130
    assert {tuple(sorted(x)) for x in subset} == {('answer', 'id', 'question')}
