# HiPhO official grader, vendored

HiPhO is graded by running the benchmark's own implementation, not a
reimplementation of it. These three files are copied **unmodified** from
[open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit),
`main` branch, retrieved 2026-09-30:

| file | upstream path |
|--|--|
| `hipho.py` | `vlmeval/dataset/hipho.py` -- `HiPhODataset`, the grading logic |
| `hipho_verifier.py` | `vlmeval/dataset/utils/hipho_verifier.py` -- answer-level verification |
| `hipho_prompt_inference.py` | `vlmeval/dataset/utils/hipho_prompt_inference.py` -- prompt templates |

`hipho_verifier.py` carries an Apache-2.0 header (Copyright 2025 Garena Online
Private Limited). VLMEvalKit is Apache-2.0.

`shim.py` is the only file here that is ours. It fakes the `vlmeval` modules
those three import, so they load outside VLMEvalKit, and builds a
`HiPhODataset` instance usable for grading without running `__init__` (which
would download the dataset and build a dataframe). Every faked symbol that the
grading path does not need raises if called, so execution cannot silently drift
onto a different code path.

## Dependencies

The official files import `math_verify`, `sympy`, `pylatexenc` and `pandas`;
`math_verify` in turn needs `latex2sympy2_extended` and the ANTLR runtime.
These are pinned in `../../requirements.txt` -- the pinned versions are the ones
the vendored code was tested against.

## Refreshing

Copy the three files again from the same upstream paths and record the new
retrieval date here. Do not edit them: `shim.py` adapts to them, not the other
way round. If an upstream change breaks the shim, fix the shim.

To check that the grader still loads:

```bash
python -c "import shim; s, m = shim.new_scorer(); \
print(m.HiPhODataset.__module__, hasattr(s, '_evaluate_single_problem'))"
```
