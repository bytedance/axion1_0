# HiPhO official grader, vendored

These three files contain the official HiPhO grading implementation, copied
unmodified from
[open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit),
`main` branch, retrieved 2026-09-30:

| file | upstream path |
|--|--|
| `hipho.py` | `vlmeval/dataset/hipho.py` -- `HiPhODataset`, the grading logic |
| `hipho_verifier.py` | `vlmeval/dataset/utils/hipho_verifier.py` -- answer-level verification |
| `hipho_prompt_inference.py` | `vlmeval/dataset/utils/hipho_prompt_inference.py` -- prompt templates |

`hipho_verifier.py` carries an Apache-2.0 header (Copyright 2025 Garena Online
Private Limited). VLMEvalKit is Apache-2.0.

`shim.py` supplies the small subset of VLMEvalKit interfaces used by the
grader, so these files can run without installing the full toolkit. Unsupported
interfaces raise an error.

## Dependencies

The official files import `math_verify`, `sympy`, `pylatexenc` and `pandas`;
`math_verify` in turn needs `latex2sympy2_extended` and the ANTLR runtime.
Install the evaluation extra and `../../requirements.txt` as described in
the [evaluation guide](../../README.md).

## Refreshing

Copy the three files again from the same upstream paths and record the new
retrieval date here. Keep upstream files unmodified and update `shim.py` for
interface changes.

To check that the grader still loads:

```bash
python -c "import shim; s, m = shim.new_scorer(); \
print(m.HiPhODataset.__module__, hasattr(s, '_evaluate_single_problem'))"
```
