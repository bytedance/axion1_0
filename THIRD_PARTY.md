# Third-party components

The Axion Non-Commercial Research and Responsible Use License in `LICENSE`
covers this project's own materials. Under Section 1.3 of that license, the
components listed here are **not** Covered Materials: they are third-party
works redistributed under their own terms, and nothing in `LICENSE` restricts
your rights in them. The same components can be obtained directly from their
upstream projects under those terms.

| Path | Project | License |
|---|---|---|
| `train/verl/` | [verl](https://github.com/verl-project/verl) | Apache-2.0 — `train/verl/LICENSE`, notice in `train/verl/Notice.txt` |
| `train/verl/recipe/deepresearch/` | DeepResearch recipe, from [OSU-NLP-Group/QUEST](https://github.com/OSU-NLP-Group/QUEST) | repository MIT (Copyright 2026 OSU-NLP-Group); source files Apache-2.0 (Copyright 2025 DeepResearch authors) |
| `train/sft/LlamaFactory/` | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | Apache-2.0 — `train/sft/LlamaFactory/LICENSE` |
| `eval/hipho/official_impl/` | HiPhO grader, from [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | Apache-2.0 (Copyright 2025 Garena Online Private Limited) — see that directory's README |

## Modifications

Files changed from their upstream versions carry a change notice in the file,
as required by Apache-2.0 Section 4(b):

- 32 files under `train/verl/recipe/deepresearch/`
- `train/sft/LlamaFactory/src/llamafactory/train/mca/workflow.py`

The files under `eval/hipho/official_impl/` are unmodified; `shim.py` in that
directory is this project's own code and adapts to them.

## Python dependencies

`requirements.txt` pins 70 packages, all under permissive licenses
(Apache-2.0, MIT, BSD, the matplotlib license, Unlicense). None is copyleft
and none carries non-commercial terms. PyTorch (BSD-3-Clause) is installed
separately from the CUDA wheel index.

Two packages are deliberately left out of the pinned set because their
licenses do not fit this project's terms. Both imports are guarded and the
code works without them; install them yourself, under their own licenses, if
you need the optional features:

- PyMuPDF (AGPL-3.0 or a paid commercial license) — PDF parsing in
  `obj_task_eval`
- pycountry (LGPL-2.1-only) — country names in `gen_ranked_list.py`

## External data

The Stage II generators query public databases at run time. No data from them
is redistributed in this repository. Each generator names its source and that
source's license in its module docstring; `datagen/stage2/README.md` lists
them together.
