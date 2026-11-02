# Third-party components

This project's own code and associated documentation are released under the
Apache License, Version 2.0, in `LICENSE`. The third-party components listed
below retain their original licenses and notices. They can also be obtained
directly from their upstream projects under those terms.

| Path | Project | License |
|---|---|---|
| `third_party/verl/` | [verl](https://github.com/verl-project/verl) | Apache-2.0 — `third_party/verl/LICENSE`, notice in `third_party/verl/Notice.txt` |
| `axion/training/rl/` | DeepResearch recipe, from [OSU-NLP-Group/QUEST](https://github.com/OSU-NLP-Group/QUEST) | repository MIT (Copyright 2026 OSU-NLP-Group); source files Apache-2.0 (Copyright 2025 DeepResearch authors) |
| `third_party/swift/` | [ms-swift](https://github.com/modelscope/ms-swift) | Apache-2.0 — `third_party/swift/LICENSE`; source and changes in `third_party/swift/SOURCE.json` |
| `axion/training/sft/patches/` | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) GDN context-parallel backport | Apache-2.0 — patch provenance and hashes in `axion/training/sft/provenance.json` |
| `axion/evaluation/hipho/official_impl/` | HiPhO grader, from [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | Apache-2.0 (Copyright 2025 Garena Online Private Limited) — see that directory's README |

## Modifications

Files changed from their upstream versions carry a change notice in the file,
as required by Apache-2.0 Section 4(b):

- Modified DeepResearch files under `axion/training/rl/`
- The fully asynchronous trainer configuration, rollout/trainer group handling,
  optional validation, routing capture/replay and Megatron padding adapters in
  `third_party/verl/`; Axion-specific training hooks are enabled by the paper recipe
- Swift publication and training-runtime adaptations listed in `third_party/swift/SOURCE.json`

The paper reward implementation also adapts Axion/sciteacher verification logic;
its source fingerprint is recorded in `axion/training/rl/provenance.json`.

The files under `axion/evaluation/hipho/official_impl/` are unmodified; `shim.py` in that
directory is this project's own code and adapts to them.

## Python dependencies

Data and evaluation dependencies are declared in `pyproject.toml`. SFT runtime
requirements and upstream compatibility ranges are in `axion/training/sft/` and
`third_party/swift/`; these are separate from the RL framework's package metadata.
Dependencies retain their respective licenses.

Two optional integrations have separate license terms and are not installed by
the root package extras:

- PyMuPDF (AGPL-3.0 or a commercial license): PDF parsing in `obj_task_eval`.
- pycountry (LGPL-2.1-only): country names in `gen_ranked_list.py`.

## External data

The Stage II generators query public databases at run time. No data from them
is redistributed in this repository. Each generator names its source and that
source's license in its module docstring; `axion/datagen/stage2/README.md` lists
them together.
