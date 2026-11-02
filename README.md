<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/axion-logo-dark.svg">
    <img src="assets/axion-logo.svg" width="260" alt="Axion 1.0">
  </picture>

  <h1>Open Agent Foundation Models, Simplified</h1>

  <p>
    <a href="https://axion1-0.github.io/"><img src="https://img.shields.io/badge/Project-Website-2563EB?style=flat-square" alt="Project website"></a>
    <a href="https://axion1-0.github.io/paper/axion.pdf"><img src="https://img.shields.io/badge/Read-Technical_Report-0F766E?style=flat-square" alt="Technical report"></a>
    <a href="https://huggingface.co/datasets/Axion1-0/axion-training-data"><img src="https://img.shields.io/badge/Data-Hugging_Face-0F766E?style=flat-square" alt="Training data on Hugging Face"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2563EB?style=flat-square" alt="Apache 2.0 license"></a>
  </p>

  <p>
    <a href="https://jyqhahah.github.io/">Yuqi Jia</a><sup>1</sup> · <a href="https://sunsmarterjie.github.io/">Yunjie Tian</a><sup>2,†</sup> · <a href="https://www.linkedin.com/in/jiayu-zheng-5aa231250/">Jiayu Zheng</a><sup>2</sup> · <a href="https://joliang17.github.io/">Yijun Liang</a><sup>3</sup> · <a href="https://jingchensun.github.io/">Jingchen Sun</a><sup>4</sup> · <a href="https://www.linkedin.com/in/mengyu-zhao-7ab19274/">Mengyu Zhao</a><sup>2</sup> · Di Fu<sup>2</sup>
  </p>
  <p>
    <sup>1</sup> Duke University &nbsp; <sup>2</sup> ByteDance &nbsp; <sup>3</sup> University of Maryland, College Park &nbsp; <sup>4</sup> University at Buffalo<br>
    <sup>†</sup> Project Lead
  </p>
</div>

## 1. Highlights

- **Three-stage training.** Two stages of supervised fine-tuning (SFT) followed
  by reinforcement learning (RL), without dedicated mid-training, DPO, or
  on-policy distillation.
- **Scientific reasoning and tool use.** Axion 1.0-122B scores **59.0** on full
  HLE with restricted website access, **87.0** on FrontierScience Olympiad, and
  **66.7** on FrontierScience Research. Axion 1.0-35B scores **62.2** on SEAL-0.
- **Verifiable training data.** Reasoning problems are strengthened using
  feedback from the base model. Tasks requiring tools are constructed from
  public data sources with programmatically determined reference answers.
- **Open training code.** The repository includes data generation and
  filtering, prompts, two-stage SFT, asynchronous RL, and benchmark preparation
  and grading.

## 2. Results

<p align="center">
  <a href="assets/benchmark-results.png">
    <img src="assets/benchmark-results.png" width="100%" alt="Axion 1.0 compared with selected open and proprietary models on HLE, BioMysteryBench, SEAL-0, HiPhO, and both FrontierScience splits">
  </a>
</p>

Selected results from Table 1 of the
[technical report](https://axion1-0.github.io/paper/axion.pdf#page=9):

| Model | HLE | BMB | SEAL-0 | HiPhO | FS Olympiad | FS Research |
|:---|---:|---:|---:|---:|---:|---:|
| Agents-A1 | 47.6 | — | 56.4 | 46.4 | 79.0 | 40.0 |
| Apodex 1.1 (Agent Team) | 56.1 | 35.3 | — | — | — | 63.3 |
| GPT-5.6 Sol | — | 47.1* | 53.2* | 42.9* | 79.0* | 63.3* |
| Gemini 3.1 Pro | 51.4 | 35.3* | 53.2* | 38.5* | 85.0* | 26.7* |
| Claude Fable 5 | 63.8 | 46.1 | — | — | — | — |
| **Axion 1.0-35B** | **52.2** | **35.3** | **62.2** | **45.7** | **86.0** | **56.7** |
| **Axion 1.0-122B** | **59.0** | **41.7** | **58.6** | **46.2** | **87.0** | **66.7** |

BMB denotes the human-difficult subset of BioMysteryBench; FS denotes
FrontierScience. * marks baselines evaluated by the Axion team; other baseline
scores are from official reports. — denotes an unreported result.

### 2.1 Models

| Model | Base model | Total / active parameters |
|:---|:---|:---:|
| Axion 1.0-35B | Qwen3.6-35B-A3B | 35B / 3B |
| Axion 1.0-122B | Qwen3.5-122B-A10B | 122B / 10B |

Both models in the paper use SFT and RL. The paper also reports 4B and 9B
variants trained with SFT only.

Training data is available on
[Hugging Face](https://huggingface.co/datasets/Axion1-0/axion-training-data).
Model weights are coming soon.

## 3. Training Recipe

<p align="center">
  <a href="assets/data-construction.svg">
    <img src="assets/data-construction.svg" width="100%" alt="Two-stage data construction: model-aware strengthening of self-contained reasoning problems, followed by deterministic construction and verification of tool-dependent tasks">
  </a>
</p>

### 3.1 Stage I: Self-Contained Reasoning

Stage I uses **10,819 trajectories** of scientific and mathematical reasoning.
Public seed problems are rewritten into harder questions. Problems solved by
the base model are further strengthened through composition, parameterization,
range shifts, or case analysis, with reference answers checked after each
rewrite. Solvers may use Python, but not search or webpage retrieval.

### 3.2 Stage II: Tool-Dependent Reasoning

Stage II uses **7,544 trajectories**, approximately **312M tokens**. Programs
filter records, select a unique entity, and compute a reference answer before
an LLM formulates the question. Solvers must find the required evidence through
tools; construction data and answers are withheld during solving. Questions
solvable without retrieval are removed.

Generators cover individual entities, ranked lists, and chains across
databases, including the NASA Exoplanet Archive, OurAirports, OpenAlex,
UniProt, and Ensembl.

### 3.3 Reinforcement Learning

Starting from the Stage II checkpoint, **asynchronous GRPO** trains on agent
interactions with search, webpage visits, scholarly retrieval, and Python.
Correct final answers receive full reward; verified intermediate progress can
receive partial credit. Context condensation divides long rollouts into
sessions for training.

### 3.4 Training Infrastructure

The paper's SFT runs use a 128K context window with FSDP, expert parallelism,
and context parallelism. RL separates rollout workers, trainers, and auxiliary
services into distinct GPU pools. Token-In-Token-Out (TITO) preserves token
IDs, and Rollout Routing Replay (R3) aligns MoE expert choices between rollout
and training.

Two-stage SFT costs reported in Section 5.1:

| Model | GPUs | Stage I | Stage II | Total GPU-hours |
|:---|---:|---:|---:|---:|
| Axion 1.0-35B | 32 | 1h 02m | 1h 46m | ≈90 |
| Axion 1.0-122B | 32 | 3h 32m | 3h 58m | 240 |

These costs are reported in the paper. The released 122B SFT recipe uses
32 GPUs, a 128K context window, and a global batch of 8. Its saved training
arguments record micro-batch size 1; Section 5.1 lists 2. The released
[SFT recipe](axion/training/sft/README.md) preserves the saved training arguments.

## 4. Getting Started

Clone the repository and follow the [training guide](axion/training/README.md)
for environment setup, two-stage SFT and checkpoint export. SFT uses Swift with
Megatron-FSDP; RL uses verl. They implement successive stages of the same
training pipeline.

```bash
git clone https://github.com/bytedance/axion1_0.git
cd axion1_0
```

```text
axion/
  training/
    sft/         Two-stage SFT, runtime patches and checkpoint export
    rl/          Agent rollouts, rewards, tools and RL configuration
  datagen/       Stage I and Stage II data construction
  evaluation/    Benchmark preparation and grading
third_party/
  swift/         SFT backend
  verl/          RL backend
tests/           Regression and packaging checks
```

The training guides provide one 122B pipeline: two-stage SFT followed by
paper-based asynchronous GRPO. The RL implementation includes session training,
TITO and R3; its distributed GPU execution and benchmark reproduction remain
unverified.

For data construction and evaluation, use a separate Python 3.11+ environment:

```bash
python -m pip install -e ".[data,evaluation]"
```

### 4.1 Documentation

| Guide | Contents |
|:---|:---|
| [Stage I data](axion/datagen/stage1/README.md) | Problem generation, rewriting, answer verification, and trajectory filtering |
| [Stage II data](axion/datagen/stage2/README.md) | Database generators, question filtering, scoring, and trajectory processing |
| [SFT](axion/training/sft/README.md) | Runtime setup, data requirements, two-stage training, and checkpoint export |
| [RL](axion/training/rl/README.md) | Paper recipe, task selection, services, and launch |
| [Evaluation](axion/evaluation/README.md) | Dataset preparation and grading for FrontierScience, HiPhO, HLE-130, and SEAL-0 |

Evaluation scripts prepare datasets and grade rollouts collected with an
external agent harness. The evaluation guide documents the required format.

## 5. License

[Apache 2.0](LICENSE).

## 6. Citation

```bibtex
@techreport{jia2026axion,
  title  = {Axion 1.0: Open Agent Foundation Models, Simplified},
  author = {Jia, Yuqi and Tian, Yunjie and Zheng, Jiayu and Liang, Yijun
            and Sun, Jingchen and Zhao, Mengyu and Fu, Di},
  year   = {2026},
  url    = {https://axion1-0.github.io/paper/axion.pdf}
}
```

