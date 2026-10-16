# Axion

Data generation, training and evaluation for the Axion models.

```
datagen/    generate more training data of the two kinds the models were trained on
train/      SFT then RL
eval/       build the benchmark question sets and grade rollouts on them
```

Each directory has its own README with the commands. This page is the map and
the one-time setup.

## Setup

```bash
conda create -n axion python=3.12
conda activate axion

pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install -r requirements.txt

pip install -e train/verl                 # RL
pip install -e train/sft/LlamaFactory     # SFT
```

`requirements.txt` lists the versions the released models were trained and
evaluated in. Two dependencies are not in it: `transformer_engine`, which has
to be built against your own CUDA and torch, and the two frameworks above,
which are vendored here.

Grading HiPhO needs one more set of packages that conflicts with the training
stack; `eval/README.md` explains how to install it into a separate prefix.

### Credentials

Search, page fetching and the judge models are external services. Pass their
keys through the environment, or put them in a file you source. Never put them
on a command line and never commit them.

```bash
# ~/.axion_keys   (chmod 600)
export SERPER_KEY_ID=...        # serper.dev, for search and scholar
export JINA_API_KEYS=...        # jina.ai, for page fetching; comma-separate several
export OPENAI_API_KEY=...       # the model that writes and audits questions
```

`datagen/stage1/README.md` and `datagen/stage2/README.md` list what each
generator needs; `train/README.md` covers the services RL expects.

## Check the installation

```bash
bash train/smoke_test.sh
```

No GPUs required. It parses every config and script, checks the recipes reject
missing inputs, and starts the code-execution sandbox to confirm it runs code
with its isolation in place.

## Pipeline

```
datagen/stage1   closed-book problems, solved with Python and no retrieval
datagen/stage2   tool-use problems built by walking a public database
       |
       v
train/sft        stage 1 trains from the base model, stage 2 continues from it
       |
       v
train/rl         fully-async GRPO from the stage-2 checkpoint
       |
       v
eval/            five benchmarks
```

## Licensing

This project is released under the Axion Non-Commercial Research and
Responsible Use License in `LICENSE`. **Commercial use is prohibited**, and
Schedule A of that license sets out binding acceptable-use and safety
conditions.

`LICENSE` covers this project's own materials. The vendored third-party
components keep their own licenses and are not restricted by it —
`THIRD_PARTY.md` lists which paths those are, where they come from, and which
files were modified.

