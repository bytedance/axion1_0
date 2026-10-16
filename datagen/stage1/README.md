# Stage I — Model-Aware Reasoning Data

The released Stage I set is 10,819 trajectories, shipped with the paper.
**This page is how to generate more of the same kind** — against whatever base
model you are about to train.

Closed-book problems: the model gets Python and no search, so what is being
trained is its own derivation and computation. Difficulty is calibrated against
**your** base model rather than in the abstract: a problem it already solves is
rewritten until that particular solution route stops working.

The steps are:

1. **Generate a harder question** from a seed problem, then solve and audit it
   to fix a reference answer
2. **Let your base model attempt each question once.** Wrong → the item is
   already hard enough, keep it. Right → step 3
3. **Rewrite the question against the route your base model actually took**,
   then solve and audit again
4. **Sample an advanced model** on the whole bank in a Python-only harness
5. **Judge and filter** into SFT data

## Install

```bash
pip install openai
export OPENAI_API_KEY=...            # steps 1, 3, 5
export OPENAI_BASE_URL=...           # OpenAI-compatible endpoint
```

## A worked example

One real chain from the released set, a seed from DeepMath:

```
seed       lim (n!)^(1/n) / n  as n → ∞                               answer 1/e
           A textbook limit. The base model has almost certainly seen it, so
           the seed itself is never published or trained on — it is material.

step 1     F(z) = Σ_{n≥1} (n!/n^n) z^n.  For which integers m does F(m)
           converge?                                        answer {-2,-1,0,1,2}
           The radius of convergence is 1 / (the seed's limit) = e, so the seed
           answer is a step the solver cannot route around.

step 2     The base model solves it: gets |m| < e, then lists the integers.
           → it needs to be harder

step 3     For each non-negative integer k, let
           S_k = {m ∈ Z : F(m² - km - 2) converges}.
           Find every k for which S_k has exactly 4 elements.   answer {1,2,3}
           Listing integers no longer suffices: the condition has to be turned
           into -2 ≤ m² - km - 2 ≤ 2, including the boundary cases, and solved
           per k. Reusing the old answer lands on a wrong one.
```

## 1. Seeds → questions with reference answers

```bash
python gen_from_seed.py --seed_bench seeds.jsonl --n 500 --out bank.jsonl \
    --model gpt-5.5
```

`--seed_bench` takes a jsonl of `{"question": ..., "answer": ...}`. Any public
problem set with answers works — the example above uses a seed from
[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) (MIT).

Seeds are material only: a seed never enters the output, and nothing derived
from one keeps its wording. Check a new seed set against whatever you intend to
evaluate on before using it.

For each seed the script runs three prompts (all in `prompts.py`):

- **GEN** — name the mathematical object the seed is about, then ask a deeper
  question about that same object whose solution has to pass through the seed's
  answer. Arithmetic on the seed's answer is explicitly forbidden, as is
  anything requiring an external lookup. The answer form is constrained to
  something checkable: an integer, a finite set, a fraction, a closed form, or
  a tuple.
- **SOLVE** — write out the complete derivation, ending in a single final
  answer, or declare the problem ill-posed.
- **AUDIT** — re-read that derivation **without being shown the answer** and
  look for algebra slips, unjustified jumps, dropped cases, and misreadings.
  Only an item that passes the audit keeps its answer.

Only an item the audit returns CORRECT on keeps its answer and goes on to
step 2; anything else is dropped.

## 2. Split on what your base model can already do

Run your base model once per question in a Python-only harness, then judge it.
Items it gets **wrong** are already hard enough for it; keep them as they are.
Items it gets **right** go to step 3, together with the solution text it
produced — that text is what the attacker needs.

## 3. Rewrite against the route it took

```bash
python gen_from_solution.py --judgment judgment_results.json --traj-dir rollouts/ \
    --out attacked.jsonl --model gpt-5.5
```

`--judgment` is your base model's judged rollouts, `--traj-dir` the rollouts
themselves — the rewrite needs the solution text, not just the final answer.

The new question is then solved and audited exactly as in step 1, and kept only
if the audit returns CORRECT.

## 4–5. Sample, judge, assemble

Sample an advanced model over the whole bank in the same Python-only harness.
The harness is not part of this repo; the only outside service it needs here is
a sandbox to run code in (ours took `E2B_API_KEY`, or a self-hosted endpoint in
`SANDBOX_FUSION_ENDPOINT`). No search or browsing keys are involved, by design.
Judge each rollout against the reference answer — `prompts.judge()` does this
with the same prompt we used — then flatten the accepted ones:

```bash
python ../utils/to_sft.py --results rollouts/ \
    --judgment judgment_results.json --out sft.jsonl \
    --tools PythonInterpreter
```

`--tools PythonInterpreter` is what enforces the closed-book setting: a
trajectory that called a retrieval tool is rejected, so a harness configured
for the wrong stage shows up here instead of silently contaminating the data.

A trajectory is kept only if the answer is judged correct, it ended on its own
`<answer>` rather than at a turn or context limit, every assistant turn has a
matching `<think>` pair, no tool call is malformed, and the question text is
not a duplicate. The same script handles Stage II, so both stages come out in
one format.
