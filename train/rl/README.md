# Reinforcement learning

RL framework source and agent tools are included under [`../verl/`](../verl/).
The complete training recipe and runtime for reproducing the paper's final RL
runs are not yet available. The SFT environment does not cover this backend.

The 457 core files match `quest-deepresearch-port` revision
`b1f05f92ad76e70e5684747d97d0dc6b7920834f`. Reward, tool and session code is
preserved in `recipe/deepresearch/`.

| Path under `train/verl/` | Purpose |
|---|---|
| `verl/experimental/fully_async_policy/` | Asynchronous rollout and training |
| `verl/workers/` | Actor, rollout and reference workers |
| `verl/models/mcore/` | Megatron model integration, including fused forward code |
| `recipe/deepresearch/agent_loop/` | Multi-turn agent interaction |
| `recipe/deepresearch/tools/` | Search, visit, scholar and Python tools |
| `recipe/deepresearch/reward.py` | Reward calculation |
| `recipe/deepresearch/config/` | Agent and service configuration |

The presence of fused forward code does not establish a validated 122B launch.
The final resolved configuration, matching runtime and successful training
records are needed before publishing reproduction commands.
