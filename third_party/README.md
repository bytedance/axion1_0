# Training backends

These are separate Python distributions used by Axion's successive training
stages. Their upstream import names are preserved.

| Backend | Stage | Python namespace | Source and setup |
|---|---|---|---|
| Swift / Megatron-FSDP | SFT I and II | `swift` | [Source](swift/README.md) · [SFT guide](../axion/training/sft/README.md) |
| verl | RL | `verl` | [Source](verl/README.md) · [RL guide](../axion/training/rl/README.md) |

Axion-specific launchers, configuration, agent logic and rewards live in
[`axion/training`](../axion/training/README.md). Install each backend in its own
prepared environment; the root package does not install them automatically.
