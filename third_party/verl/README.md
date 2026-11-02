# verl RL backend

Axion uses a development snapshot of [verl](https://github.com/verl-project/verl),
revision `b1f05f92ad76e70e5684747d97d0dc6b7920834f`. This is the imported
snapshot's revision, not a public upstream commit. The `verl` namespace,
Apache-2.0 license and upstream notices are retained.

The [RL recipe](../../axion/training/rl/README.md) adds prompt-group filtering,
session expansion, optimizer-update accounting and optional validation through
`async_training.axion_recipe`. Routing adapters preserve MoE routes across
resumed generations and replay them in the Megatron actor.

Install this backend in a separate RL environment with `--no-deps`. Follow the
RL guide for runtime requirements; the upstream `vllm` extra predates the
routing-replay support used by this recipe.
