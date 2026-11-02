# verl RL backend

This directory vendors an Axion development snapshot of [verl](https://github.com/verl-project/verl),
source revision `b1f05f92ad76e70e5684747d97d0dc6b7920834f`. It retains the `verl`
namespace, Apache-2.0 license and upstream notices. The revision identifies
the imported snapshot, not a public upstream commit.

The [Axion RL recipe](../../axion/training/rl/README.md) adds narrowly scoped
hooks for complete prompt-group filtering, session expansion, optimizer-update
accounting and optional validation. These are enabled by
`async_training.axion_recipe`. Routing adapters preserve captured routes across
generation resumes and support BSHD replay and padding in the Megatron actor.
Modified files carry change notices.

Install this backend into a separately prepared RL environment with `--no-deps`.
The upstream `vllm` extra is too old for the Axion R3 recipe; follow the RL guide
for required capabilities and the limits of available runtime records.
