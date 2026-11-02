# verl RL backend

This directory vendors `data/open_verl`, branch `quest-deepresearch-port`,
revision `b1f05f92ad76e70e5684747d97d0dc6b7920834f`. The framework retains its
`verl` namespace and Apache-2.0 license and notices.

The Axion recipe lives in [`axion/training/rl`](../../axion/training/rl/README.md).
The fully asynchronous Megatron configuration resolves base trainer settings
through `pkg://verl.trainer.config`, allowing the backend and Axion recipe to be
installed separately. Backend computation code is unchanged by this relocation.

See the [RL guide](../../axion/training/rl/README.md) for environment requirements
and the limits of the available reproduction records.
