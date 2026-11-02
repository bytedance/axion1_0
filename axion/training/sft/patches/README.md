# GDN context parallel backport

`mcore_v0.17.1_gdn_cp_pr2642.diff` is copied byte for byte from the recovered
training runtime. It backports NVIDIA/Megatron-LM PR2642 to Megatron-core0.17.1
and changes `core/ssm/gated_delta_net.py` and
`core/transformer/transformer_config.py`.

Upstream: https://github.com/NVIDIA/Megatron-LM/pull/2642

These changes retain NVIDIA/Megatron-LM's Apache-2.0 terms and notices. The
repository's `LICENSE` contains the Apache-2.0 license text. Original/resulting
source hashes are enforced by `../runtime.py`; `../provenance.json` records the
original patch hash. Both files must be patched together. Swift's corresponding
guard and checkpoint changes are recorded in `../../../../third_party/swift/SOURCE.json`.
