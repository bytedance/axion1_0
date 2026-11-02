# Swift SFT backend

This directory vendors an Axion development snapshot of [ms-swift](https://github.com/modelscope/ms-swift),
source revision
`06686330164c1221d069d774a0dcb5092a8b2260` (`ms_swift` 4.4.0.dev0).
It provides the `swift` and `megatron` commands and `swift.megatron` API.
Apache-2.0 notices are retained; `SOURCE.json` records every source/file hash
and the exact publication and recovered-runtime adaptations.

Install from the repository root into the separately prepared SFT environment:

```bash
pip install --no-deps -e third_party/swift
```

Follow [the SFT guide](../../axion/training/sft/README.md) for the single recovered 32-GPU,
128K two-stage recipe and the required Megatron-core 0.17.1 GDN CP backport.
The Swift CP/FSDP guard change is paired with that external patch; lifting
the guard alone does not implement GDN context parallelism. The recovered
DCP writer and single-process export fixes are also included.

The original training jobs used a mutable training image. This development
snapshot is not proven byte-identical to that image; the known differences
and environment gaps are recorded in [provenance](../../axion/training/sft/provenance.json).
The revision identifies the imported source snapshot, not a public upstream commit.
Distributed numerical parity and paper metrics have not been rerun here.

Private TOS and HDFS transport adapters are unavailable in this distribution.
The TOS adapter raises an explicit error and performs no credential loading,
proxy mutation or remote storage access. The optional HDFS callback reports
that its transport is unavailable. Local and HTTP image/video loading and
local checkpoint computation retain their source implementations.

Package discovery includes the upstream `swift.ray` namespace so built wheels
retain every vendored module. Private data tools, guidelines, cluster launchers,
floating installers and unrelated example pipelines are not included.

Optional private dataset transforms retain import-compatible adapters that fail
explicitly when selected. Standard message preprocessing and its cache version
are unchanged; joint LM/regression preprocessing is not included.
