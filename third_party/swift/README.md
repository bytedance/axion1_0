# Swift SFT backend

Axion uses a development snapshot of [ms-swift](https://github.com/modelscope/ms-swift),
revision `06686330164c1221d069d774a0dcb5092a8b2260` (`ms_swift` 4.4.0.dev0).
This is the imported snapshot's revision, not a public upstream commit.
`SOURCE.json` records the source hashes and Axion changes. Upstream Apache-2.0
notices are retained.

Install from the repository root in the SFT environment:

```bash
pip install --no-deps -e third_party/swift
```

The [SFT guide](../../axion/training/sft/README.md) covers training and export.
Context parallelism requires both this backend and the bundled Megatron-core
0.17.1 GDN backport. The backend also includes the DCP writer and export fixes.

The original training image was mutable; its exact source and complete
package versions are unavailable. Known differences and runtime records are
listed in [provenance.json](../../axion/training/sft/provenance.json).

Local and HTTP media loading, standard message preprocessing and local
checkpoint storage are supported. Optional private storage transports and
dataset formats are unavailable; their adapters report this when selected.
