# Copyright (c) ModelScope Contributors. All rights reserved.
import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    # Preload TE before swift/peft for nsys/megatron CLI symbol resolution; skip when
    # MS_SWIFT_SKIP_TE_PRELOAD=1 to match pre-patch memory (~63 GiB vs ~71 GiB peak).
    if os.environ.get('MS_SWIFT_SKIP_TE_PRELOAD', '0') != '1':
        import transformer_engine  # noqa: F401 — RTLD_GLOBAL; must precede swift/peft
    from swift.megatron import megatron_sft_main
    megatron_sft_main()
