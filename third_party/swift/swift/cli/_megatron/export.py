# Copyright (c) ModelScope Contributors. All rights reserved.
import os

if __name__ == '__main__':
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    import transformer_engine  # noqa: F401 — RTLD_GLOBAL; must precede swift/peft
    from swift.megatron import megatron_export_main
    megatron_export_main()
