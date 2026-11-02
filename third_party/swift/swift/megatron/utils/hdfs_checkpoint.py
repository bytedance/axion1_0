# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for Axion publication (2026-11-02): remove the private HDFS transport.
"""Compatibility API for an unavailable private checkpoint storage transport.

The optional callback detects that the transport is unavailable. Direct storage
operations fail explicitly. Local Megatron checkpoint save/load is unaffected.
"""
from typing import Callable, Optional


def _unsupported() -> None:
    raise NotImplementedError(
        'HDFS checkpoint transport is not included in this distribution. '
        'Stage inputs locally and use local checkpoint storage.')


def hdfs_client_available() -> bool:
    return False


def detect_idc() -> str:
    _unsupported()


def get_hdfs_base() -> Optional[str]:
    return None


def remap_hdfs_path(src: str) -> str:
    return src


def hdfs_exists(path: str) -> bool:
    _unsupported()


def hdfs_rm(path: str) -> None:
    _unsupported()


def wait_for_pending_uploads(timeout: Optional[float] = None) -> bool:
    # This adapter cannot start uploads, so there is nothing to join.
    return True


def hdfs_upload(
    src,
    dst: str,
    blocking: bool = False,
    on_success: Optional[Callable[[], None]] = None,
) -> None:
    _unsupported()


def hdfs_download(src: str, dst, blocking: bool = True) -> None:
    _unsupported()


def validate_hdfs_checkpoint(hdfs_path: str) -> tuple:
    _unsupported()


def find_latest_valid_checkpoint(hdfs_dir: str) -> Optional[int]:
    _unsupported()
