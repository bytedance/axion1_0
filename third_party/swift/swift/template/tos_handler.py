# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for Axion publication (2026-11-02): remove the private TOS transport.
"""Compatibility adapter for an unavailable private video storage transport.

Local and HTTP image/video loading remains in the unchanged vision utilities.
Importing this adapter has no credential, proxy, network, or storage effects.
"""
from typing import List

from PIL import Image


def load_video_tos_zip(filepath: str, max_frames: int = 32) -> List[Image.Image]:
    """Reject private TOS input; stage the video locally or use an HTTP URL."""
    raise NotImplementedError(
        'tos:// video storage is not supported by this distribution. '
        'Stage the video locally or provide a supported HTTP URL.')
