# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified by the Axion authors: retain import compatibility without optional
# private dataset transforms. Standard message preprocessing stays in core.py.
"""Compatibility adapters for dataset formats outside the Axion recipe."""
from .core import MessagesPreprocessor, RowPreprocessor

_joint_lm_training: bool = False


def configure_joint_lm_training(enabled: bool) -> None:
    if enabled:
        raise NotImplementedError('Joint LM/regression preprocessing is not included in the Axion distribution.')


class ThemisVideoMetaPreprocessor(RowPreprocessor):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('This video metadata dataset format is not included in the Axion distribution.')


class ThemisMessagesPreprocessor(MessagesPreprocessor):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Scored-message preprocessing is not included in the Axion distribution.')
