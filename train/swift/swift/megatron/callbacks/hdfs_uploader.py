# Copyright (c) ModelScope Contributors. All rights reserved.
# Modified for Axion publication (2026-11-02): clarify the unavailable HDFS transport.
"""Callback that mirrors each saved checkpoint to an HDFS workspace.

Enabled when ``args.hdfs_dir`` is set (the script layer auto-derives it). Intermediate
checkpoints upload asynchronously on rank0 so training collectives are never blocked; at
train end the latest checkpoint is uploaded blockingly so the process cannot exit before
the durable copy + ``_SUCCESS`` marker exist. Combined with the shell layer's
resume-from-HDFS, a job survives node loss / cross-datacenter rescheduling.
"""
import os
import time

from swift.utils import get_logger, is_master
from ..utils import maybe_finalize_async_save
from ..utils.hdfs_checkpoint import (hdfs_client_available, hdfs_exists, hdfs_upload, remap_hdfs_path,
                                     wait_for_pending_uploads)
from .base import MegatronCallback

logger = get_logger()


class HdfsUploaderCallback(MegatronCallback):

    def __init__(self, trainer):
        super().__init__(trainer)
        self.hdfs_dir = getattr(self.args, 'hdfs_dir', None)
        self.cleanup_local = bool(getattr(self.args, 'cleanup_local_checkpoint_dir', False))
        # Max wait for in-flight async (intermediate) uploads to finish at train end.
        self._pending_upload_timeout = float(os.environ.get('HDFS_PENDING_UPLOAD_TIMEOUT', '1800'))
        self._enabled = bool(self.hdfs_dir) and is_master()
        if self.hdfs_dir and not hdfs_client_available():
            logger.warning('HDFS checkpoint transport is not included in this distribution; use local checkpoints.')
            self._enabled = False
        if self._enabled:
            logger.info(f'[HdfsUploaderCallback] checkpoints will be mirrored to: {self.hdfs_dir}')

    def _dst_for(self, output_dir: str) -> str:
        return os.path.join(remap_hdfs_path(self.hdfs_dir).rstrip('/'), os.path.basename(output_dir.rstrip('/')))

    def on_save(self, output_dir):
        if not self._enabled or not output_dir or not os.path.isdir(output_dir):
            return
        # The final checkpoint is uploaded (blocking) in on_train_end; skip the async
        # upload here so two `-put` to the same dst don't race.
        train_iters = getattr(self.args, 'train_iters', None)
        if train_iters and getattr(self.state, 'iteration', 0) >= train_iters:
            logger.info(f'Final checkpoint {output_dir}: deferring HDFS upload to on_train_end (blocking).')
            return
        # If megatron async save is on, the dir may still be writing; finalize first.
        if getattr(self.args, 'async_save', False):
            try:
                maybe_finalize_async_save(self.args, blocking=True)
            except Exception:
                logger.exception('maybe_finalize_async_save failed before HDFS upload')
        on_success = None
        if self.cleanup_local:
            def on_success():
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)
                logger.info(f'Removed local checkpoint after HDFS upload: {output_dir}')
        try:
            hdfs_upload(output_dir, self._dst_for(output_dir), blocking=False, on_success=on_success)
        except Exception:
            logger.exception(f'HDFS async upload failed: {output_dir}')

    def on_train_end(self):
        if not self._enabled:
            return
        # Join outstanding async uploads (intermediate checkpoints) so their `_SUCCESS`
        # markers land before the process exits; otherwise a daemon thread that finished
        # the data `-put` but not the marker leaves that checkpoint looking invalid.
        wait_for_pending_uploads(timeout=self._pending_upload_timeout)
        output_dir = getattr(self.state, 'last_model_checkpoint', None)
        if not output_dir:
            return
        dst = self._dst_for(output_dir)
        # Prefer a blocking re-upload (idempotent `-put -f`) so the final checkpoint is
        # durably on HDFS with _SUCCESS before the process exits. If the local dir was
        # already cleaned up, fall back to polling for the async _SUCCESS.
        if os.path.isdir(output_dir):
            try:
                hdfs_upload(output_dir, dst, blocking=True)
                return
            except Exception:
                logger.exception(f'Final HDFS upload failed: {output_dir}')
        success_marker = os.path.join(dst, '_SUCCESS')
        for _ in range(120):
            if hdfs_exists(success_marker):
                logger.info(f'Verified final checkpoint on HDFS: {dst}')
                return
            time.sleep(5)
        logger.error(f'Final checkpoint _SUCCESS not found on HDFS: {success_marker}')
