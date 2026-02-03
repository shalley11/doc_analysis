"""
Background cleanup service for removing old temporary files.

Cleans up:
- Old images from IMAGE_STORE_DIR
- Old batch PDF files from /tmp

Runs periodically based on CLEANUP_INTERVAL_MINUTES config.
"""
import os
import time
import threading
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from doc_analysis.config import (
    IMAGE_STORE_DIR,
    CLEANUP_MAX_FILE_AGE_HOURS,
    CLEANUP_INTERVAL_MINUTES,
    CLEANUP_INCLUDE_TEMP_PDFS
)
from doc_analysis.logging_config import get_cleanup_logger

logger = get_cleanup_logger()


class CleanupService:
    """
    Background service that periodically cleans up old temporary files.

    Usage:
        service = CleanupService()
        service.start()  # Runs in background thread

        # Or run in foreground (blocking):
        service.run_forever()
    """

    def __init__(
        self,
        max_age_hours: Optional[float] = None,
        interval_minutes: Optional[float] = None,
        include_temp_pdfs: Optional[bool] = None
    ):
        """
        Initialize cleanup service.

        Args:
            max_age_hours: Max file age before deletion (default from config)
            interval_minutes: Cleanup interval (default from config)
            include_temp_pdfs: Whether to clean temp PDFs (default from config)
        """
        self.max_age_hours = max_age_hours or CLEANUP_MAX_FILE_AGE_HOURS
        self.interval_minutes = interval_minutes or CLEANUP_INTERVAL_MINUTES
        self.include_temp_pdfs = include_temp_pdfs if include_temp_pdfs is not None else CLEANUP_INCLUDE_TEMP_PDFS

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        logger.info(f"[INIT] CleanupService initialized | max_age={self.max_age_hours}h | interval={self.interval_minutes}min | include_temp_pdfs={self.include_temp_pdfs}")

    def _get_cutoff_time(self) -> datetime:
        """Get the cutoff time for file deletion."""
        return datetime.now() - timedelta(hours=self.max_age_hours)

    def _is_file_old(self, file_path: Path) -> bool:
        """Check if a file is older than the max age."""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return mtime < self._get_cutoff_time()
        except (OSError, ValueError) as e:
            logger.warning(f"[CHECK] Could not get mtime for {file_path}: {e}")
            return False

    def _get_file_age_hours(self, file_path: Path) -> float:
        """Get file age in hours."""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age = datetime.now() - mtime
            return age.total_seconds() / 3600
        except (OSError, ValueError):
            return 0.0

    def cleanup_image_store(self) -> Tuple[int, int, int]:
        """
        Clean up old files from IMAGE_STORE_DIR.

        Returns:
            Tuple of (files_deleted, bytes_freed, errors)
        """
        image_dir = Path(IMAGE_STORE_DIR)

        if not image_dir.exists():
            logger.debug(f"[IMAGE_STORE] Directory does not exist: {image_dir}")
            return 0, 0, 0

        logger.info(f"[IMAGE_STORE] Scanning {image_dir}")

        files_deleted = 0
        bytes_freed = 0
        errors = 0
        files_scanned = 0

        try:
            for file_path in image_dir.rglob("*"):
                if not file_path.is_file():
                    continue

                files_scanned += 1

                if self._is_file_old(file_path):
                    try:
                        file_size = file_path.stat().st_size
                        age_hours = self._get_file_age_hours(file_path)

                        file_path.unlink()

                        files_deleted += 1
                        bytes_freed += file_size
                        logger.debug(f"[IMAGE_STORE] Deleted: {file_path.name} | age={age_hours:.1f}h | size={file_size} bytes")

                    except OSError as e:
                        errors += 1
                        logger.error(f"[IMAGE_STORE] Failed to delete {file_path}: {e}")

        except Exception as e:
            logger.error(f"[IMAGE_STORE] Error scanning directory: {e}", exc_info=True)
            errors += 1

        logger.info(f"[IMAGE_STORE] Complete | scanned={files_scanned} | deleted={files_deleted} | freed={bytes_freed/1024/1024:.2f}MB | errors={errors}")
        return files_deleted, bytes_freed, errors

    def cleanup_temp_pdfs(self) -> Tuple[int, int, int]:
        """
        Clean up old batch PDF files from /tmp.

        Matches pattern: /tmp/{batch_id}_pdf_*.pdf

        Returns:
            Tuple of (files_deleted, bytes_freed, errors)
        """
        if not self.include_temp_pdfs:
            logger.debug(f"[TEMP_PDFS] Skipped (disabled in config)")
            return 0, 0, 0

        tmp_dir = Path("/tmp")

        logger.info(f"[TEMP_PDFS] Scanning {tmp_dir} for batch PDFs")

        files_deleted = 0
        bytes_freed = 0
        errors = 0
        files_scanned = 0

        try:
            # Match batch PDF pattern: UUID_pdf_N.pdf
            for file_path in tmp_dir.glob("*_pdf_*.pdf"):
                if not file_path.is_file():
                    continue

                files_scanned += 1

                if self._is_file_old(file_path):
                    try:
                        file_size = file_path.stat().st_size
                        age_hours = self._get_file_age_hours(file_path)

                        file_path.unlink()

                        files_deleted += 1
                        bytes_freed += file_size
                        logger.debug(f"[TEMP_PDFS] Deleted: {file_path.name} | age={age_hours:.1f}h | size={file_size} bytes")

                    except OSError as e:
                        errors += 1
                        logger.error(f"[TEMP_PDFS] Failed to delete {file_path}: {e}")

        except Exception as e:
            logger.error(f"[TEMP_PDFS] Error scanning directory: {e}", exc_info=True)
            errors += 1

        logger.info(f"[TEMP_PDFS] Complete | scanned={files_scanned} | deleted={files_deleted} | freed={bytes_freed/1024/1024:.2f}MB | errors={errors}")
        return files_deleted, bytes_freed, errors

    def cleanup_empty_dirs(self) -> int:
        """
        Remove empty subdirectories from IMAGE_STORE_DIR.

        Returns:
            Number of directories removed
        """
        image_dir = Path(IMAGE_STORE_DIR)

        if not image_dir.exists():
            return 0

        dirs_removed = 0

        # Walk bottom-up to remove nested empty dirs
        for dir_path in sorted(image_dir.rglob("*"), reverse=True):
            if dir_path.is_dir():
                try:
                    # rmdir only works on empty directories
                    dir_path.rmdir()
                    dirs_removed += 1
                    logger.debug(f"[EMPTY_DIRS] Removed empty directory: {dir_path}")
                except OSError:
                    # Directory not empty, skip
                    pass

        if dirs_removed > 0:
            logger.info(f"[EMPTY_DIRS] Removed {dirs_removed} empty directories")

        return dirs_removed

    def run_cleanup(self) -> dict:
        """
        Run a single cleanup cycle.

        Returns:
            Summary dictionary with cleanup results
        """
        start_time = time.time()
        logger.info(f"[CLEANUP] START | max_age={self.max_age_hours}h")

        # Cleanup image store
        img_deleted, img_bytes, img_errors = self.cleanup_image_store()

        # Cleanup temp PDFs
        pdf_deleted, pdf_bytes, pdf_errors = self.cleanup_temp_pdfs()

        # Cleanup empty directories
        dirs_removed = self.cleanup_empty_dirs()

        elapsed = time.time() - start_time

        total_deleted = img_deleted + pdf_deleted
        total_bytes = img_bytes + pdf_bytes
        total_errors = img_errors + pdf_errors

        logger.info(
            f"[CLEANUP] END | elapsed={elapsed:.2f}s | "
            f"files_deleted={total_deleted} | freed={total_bytes/1024/1024:.2f}MB | "
            f"dirs_removed={dirs_removed} | errors={total_errors}"
        )

        return {
            "elapsed_seconds": elapsed,
            "image_store": {
                "files_deleted": img_deleted,
                "bytes_freed": img_bytes,
                "errors": img_errors
            },
            "temp_pdfs": {
                "files_deleted": pdf_deleted,
                "bytes_freed": pdf_bytes,
                "errors": pdf_errors
            },
            "empty_dirs_removed": dirs_removed,
            "total_files_deleted": total_deleted,
            "total_bytes_freed": total_bytes,
            "total_errors": total_errors
        }

    def _run_loop(self):
        """Internal loop for background thread."""
        logger.info(f"[SERVICE] Background cleanup loop started | interval={self.interval_minutes}min")

        while not self._stop_event.is_set():
            try:
                self.run_cleanup()
            except Exception as e:
                logger.error(f"[SERVICE] Cleanup cycle failed: {e}", exc_info=True)

            # Wait for next cycle or stop signal
            interval_seconds = self.interval_minutes * 60
            logger.debug(f"[SERVICE] Next cleanup in {self.interval_minutes} minutes")

            if self._stop_event.wait(timeout=interval_seconds):
                break

        logger.info(f"[SERVICE] Background cleanup loop stopped")

    def start(self):
        """Start the cleanup service in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"[SERVICE] Already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"[SERVICE] Started background thread")

    def stop(self, timeout: float = 10.0):
        """
        Stop the cleanup service.

        Args:
            timeout: Max seconds to wait for thread to stop
        """
        if self._thread is None or not self._thread.is_alive():
            logger.debug(f"[SERVICE] Not running")
            return

        logger.info(f"[SERVICE] Stopping...")
        self._stop_event.set()
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.warning(f"[SERVICE] Thread did not stop within {timeout}s")
        else:
            logger.info(f"[SERVICE] Stopped")

    def run_forever(self):
        """
        Run the cleanup service in foreground (blocking).

        Handles SIGINT and SIGTERM for graceful shutdown.
        """
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"[SERVICE] Received {sig_name}, shutting down...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"[SERVICE] Running in foreground (Ctrl+C to stop)")

        # Run initial cleanup immediately
        self.run_cleanup()

        # Then run the loop
        self._run_loop()


def main():
    """Entry point for running cleanup service as a standalone process."""
    logger.info("=" * 60)
    logger.info("Cleanup Service Starting")
    logger.info(f"  IMAGE_STORE_DIR: {IMAGE_STORE_DIR}")
    logger.info(f"  Max file age: {CLEANUP_MAX_FILE_AGE_HOURS} hours")
    logger.info(f"  Cleanup interval: {CLEANUP_INTERVAL_MINUTES} minutes")
    logger.info(f"  Include temp PDFs: {CLEANUP_INCLUDE_TEMP_PDFS}")
    logger.info("=" * 60)

    service = CleanupService()
    service.run_forever()


if __name__ == "__main__":
    main()
