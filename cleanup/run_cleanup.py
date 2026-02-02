#!/usr/bin/env python3
"""
Standalone runner for the cleanup service.

Usage:
    python -m doc_analysis.cleanup.run_cleanup

    # Or with custom settings:
    python -m doc_analysis.cleanup.run_cleanup --max-age 12 --interval 30

    # Run once (no loop):
    python -m doc_analysis.cleanup.run_cleanup --once
"""
import argparse
import sys

from doc_analysis.logging_config import setup_all_loggers
setup_all_loggers()

from doc_analysis.cleanup.cleanup_service import CleanupService, logger
from doc_analysis.config import (
    IMAGE_STORE_DIR,
    CLEANUP_MAX_FILE_AGE_HOURS,
    CLEANUP_INTERVAL_MINUTES,
    CLEANUP_INCLUDE_TEMP_PDFS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cleanup service for removing old temporary files"
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=CLEANUP_MAX_FILE_AGE_HOURS,
        help=f"Max file age in hours before deletion (default: {CLEANUP_MAX_FILE_AGE_HOURS})"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=CLEANUP_INTERVAL_MINUTES,
        help=f"Cleanup interval in minutes (default: {CLEANUP_INTERVAL_MINUTES})"
    )
    parser.add_argument(
        "--no-temp-pdfs",
        action="store_true",
        help="Skip cleaning temp PDF files"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run cleanup once and exit (no loop)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Cleanup Service")
    logger.info("=" * 60)
    logger.info(f"  IMAGE_STORE_DIR: {IMAGE_STORE_DIR}")
    logger.info(f"  Max file age: {args.max_age} hours")
    logger.info(f"  Cleanup interval: {args.interval} minutes")
    logger.info(f"  Include temp PDFs: {not args.no_temp_pdfs}")
    logger.info(f"  Mode: {'single run' if args.once else 'continuous'}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY-RUN] Dry run mode - no files will be deleted")
        # TODO: Implement dry-run mode
        logger.warning("[DRY-RUN] Dry run not yet implemented, exiting")
        sys.exit(0)

    service = CleanupService(
        max_age_hours=args.max_age,
        interval_minutes=args.interval,
        include_temp_pdfs=not args.no_temp_pdfs
    )

    if args.once:
        result = service.run_cleanup()
        logger.info(f"Cleanup complete: {result['total_files_deleted']} files deleted, "
                   f"{result['total_bytes_freed']/1024/1024:.2f}MB freed")
    else:
        service.run_forever()


if __name__ == "__main__":
    main()
