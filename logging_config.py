"""
Centralized logging configuration for document analysis system.

Log files:
- api.log: FastAPI request/response logs
- worker.log: RQ worker job processing logs
- ingestion.log: PDF ingestion pipeline logs
- chunking.log: Chunking operations logs
- embedding.log: Embedding generation logs
- error.log: All ERROR level logs (aggregated)
"""

import os
import logging
import logging.handlers
from pathlib import Path
from contextvars import ContextVar
from typing import Optional

# Log directory
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Context variable for batch_id (thread-safe)
current_batch_id: ContextVar[Optional[str]] = ContextVar('batch_id', default=None)

# Log format with batch_id
DETAILED_FORMAT = "%(asctime)s | %(levelname)-8s | %(batch_id)-36s | %(name)-20s | %(funcName)-20s | %(message)s"
SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(batch_id)-36s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class BatchIdFilter(logging.Filter):
    """Filter that adds batch_id to log records."""

    def filter(self, record):
        batch_id = current_batch_id.get()
        record.batch_id = batch_id if batch_id else "-"
        return True


def set_batch_id(batch_id: str):
    """Set the current batch_id for logging context."""
    current_batch_id.set(batch_id)


def clear_batch_id():
    """Clear the current batch_id."""
    current_batch_id.set(None)


def get_file_handler(filename: str, level=logging.DEBUG, max_bytes=10*1024*1024, backup_count=5):
    """Create a rotating file handler with batch_id filter."""
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DETAILED_FORMAT, DATE_FORMAT))
    handler.addFilter(BatchIdFilter())
    return handler


def get_console_handler(level=logging.INFO):
    """Create a console handler with batch_id filter."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT))
    handler.addFilter(BatchIdFilter())
    return handler


def setup_logger(name: str, log_file: str, level=logging.DEBUG, console=True) -> logging.Logger:
    """
    Set up a logger with file and optional console handlers.

    Args:
        name: Logger name
        log_file: Log filename (e.g., 'api.log')
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler for this logger
    logger.addHandler(get_file_handler(log_file, level))

    # Error file handler (all errors go to error.log)
    error_handler = get_file_handler("error.log", logging.ERROR)
    logger.addHandler(error_handler)

    # Console handler
    if console:
        logger.addHandler(get_console_handler(logging.INFO))

    return logger


# Pre-configured loggers
def get_api_logger():
    """Logger for API endpoints."""
    return setup_logger("doc_analysis.api", "api.log")


def get_worker_logger():
    """Logger for RQ worker."""
    return setup_logger("doc_analysis.worker", "worker.log")


def get_ingestion_logger():
    """Logger for PDF ingestion pipeline."""
    return setup_logger("doc_analysis.ingestion", "ingestion.log")


def get_chunking_logger():
    """Logger for chunking operations."""
    return setup_logger("doc_analysis.chunking", "chunking.log")


def get_embedding_logger():
    """Logger for embedding generation."""
    return setup_logger("doc_analysis.embedding", "embedding.log")


def get_summarization_logger():
    """Logger for summarization operations."""
    return setup_logger("doc_analysis.summarization", "summarization.log")


def get_cleanup_logger():
    """Logger for cleanup service."""
    return setup_logger("doc_analysis.cleanup", "cleanup.log")


def setup_all_loggers():
    """Initialize all loggers at application startup."""
    get_api_logger()
    get_worker_logger()
    get_ingestion_logger()
    get_chunking_logger()
    get_embedding_logger()
    get_summarization_logger()
    get_cleanup_logger()

    # Also configure uvicorn loggers
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn").addHandler(get_file_handler("api.log"))
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").addHandler(get_file_handler("api.log"))


# Utility for logging function entry/exit with batch context
class LogContext:
    """Context manager for logging function execution with optional batch_id."""

    def __init__(self, logger: logging.Logger, operation: str, batch_id: str = None, **kwargs):
        self.logger = logger
        self.operation = operation
        self.batch_id = batch_id
        self.kwargs = kwargs
        self._prev_batch_id = None

    def __enter__(self):
        if self.batch_id:
            self._prev_batch_id = current_batch_id.get()
            set_batch_id(self.batch_id)
        self.logger.info(f"START: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"FAILED: {self.operation} - {exc_val}", exc_info=True)
        else:
            self.logger.info(f"END: {self.operation}")

        if self.batch_id:
            if self._prev_batch_id:
                set_batch_id(self._prev_batch_id)
            else:
                clear_batch_id()
        return False


class BatchContext:
    """Context manager for setting batch_id in logging context."""

    def __init__(self, batch_id: str):
        self.batch_id = batch_id
        self._prev_batch_id = None

    def __enter__(self):
        self._prev_batch_id = current_batch_id.get()
        set_batch_id(self.batch_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_batch_id:
            set_batch_id(self._prev_batch_id)
        else:
            clear_batch_id()
        return False
