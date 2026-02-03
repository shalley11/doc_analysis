"""
Centralized logging configuration for LLM_Utilities.

Features:
- Structured logging for LLM requests/responses
- Request ID tracking for tracing
- Token usage and latency metrics
- Rotating file handlers
- Separate log files for different concerns

Log files:
- llm_requests.log: All LLM API calls (requests & responses)
- llm_errors.log: Error-level logs only
- llm_metrics.log: Performance metrics (latency, tokens, etc.)
- llm_debug.log: Detailed debug information
"""

import os
import json
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from contextvars import ContextVar
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from functools import wraps
import time
import uuid


# =========================
# Configuration
# =========================

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Context variables for request tracking (thread-safe)
current_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
current_session_id: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
current_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

# Log formats (includes user_id)
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(request_id)-36s | %(user_id)-20s | "
    "%(name)-25s | %(funcName)-20s | %(message)s"
)
SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(request_id)-36s | %(user_id)-20s | %(message)s"
JSON_FORMAT = "%(message)s"  # For structured JSON logs
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# =========================
# Data Classes for Structured Logging
# =========================

@dataclass
class LLMRequestLog:
    """Structured log entry for LLM requests."""
    request_id: str
    timestamp: str
    model: str
    backend: str  # ollama or vllm
    task: str
    prompt_length: int
    prompt_preview: str  # First N chars
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_limit: Optional[int] = None
    estimated_tokens: Optional[int] = None
    context_usage_percent: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class LLMResponseLog:
    """Structured log entry for LLM responses."""
    request_id: str
    timestamp: str
    model: str
    backend: str
    status: str  # success, error, timeout
    response_length: int
    response_preview: str  # First N chars
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class LLMMetrics:
    """Metrics for monitoring and analysis."""
    request_id: str
    timestamp: str
    model: str
    backend: str
    task: str
    latency_ms: float
    prompt_chars: int
    response_chars: int
    status: str
    user_id: Optional[str] = None
    context_limit: Optional[int] = None
    estimated_tokens: Optional[int] = None
    context_usage_percent: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ContextUsageLog:
    """Structured log entry for context usage tracking."""
    request_id: str
    timestamp: str
    model: str
    context_limit: int
    estimated_tokens: int
    usage_percent: float
    warning_level: str  # "normal", "warning", "error", "critical"
    user_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# =========================
# Filters
# =========================

class RequestIdFilter(logging.Filter):
    """Filter that adds request_id and user_id to log records."""

    def filter(self, record):
        request_id = current_request_id.get()
        record.request_id = request_id if request_id else "-"

        user_id = current_user_id.get()
        record.user_id = user_id if user_id else "-"
        return True


class SessionIdFilter(logging.Filter):
    """Filter that adds session_id to log records."""

    def filter(self, record):
        session_id = current_session_id.get()
        record.session_id = session_id if session_id else "-"
        return True


class UserIdFilter(logging.Filter):
    """Filter that adds user_id to log records."""

    def filter(self, record):
        user_id = current_user_id.get()
        record.user_id = user_id if user_id else "-"
        return True


# =========================
# Context Management
# =========================

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def set_request_id(request_id: str):
    """Set the current request_id for logging context."""
    current_request_id.set(request_id)


def get_request_id() -> Optional[str]:
    """Get the current request_id."""
    return current_request_id.get()


def clear_request_id():
    """Clear the current request_id."""
    current_request_id.set(None)


def set_session_id(session_id: str):
    """Set the current session_id for logging context."""
    current_session_id.set(session_id)


def clear_session_id():
    """Clear the current session_id."""
    current_session_id.set(None)


def set_user_id(user_id: str):
    """Set the current user_id for logging context."""
    current_user_id.set(user_id)


def get_user_id() -> Optional[str]:
    """Get the current user_id."""
    return current_user_id.get()


def clear_user_id():
    """Clear the current user_id."""
    current_user_id.set(None)


# =========================
# Handler Factory
# =========================

def get_file_handler(
    filename: str,
    level=logging.DEBUG,
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5,
    format_str=DETAILED_FORMAT
) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler."""
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_str, DATE_FORMAT))
    handler.addFilter(RequestIdFilter())
    return handler


def get_json_file_handler(
    filename: str,
    level=logging.INFO,
    max_bytes=10*1024*1024,
    backup_count=5
) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler for JSON logs."""
    handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(JSON_FORMAT))
    return handler


def get_console_handler(level=logging.INFO) -> logging.StreamHandler:
    """Create a console handler."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT))
    handler.addFilter(RequestIdFilter())
    return handler


# =========================
# Logger Setup
# =========================

def setup_logger(
    name: str,
    log_file: str,
    level=logging.DEBUG,
    console=True
) -> logging.Logger:
    """
    Set up a logger with file and optional console handlers.

    Args:
        name: Logger name
        log_file: Log filename
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

    # Main file handler
    logger.addHandler(get_file_handler(log_file, level))

    # Error file handler (all errors aggregated)
    logger.addHandler(get_file_handler("llm_errors.log", logging.ERROR))

    # Console handler
    if console:
        logger.addHandler(get_console_handler(logging.INFO))

    return logger


# Pre-configured loggers
_llm_logger: Optional[logging.Logger] = None
_metrics_logger: Optional[logging.Logger] = None


def get_llm_logger() -> logging.Logger:
    """Get the main LLM logger."""
    global _llm_logger
    if _llm_logger is None:
        _llm_logger = setup_logger("llm_utilities", "llm_requests.log")
    return _llm_logger


def get_metrics_logger() -> logging.Logger:
    """Get the metrics logger (JSON format)."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = logging.getLogger("llm_utilities.metrics")
        _metrics_logger.setLevel(logging.INFO)
        if not _metrics_logger.handlers:
            _metrics_logger.addHandler(
                get_json_file_handler("llm_metrics.log")
            )
    return _metrics_logger


# =========================
# Logging Helper Functions
# =========================

def log_llm_request(
    model: str,
    backend: str,
    task: str,
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    user_id: Optional[str] = None,
    **extra
) -> str:
    """
    Log an LLM request and return the request_id.

    Args:
        model: Model name
        backend: Backend (ollama/vllm)
        task: Task type (summary, translate, etc.)
        prompt: The prompt text
        temperature: Temperature setting
        max_tokens: Max tokens setting
        user_id: Optional user identifier
        **extra: Additional fields to log

    Returns:
        Generated request_id for tracking
    """
    request_id = generate_request_id()
    set_request_id(request_id)

    # Set user_id in context if provided
    if user_id:
        set_user_id(user_id)

    logger = get_llm_logger()

    # Get user_id from context (may have been set earlier)
    current_user = user_id or current_user_id.get()

    # Create structured log entry
    log_entry = LLMRequestLog(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        model=model,
        backend=backend,
        task=task,
        prompt_length=len(prompt),
        prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        session_id=current_session_id.get(),
        user_id=current_user,
        extra=extra
    )

    user_info = f" | user_id={current_user}" if current_user else ""
    logger.info(f"LLM_REQUEST | model={model} | backend={backend} | task={task} | prompt_len={len(prompt)}{user_info}")
    logger.debug(f"LLM_REQUEST_DETAIL | {log_entry.to_json()}")

    return request_id


def log_llm_response(
    request_id: str,
    model: str,
    backend: str,
    response: str,
    latency_ms: float,
    status: str = "success",
    error_message: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    user_id: Optional[str] = None,
    **extra
):
    """
    Log an LLM response.

    Args:
        request_id: The request ID from log_llm_request
        model: Model name
        backend: Backend (ollama/vllm)
        response: The response text
        latency_ms: Response latency in milliseconds
        status: Status (success/error/timeout)
        error_message: Error message if failed
        prompt_tokens: Token count for prompt
        completion_tokens: Token count for completion
        user_id: Optional user identifier
        **extra: Additional fields to log
    """
    logger = get_llm_logger()

    total_tokens = None
    if prompt_tokens and completion_tokens:
        total_tokens = prompt_tokens + completion_tokens

    # Get user_id from context or parameter
    current_user = user_id or current_user_id.get()

    log_entry = LLMResponseLog(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        model=model,
        backend=backend,
        status=status,
        response_length=len(response) if response else 0,
        response_preview=(response[:200] + "...") if response and len(response) > 200 else (response or ""),
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        error_message=error_message,
        user_id=current_user,
        extra=extra
    )

    user_info = f" | user_id={current_user}" if current_user else ""

    if status == "success":
        logger.info(
            f"LLM_RESPONSE | status={status} | latency={latency_ms:.2f}ms | "
            f"response_len={len(response) if response else 0}{user_info}"
        )
    else:
        logger.error(
            f"LLM_RESPONSE | status={status} | latency={latency_ms:.2f}ms | "
            f"error={error_message}{user_info}"
        )

    logger.debug(f"LLM_RESPONSE_DETAIL | {log_entry.to_json()}")


def log_metrics(
    request_id: str,
    model: str,
    backend: str,
    task: str,
    latency_ms: float,
    prompt_chars: int,
    response_chars: int,
    status: str,
    user_id: Optional[str] = None,
    context_limit: Optional[int] = None,
    estimated_tokens: Optional[int] = None,
    context_usage_percent: Optional[float] = None
):
    """Log metrics for monitoring and analysis."""
    metrics_logger = get_metrics_logger()

    # Get user_id from context or parameter
    current_user = user_id or current_user_id.get()

    metrics = LLMMetrics(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        model=model,
        backend=backend,
        task=task,
        latency_ms=latency_ms,
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        status=status,
        user_id=current_user,
        context_limit=context_limit,
        estimated_tokens=estimated_tokens,
        context_usage_percent=context_usage_percent
    )

    metrics_logger.info(metrics.to_json())


def log_context_usage(
    request_id: str,
    model: str,
    prompt: str,
    context_limit: int,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Log context usage and warn if exceeding threshold.

    Args:
        request_id: The request ID for tracking
        model: Model name
        prompt: The prompt text
        context_limit: Model's context window limit
        user_id: Optional user identifier

    Returns:
        Dict with context usage stats
    """
    from config import estimate_tokens, CONTEXT_WARNING_THRESHOLD, CONTEXT_ERROR_THRESHOLD

    logger = get_llm_logger()
    current_user = user_id or current_user_id.get()

    # Estimate tokens
    estimated_tokens = estimate_tokens(prompt)
    usage_percent = (estimated_tokens / context_limit) * 100 if context_limit > 0 else 0

    # Determine warning level
    if usage_percent >= 100:
        warning_level = "critical"
    elif usage_percent >= CONTEXT_ERROR_THRESHOLD:
        warning_level = "error"
    elif usage_percent >= CONTEXT_WARNING_THRESHOLD:
        warning_level = "warning"
    else:
        warning_level = "normal"

    # Create log entry
    log_entry = ContextUsageLog(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        model=model,
        context_limit=context_limit,
        estimated_tokens=estimated_tokens,
        usage_percent=round(usage_percent, 2),
        warning_level=warning_level,
        user_id=current_user
    )

    user_info = f" | user_id={current_user}" if current_user else ""

    # Log based on warning level
    if warning_level == "critical":
        logger.critical(
            f"CONTEXT_EXCEEDED | model={model} | usage={usage_percent:.1f}% | "
            f"tokens={estimated_tokens}/{context_limit}{user_info}"
        )
    elif warning_level == "error":
        logger.error(
            f"CONTEXT_NEAR_LIMIT | model={model} | usage={usage_percent:.1f}% | "
            f"tokens={estimated_tokens}/{context_limit}{user_info}"
        )
    elif warning_level == "warning":
        logger.warning(
            f"CONTEXT_USAGE_HIGH | model={model} | usage={usage_percent:.1f}% | "
            f"tokens={estimated_tokens}/{context_limit}{user_info}"
        )
    else:
        logger.debug(
            f"CONTEXT_USAGE | model={model} | usage={usage_percent:.1f}% | "
            f"tokens={estimated_tokens}/{context_limit}{user_info}"
        )

    logger.debug(f"CONTEXT_USAGE_DETAIL | {log_entry.to_json()}")

    return {
        "context_limit": context_limit,
        "estimated_tokens": estimated_tokens,
        "usage_percent": round(usage_percent, 2),
        "warning_level": warning_level
    }


# =========================
# Decorators
# =========================

def log_llm_call(task: str = "unknown"):
    """
    Decorator to automatically log LLM calls.

    Usage:
        @log_llm_call(task="summarize")
        def generate_summary(prompt, model):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract common parameters
            prompt = kwargs.get('prompt') or (args[0] if args else "")
            model = kwargs.get('model') or (args[1] if len(args) > 1 else "unknown")
            backend = kwargs.get('backend', 'ollama')

            # Log request
            request_id = log_llm_request(
                model=model,
                backend=backend,
                task=task,
                prompt=prompt
            )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Log response
                log_llm_response(
                    request_id=request_id,
                    model=model,
                    backend=backend,
                    response=result if isinstance(result, str) else str(result),
                    latency_ms=latency_ms,
                    status="success"
                )

                # Log metrics
                log_metrics(
                    request_id=request_id,
                    model=model,
                    backend=backend,
                    task=task,
                    latency_ms=latency_ms,
                    prompt_chars=len(prompt),
                    response_chars=len(result) if isinstance(result, str) else 0,
                    status="success"
                )

                return result

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000

                # Log error
                log_llm_response(
                    request_id=request_id,
                    model=model,
                    backend=backend,
                    response="",
                    latency_ms=latency_ms,
                    status="error",
                    error_message=str(e)
                )

                # Log metrics
                log_metrics(
                    request_id=request_id,
                    model=model,
                    backend=backend,
                    task=task,
                    latency_ms=latency_ms,
                    prompt_chars=len(prompt),
                    response_chars=0,
                    status="error"
                )

                raise
            finally:
                clear_request_id()

        return wrapper
    return decorator


# =========================
# Context Managers
# =========================

class RequestContext:
    """Context manager for request-scoped logging with optional user_id."""

    def __init__(self, request_id: Optional[str] = None, user_id: Optional[str] = None):
        self.request_id = request_id or generate_request_id()
        self.user_id = user_id
        self._prev_request_id = None
        self._prev_user_id = None

    def __enter__(self):
        self._prev_request_id = current_request_id.get()
        set_request_id(self.request_id)

        if self.user_id:
            self._prev_user_id = current_user_id.get()
            set_user_id(self.user_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_request_id:
            set_request_id(self._prev_request_id)
        else:
            clear_request_id()

        if self.user_id:
            if self._prev_user_id:
                set_user_id(self._prev_user_id)
            else:
                clear_user_id()

        return False


class UserContext:
    """Context manager for user-scoped logging."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._prev_user_id = None

    def __enter__(self):
        self._prev_user_id = current_user_id.get()
        set_user_id(self.user_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_user_id:
            set_user_id(self._prev_user_id)
        else:
            clear_user_id()
        return False


class SessionContext:
    """Context manager for session-scoped logging (for chat sessions)."""

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id
        self._prev_session_id = None
        self._prev_user_id = None

    def __enter__(self):
        self._prev_session_id = current_session_id.get()
        set_session_id(self.session_id)

        if self.user_id:
            self._prev_user_id = current_user_id.get()
            set_user_id(self.user_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev_session_id:
            set_session_id(self._prev_session_id)
        else:
            clear_session_id()

        if self.user_id:
            if self._prev_user_id:
                set_user_id(self._prev_user_id)
            else:
                clear_user_id()

        return False


# =========================
# Initialization
# =========================

def setup_llm_logging():
    """Initialize all LLM loggers at application startup."""
    get_llm_logger()
    get_metrics_logger()

    logger = get_llm_logger()
    logger.info("=" * 60)
    logger.info("LLM_UTILITIES LOGGING INITIALIZED")
    logger.info(f"Log directory: {LOG_DIR}")
    logger.info("=" * 60)
