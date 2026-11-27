"""
Structured Logging for News AI Platform
Replaces print() statements with proper logging levels, formatting, and file output.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config import LoggingConfig, get_config


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Format: timestamp | LEVEL     | logger | message
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = f"{color}{record.levelname:<8}{reset}"
        logger_name = record.name
        message = record.getMessage()

        return f"{timestamp} | {level} | {logger_name} | {message}"


# Module-level flag to track initialization
_logging_initialized = False
_logging_config_hash = None


def _get_config_hash(level: str, log_file: Optional[str], json_format: bool) -> str:
    """Generate a hash of the logging configuration to detect changes."""
    import hashlib
    config_str = f"{level}:{log_file or '':}{json_format}"
    return hashlib.md5(config_str.encode()).hexdigest()[:16]


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: Optional[bool] = None,
    config_override: Optional[LoggingConfig] = None,
    force: bool = False
) -> logging.Logger:
    """
    Setup application-wide logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None = stdout only)
        json_format: Use JSON formatting (for production/log aggregation)
        config_override: Optional LoggingConfig to use instead of global config
        force: Force reinitialization even if config unchanged

    Returns:
        Root logger instance
    """
    global _logging_initialized, _logging_config_hash

    cfg = config_override or get_config().logging

    # Resolve parameters with priority: explicit arg > config > default
    log_level = (level or cfg.level).upper()
    log_file_path = log_file or cfg.log_file
    use_json = json_format if json_format is not None else cfg.json_format

    # Check if already initialized with same config
    config_hash = _get_config_hash(log_level, log_file_path, use_json)
    if _logging_initialized and _logging_config_hash == config_hash and not force:
        return logging.getLogger()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colored formatting
    # Ensure UTF-8 encoding and immediate (unbuffered) output on every platform.
    # When stdout is a terminal (local dev) Python usually line-buffers; when it
    # is a pipe or captured stream (Render, CI) Python uses full block buffering
    # (~8 KiB) and INFO log lines would not appear until the buffer fills or the
    # process exits. write_through + line_buffering makes every log line show up
    # in real time on both.
    try:
        sys.stdout.reconfigure(encoding="utf-8", write_through=True, line_buffering=True)
    except (AttributeError, OSError, ValueError):
        import io

        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding="utf-8",
                errors="replace",
                write_through=True,
            )
        except (AttributeError, OSError, ValueError):
            # stdout cannot be wrapped (e.g. replaced by a test double); leave as-is
            pass

    console_handler = logging.StreamHandler(sys.stdout)
    if use_json:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ColoredConsoleFormatter(
            fmt=cfg.format,
            datefmt=cfg.date_format
        ))
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler: 10MB max, keep 5 backups
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(JSONFormatter() if use_json else logging.Formatter(
            fmt=cfg.format,
            datefmt=cfg.date_format
        ))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("feedparser").setLevel(logging.WARNING)

    # Mark as initialized
    _logging_initialized = True
    _logging_config_hash = config_hash

    # Log startup message
    logger = logging.getLogger("news_ai")
    logger.info(
        "Logging initialized",
        extra={"extra_fields": {
            "level": log_level,
            "json_format": use_json,
            "log_file": log_file_path
        }}
    )

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(f"news_ai.{name}")


# Convenience functions for common logging patterns
def log_pipeline_start(logger: logging.Logger, api_key_present: bool, max_per_source: int):
    logger.info(
        "Pipeline started",
        extra={"extra_fields": {
            "event": "pipeline_start",
            "api_key_present": api_key_present,
            "max_per_source": max_per_source
        }}
    )


def log_pipeline_complete(logger: logging.Logger, articles_count: int, duration_ms: float):
    logger.info(
        "Pipeline completed",
        extra={"extra_fields": {
            "event": "pipeline_complete",
            "articles_count": articles_count,
            "duration_ms": duration_ms
        }}
    )


def log_api_request(logger: logging.Logger, method: str, path: str, status_code: int, duration_ms: float):
    logger.info(
        f"{method} {path} -> {status_code}",
        extra={"extra_fields": {
            "event": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms
        }}
    )


def log_groq_call(logger: logging.Logger, model: str, success: bool, duration_ms: float, fallback: bool = False):
    logger.info(
        f"Groq API call: {model} {'success' if success else 'failed'}{' (fallback)' if fallback else ''}",
        extra={"extra_fields": {
            "event": "groq_call",
            "model": model,
            "success": success,
            "duration_ms": duration_ms,
            "fallback": fallback
        }}
    )


def log_rss_fetch(logger: logging.Logger, source: str, count: int, success: bool, error: str = None):
    if success:
        logger.info(
            f"RSS fetch: {source} -> {count} articles",
            extra={"extra_fields": {
                "event": "rss_fetch",
                "source": source,
                "count": count,
                "success": True
            }}
        )
    else:
        logger.warning(
            f"RSS fetch failed: {source} - {error}",
            extra={"extra_fields": {
                "event": "rss_fetch",
                "source": source,
                "success": False,
                "error": error
            }}
        )