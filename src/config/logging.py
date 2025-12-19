"""Structured logging configuration.

Provides:
- JSON logging for production (structured, machine-parseable)
- Human-readable logging for development
- Request correlation ID propagation
- Consistent log format across the application
"""
import logging
import logging.config
import sys
from datetime import datetime, timezone
from typing import Any

import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Produces logs in the format:
    {
        "timestamp": "ISO8601",
        "level": "INFO",
        "logger": "src.module",
        "message": "Log message",
        "request_id": "uuid",
        "user_id": 123,
        ...extra fields
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        extra_fields = [
            "request_id",
            "user_id",
            "path",
            "method",
            "status_code",
            "duration_ms",
            "error_code",
            "model",
            "dialog_id",
            "message_id",
            "prompt_tokens",
            "completion_tokens",
        ]
        for field in extra_fields:
            if hasattr(record, field) and getattr(record, field) is not None:
                log_data[field] = getattr(record, field)

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for development.

    Format: [LEVEL] timestamp - logger - message {extra}
    """

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        # Build timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Color handling
        level = record.levelname
        if self.use_colors and sys.stderr.isatty():
            color = self.LEVEL_COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        # Build extra fields string
        extra_parts = []
        extra_fields = ["request_id", "user_id", "path", "method", "status_code", "duration_ms"]
        for field in extra_fields:
            if hasattr(record, field) and getattr(record, field) is not None:
                extra_parts.append(f"{field}={getattr(record, field)}")

        extra_str = f" | {' '.join(extra_parts)}" if extra_parts else ""

        # Build message
        message = f"[{level_str}] {timestamp} - {record.name} - {record.getMessage()}{extra_str}"

        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    use_colors: bool = True,
) -> None:
    """Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format (for production) or console format (for dev)
        use_colors: Use colors in console output (only applies to console format)
    """
    # Create handler
    handler = logging.StreamHandler(sys.stderr)

    # Choose formatter
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(use_colors=use_colors))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Set levels for noisy third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
