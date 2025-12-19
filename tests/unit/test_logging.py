"""Unit tests for structured logging configuration."""
import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest

from src.config.logging import (
    JSONFormatter,
    ConsoleFormatter,
    configure_logging,
    get_logger,
)


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_formats_basic_message(self):
        """Test basic message formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_includes_extra_fields(self):
        """Test extra fields are included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Request processed",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        record.user_id = 456
        record.path = "/api/v1/dialogs"
        record.method = "POST"
        record.status_code = 200
        record.duration_ms = 150

        output = formatter.format(record)
        data = json.loads(output)

        assert data["request_id"] == "abc-123"
        assert data["user_id"] == 456
        assert data["path"] == "/api/v1/dialogs"
        assert data["method"] == "POST"
        assert data["status_code"] == 200
        assert data["duration_ms"] == 150

    def test_includes_exception_info(self):
        """Test exception info is included."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_omits_none_extra_fields(self):
        """Test None extra fields are not included."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.request_id = None
        record.user_id = None

        output = formatter.format(record)
        data = json.loads(output)

        assert "request_id" not in data
        assert "user_id" not in data


class TestConsoleFormatter:
    """Tests for console log formatter."""

    def test_formats_basic_message(self):
        """Test basic message formatting."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.logger" in output
        assert "Test message" in output

    def test_includes_extra_fields(self):
        """Test extra fields are included in console output."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Request",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        record.user_id = 456
        record.status_code = 200

        output = formatter.format(record)

        assert "request_id=abc-123" in output
        assert "user_id=456" in output
        assert "status_code=200" in output

    def test_includes_exception_info(self):
        """Test exception info is included."""
        formatter = ConsoleFormatter(use_colors=False)
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error",
                args=(),
                exc_info=sys.exc_info(),
            )

        output = formatter.format(record)

        assert "ValueError" in output
        assert "Test error" in output


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_with_json_format(self):
        """Test logging configured with JSON format."""
        configure_logging(level="DEBUG", json_format=True)

        logger = logging.getLogger("test.json")
        # Check handler uses JSON formatter
        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_configure_with_console_format(self):
        """Test logging configured with console format."""
        configure_logging(level="DEBUG", json_format=False)

        root = logging.getLogger()
        assert len(root.handlers) > 0
        assert isinstance(root.handlers[0].formatter, ConsoleFormatter)

    def test_configure_sets_level(self):
        """Test logging level is set correctly."""
        configure_logging(level="WARNING", json_format=False)

        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_quiets_noisy_loggers(self):
        """Test noisy third-party loggers are quieted."""
        configure_logging(level="DEBUG", json_format=False)

        # These should be set to WARNING even when root is DEBUG
        assert logging.getLogger("uvicorn").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING


class TestGetLogger:
    """Tests for get_logger helper."""

    def test_returns_logger(self):
        """Test get_logger returns a logger."""
        logger = get_logger("test.module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_same_name_returns_same_logger(self):
        """Test same name returns the same logger instance."""
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")

        assert logger1 is logger2
