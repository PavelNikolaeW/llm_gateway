"""Unit tests for Prometheus metrics."""
import pytest
from prometheus_client import REGISTRY

from src.shared.metrics import (
    record_http_request,
    record_llm_request,
    record_token_usage,
    record_token_balance,
    record_dialog_created,
    record_message_sent,
    record_error,
    _normalize_path,
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION_SECONDS,
    TOKENS_USED_TOTAL,
    TOKENS_PROMPT_TOTAL,
    TOKENS_COMPLETION_TOTAL,
    TOKEN_BALANCE,
    DIALOGS_CREATED_TOTAL,
    MESSAGES_SENT_TOTAL,
    ERRORS_TOTAL,
)


class TestNormalizePath:
    """Tests for path normalization."""

    def test_normalizes_uuid(self):
        """Test UUID in path is replaced with {id}."""
        path = "/api/v1/dialogs/123e4567-e89b-12d3-a456-426614174000"
        assert _normalize_path(path) == "/api/v1/dialogs/{id}"

    def test_normalizes_multiple_uuids(self):
        """Test multiple UUIDs are replaced."""
        path = "/api/v1/dialogs/123e4567-e89b-12d3-a456-426614174000/messages/987fcdeb-51a2-3bc4-d567-890123456789"
        assert _normalize_path(path) == "/api/v1/dialogs/{id}/messages/{id}"

    def test_normalizes_numeric_id(self):
        """Test numeric IDs are replaced."""
        path = "/api/v1/users/123/tokens"
        assert _normalize_path(path) == "/api/v1/users/{id}/tokens"

    def test_preserves_path_without_ids(self):
        """Test path without IDs is unchanged."""
        path = "/api/v1/dialogs"
        assert _normalize_path(path) == "/api/v1/dialogs"

    def test_handles_trailing_slash(self):
        """Test path with trailing slash."""
        path = "/api/v1/users/42/"
        assert _normalize_path(path) == "/api/v1/users/{id}/"


class TestRecordHttpRequest:
    """Tests for HTTP request metrics."""

    def test_records_request_count(self):
        """Test request count is recorded."""
        initial = HTTP_REQUESTS_TOTAL.labels(
            method="GET",
            path="/test",
            status_code="200",
        )._value.get()

        record_http_request("GET", "/test", 200, 0.1)

        assert HTTP_REQUESTS_TOTAL.labels(
            method="GET",
            path="/test",
            status_code="200",
        )._value.get() == initial + 1

    def test_records_request_duration(self):
        """Test request duration is recorded."""
        # Record a request
        record_http_request("POST", "/api/test", 201, 0.5)

        # Duration histogram should have data
        # We just verify no exception is raised
        assert True

    def test_normalizes_path_in_metrics(self):
        """Test path is normalized in metrics."""
        record_http_request(
            "GET",
            "/api/v1/dialogs/123e4567-e89b-12d3-a456-426614174000",
            200,
            0.1,
        )

        # Should be recorded under normalized path
        count = HTTP_REQUESTS_TOTAL.labels(
            method="GET",
            path="/api/v1/dialogs/{id}",
            status_code="200",
        )._value.get()
        assert count >= 1


class TestRecordLlmRequest:
    """Tests for LLM request metrics."""

    def test_records_llm_request_count(self):
        """Test LLM request count is recorded."""
        initial = LLM_REQUESTS_TOTAL.labels(
            provider="openai",
            model="gpt-4",
            status="success",
        )._value.get()

        record_llm_request(
            provider="openai",
            model="gpt-4",
            status="success",
            duration=1.5,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert LLM_REQUESTS_TOTAL.labels(
            provider="openai",
            model="gpt-4",
            status="success",
        )._value.get() == initial + 1

    def test_records_token_counts(self):
        """Test token counts are recorded."""
        initial_prompt = TOKENS_PROMPT_TOTAL.labels(model="gpt-4")._value.get()
        initial_completion = TOKENS_COMPLETION_TOTAL.labels(model="gpt-4")._value.get()

        record_llm_request(
            provider="openai",
            model="gpt-4",
            status="success",
            duration=1.0,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert TOKENS_PROMPT_TOTAL.labels(model="gpt-4")._value.get() == initial_prompt + 100
        assert TOKENS_COMPLETION_TOTAL.labels(model="gpt-4")._value.get() == initial_completion + 50

    def test_handles_zero_tokens(self):
        """Test zero tokens doesn't increment counters."""
        initial_prompt = TOKENS_PROMPT_TOTAL.labels(model="claude-3")._value.get()

        record_llm_request(
            provider="anthropic",
            model="claude-3",
            status="error",
            duration=0.5,
            prompt_tokens=0,
            completion_tokens=0,
        )

        # Should not increment when 0
        assert TOKENS_PROMPT_TOTAL.labels(model="claude-3")._value.get() == initial_prompt


class TestRecordTokenUsage:
    """Tests for token usage metrics."""

    def test_records_token_usage(self):
        """Test token usage is recorded."""
        initial = TOKENS_USED_TOTAL.labels(user_id="999", model="test-model")._value.get()

        record_token_usage(999, "test-model", 500)

        assert TOKENS_USED_TOTAL.labels(
            user_id="999",
            model="test-model",
        )._value.get() == initial + 500


class TestRecordTokenBalance:
    """Tests for token balance metrics."""

    def test_records_token_balance(self):
        """Test token balance is set."""
        record_token_balance(888, 10000)

        assert TOKEN_BALANCE.labels(user_id="888")._value.get() == 10000

    def test_updates_token_balance(self):
        """Test token balance updates correctly."""
        record_token_balance(777, 5000)
        record_token_balance(777, 4500)

        assert TOKEN_BALANCE.labels(user_id="777")._value.get() == 4500


class TestRecordDialogCreated:
    """Tests for dialog creation metrics."""

    def test_increments_dialog_count(self):
        """Test dialog count is incremented."""
        initial = DIALOGS_CREATED_TOTAL._value.get()

        record_dialog_created()

        assert DIALOGS_CREATED_TOTAL._value.get() == initial + 1


class TestRecordMessageSent:
    """Tests for message sent metrics."""

    def test_increments_user_message_count(self):
        """Test user message count is incremented."""
        initial = MESSAGES_SENT_TOTAL.labels(role="user")._value.get()

        record_message_sent("user")

        assert MESSAGES_SENT_TOTAL.labels(role="user")._value.get() == initial + 1

    def test_increments_assistant_message_count(self):
        """Test assistant message count is incremented."""
        initial = MESSAGES_SENT_TOTAL.labels(role="assistant")._value.get()

        record_message_sent("assistant")

        assert MESSAGES_SENT_TOTAL.labels(role="assistant")._value.get() == initial + 1


class TestRecordError:
    """Tests for error metrics."""

    def test_records_error(self):
        """Test error is recorded."""
        initial = ERRORS_TOTAL.labels(
            error_type="VALIDATION_ERROR",
            path="/api/test",
        )._value.get()

        record_error("VALIDATION_ERROR", "/api/test")

        assert ERRORS_TOTAL.labels(
            error_type="VALIDATION_ERROR",
            path="/api/test",
        )._value.get() == initial + 1

    def test_normalizes_error_path(self):
        """Test error path is normalized."""
        record_error(
            "NOT_FOUND",
            "/api/v1/dialogs/123e4567-e89b-12d3-a456-426614174000",
        )

        count = ERRORS_TOTAL.labels(
            error_type="NOT_FOUND",
            path="/api/v1/dialogs/{id}",
        )._value.get()
        assert count >= 1
