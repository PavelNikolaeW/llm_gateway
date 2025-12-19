"""Unit tests for messages API routes with mocked dependencies."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.messages import router
from src.api.dependencies import (
    get_db_session,
    get_message_service,
    get_current_user_id,
    get_is_admin,
)
from src.domain.message_service import MessageService
from src.shared.exceptions import (
    ValidationError,
    NotFoundError,
    ForbiddenError,
    InsufficientTokensError,
    LLMError,
    LLMTimeoutError,
)
from src.shared.schemas import MessageResponse, StreamChunk


@pytest.fixture
def mock_message_response():
    """Create mock message response."""
    return MessageResponse(
        id=uuid.uuid4(),
        dialog_id=uuid.uuid4(),
        role="assistant",
        content="Hello! How can I help you?",
        prompt_tokens=10,
        completion_tokens=20,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_service():
    """Create mock message service."""
    return MagicMock()


@pytest.fixture
def test_app(mock_service):
    """Create test FastAPI app with mocked dependencies."""
    from fastapi.responses import JSONResponse

    app = FastAPI()

    # Add exception handlers
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request, exc):
        return JSONResponse(status_code=400, content={"detail": exc.message})

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request, exc):
        return JSONResponse(status_code=404, content={"detail": exc.message})

    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request, exc):
        return JSONResponse(status_code=403, content={"detail": exc.message})

    @app.exception_handler(InsufficientTokensError)
    async def insufficient_tokens_error_handler(request, exc):
        return JSONResponse(status_code=402, content={"detail": exc.message})

    @app.exception_handler(LLMTimeoutError)
    async def llm_timeout_error_handler(request, exc):
        return JSONResponse(status_code=504, content={"detail": exc.message})

    @app.exception_handler(LLMError)
    async def llm_error_handler(request, exc):
        return JSONResponse(status_code=500, content={"detail": exc.message})

    app.include_router(router, prefix="/api/v1")

    # Override dependencies
    async def mock_db():
        yield MagicMock()

    def mock_user_id():
        return 123

    def mock_is_admin():
        return False

    app.dependency_overrides[get_db_session] = mock_db
    app.dependency_overrides[get_message_service] = lambda: mock_service
    app.dependency_overrides[get_current_user_id] = mock_user_id
    app.dependency_overrides[get_is_admin] = mock_is_admin

    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestSendMessageSync:
    """Tests for POST /dialogs/{id}/messages/sync endpoint."""

    def test_send_message_success(self, client, mock_service, mock_message_response):
        """Test successful message send."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(return_value=mock_message_response)

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["role"] == "assistant"
        assert "content" in data
        mock_service.send_message.assert_called_once()

    def test_send_message_dialog_not_found_returns_404(self, client, mock_service):
        """Test send message to non-existent dialog returns 404."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(
            side_effect=NotFoundError(f"Dialog {dialog_id} not found")
        )

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_send_message_forbidden_returns_403(self, client, mock_service):
        """Test send message to other user's dialog returns 403."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(
            side_effect=ForbiddenError(f"Access denied to dialog {dialog_id}")
        )

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()

    def test_send_message_insufficient_tokens_returns_402(self, client, mock_service):
        """Test send message with insufficient tokens returns 402."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(
            side_effect=InsufficientTokensError("Not enough tokens")
        )

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 402
        assert "tokens" in response.json()["detail"].lower()

    def test_send_message_llm_timeout_returns_504(self, client, mock_service):
        """Test LLM timeout returns 504."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(
            side_effect=LLMTimeoutError("LLM request timed out")
        )

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()

    def test_send_message_llm_error_returns_500(self, client, mock_service):
        """Test LLM error returns 500."""
        dialog_id = uuid.uuid4()
        mock_service.send_message = AsyncMock(side_effect=LLMError("LLM service failed"))

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages/sync",
            json={"content": "Hello"},
        )

        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


class TestSendMessageStream:
    """Tests for POST /dialogs/{id}/messages endpoint (SSE streaming)."""

    def test_send_message_stream_returns_sse(self, client, mock_service):
        """Test streaming endpoint returns SSE content type."""
        dialog_id = uuid.uuid4()

        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", done=False)
            yield StreamChunk(
                content="",
                done=True,
                message_id=uuid.uuid4(),
                prompt_tokens=10,
                completion_tokens=5,
            )

        mock_service.send_message_stream = mock_stream

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages",
            json={"content": "Hello"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_send_message_stream_yields_chunks(self, client, mock_service):
        """Test streaming endpoint yields proper SSE events."""
        dialog_id = uuid.uuid4()
        message_id = uuid.uuid4()

        async def mock_stream(*args, **kwargs):
            yield StreamChunk(content="Hello", done=False)
            yield StreamChunk(content=" World", done=False)
            yield StreamChunk(
                content="",
                done=True,
                message_id=message_id,
                prompt_tokens=10,
                completion_tokens=5,
            )

        mock_service.send_message_stream = mock_stream

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages",
            json={"content": "Hello"},
        )

        assert response.status_code == 200

        # Parse SSE events
        content = response.text
        assert "data:" in content
        assert "Hello" in content
        assert "World" in content


class TestGetMessages:
    """Tests for GET /dialogs/{id}/messages endpoint."""

    def test_get_messages_success(self, client, mock_service, mock_message_response):
        """Test successful message retrieval."""
        dialog_id = uuid.uuid4()
        mock_service.get_messages = AsyncMock(return_value=[mock_message_response])

        response = client.get(f"/api/v1/dialogs/{dialog_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["role"] == "assistant"

    def test_get_messages_empty(self, client, mock_service):
        """Test get messages for dialog with no messages."""
        dialog_id = uuid.uuid4()
        mock_service.get_messages = AsyncMock(return_value=[])

        response = client.get(f"/api/v1/dialogs/{dialog_id}/messages")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_messages_with_pagination(self, client, mock_service, mock_message_response):
        """Test get messages with pagination parameters."""
        dialog_id = uuid.uuid4()
        mock_service.get_messages = AsyncMock(return_value=[mock_message_response])

        response = client.get(
            f"/api/v1/dialogs/{dialog_id}/messages?skip=10&limit=50"
        )

        assert response.status_code == 200
        mock_service.get_messages.assert_called_once()
        call_kwargs = mock_service.get_messages.call_args.kwargs
        assert call_kwargs["skip"] == 10
        assert call_kwargs["limit"] == 50

    def test_get_messages_dialog_not_found_returns_404(self, client, mock_service):
        """Test get messages for non-existent dialog returns 404."""
        dialog_id = uuid.uuid4()
        mock_service.get_messages = AsyncMock(
            side_effect=NotFoundError(f"Dialog {dialog_id} not found")
        )

        response = client.get(f"/api/v1/dialogs/{dialog_id}/messages")

        assert response.status_code == 404

    def test_get_messages_forbidden_returns_403(self, client, mock_service):
        """Test get messages for other user's dialog returns 403."""
        dialog_id = uuid.uuid4()
        mock_service.get_messages = AsyncMock(
            side_effect=ForbiddenError(f"Access denied to dialog {dialog_id}")
        )

        response = client.get(f"/api/v1/dialogs/{dialog_id}/messages")

        assert response.status_code == 403


class TestMessagesRouterWithAuth:
    """Tests for messages routes with full auth middleware."""

    def test_missing_auth_returns_401(self):
        """Test missing auth returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)
        dialog_id = uuid.uuid4()

        response = client.post(
            f"/api/v1/dialogs/{dialog_id}/messages",
            json={"content": "Hello"},
        )

        assert response.status_code == 401

    def test_get_messages_missing_auth_returns_401(self):
        """Test get messages without auth returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)
        dialog_id = uuid.uuid4()

        response = client.get(f"/api/v1/dialogs/{dialog_id}/messages")

        assert response.status_code == 401
