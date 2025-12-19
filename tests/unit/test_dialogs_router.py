"""Unit tests for dialogs API routes with mocked dependencies."""
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.routes.dialogs import router
from src.api.dependencies import get_db_session, get_dialog_service, get_current_user_id, get_is_admin
from src.shared.exceptions import ValidationError, NotFoundError, ForbiddenError
from src.shared.schemas import DialogResponse, DialogList


@pytest.fixture
def mock_dialog_response():
    """Create mock dialog response."""
    return DialogResponse(
        id=uuid.uuid4(),
        user_id=123,
        title="Test Dialog",
        system_prompt="You are helpful.",
        model_name="gpt-3.5-turbo",
        agent_config=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_dialog_list(mock_dialog_response):
    """Create mock dialog list response."""
    return DialogList(
        items=[mock_dialog_response],
        total=1,
        page=1,
        page_size=20,
        has_next=False,
    )


@pytest.fixture
def mock_service():
    """Create mock dialog service."""
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

    app.include_router(router, prefix="/api/v1")

    # Override dependencies
    async def mock_db():
        yield MagicMock()

    def mock_user_id():
        return 123

    def mock_is_admin():
        return False

    app.dependency_overrides[get_db_session] = mock_db
    app.dependency_overrides[get_dialog_service] = lambda: mock_service
    app.dependency_overrides[get_current_user_id] = mock_user_id
    app.dependency_overrides[get_is_admin] = mock_is_admin

    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestCreateDialog:
    """Tests for POST /dialogs endpoint."""

    def test_create_dialog_success(self, client, mock_service, mock_dialog_response):
        """Test successful dialog creation."""
        mock_service.create_dialog = AsyncMock(return_value=mock_dialog_response)

        response = client.post(
            "/api/v1/dialogs",
            json={"title": "Test Dialog"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Dialog"
        assert data["user_id"] == 123
        mock_service.create_dialog.assert_called_once()

    def test_create_dialog_with_all_fields(self, client, mock_service, mock_dialog_response):
        """Test dialog creation with all optional fields."""
        mock_service.create_dialog = AsyncMock(return_value=mock_dialog_response)

        response = client.post(
            "/api/v1/dialogs",
            json={
                "title": "Test Dialog",
                "system_prompt": "You are helpful.",
                "model_name": "gpt-4",
                "agent_config": {"temperature": 0.7},
            },
        )

        assert response.status_code == 201
        mock_service.create_dialog.assert_called_once()

    def test_create_dialog_invalid_model_returns_400(self, client, mock_service):
        """Test dialog creation with invalid model returns 400."""
        mock_service.create_dialog = AsyncMock(
            side_effect=ValidationError("Invalid model_name 'invalid-model'")
        )

        response = client.post(
            "/api/v1/dialogs",
            json={"title": "Test", "model_name": "invalid-model"},
        )

        assert response.status_code == 400
        assert "Invalid model_name" in response.json()["detail"]

    def test_create_dialog_empty_body(self, client, mock_service, mock_dialog_response):
        """Test dialog creation with empty body (uses defaults)."""
        mock_service.create_dialog = AsyncMock(return_value=mock_dialog_response)

        response = client.post("/api/v1/dialogs", json={})

        assert response.status_code == 201


class TestListDialogs:
    """Tests for GET /dialogs endpoint."""

    def test_list_dialogs_success(self, client, mock_service, mock_dialog_list):
        """Test successful dialog listing."""
        mock_service.list_dialogs = AsyncMock(return_value=mock_dialog_list)

        response = client.get("/api/v1/dialogs")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert data["page"] == 1
        assert data["page_size"] == 20

    def test_list_dialogs_with_pagination(self, client, mock_service, mock_dialog_list):
        """Test dialog listing with pagination parameters."""
        mock_service.list_dialogs = AsyncMock(return_value=mock_dialog_list)

        response = client.get("/api/v1/dialogs?page=2&page_size=10")

        assert response.status_code == 200
        mock_service.list_dialogs.assert_called_once()
        # Verify pagination params were passed
        call_args = mock_service.list_dialogs.call_args
        assert call_args[0][1] == 123  # user_id
        assert call_args[0][2] == 2  # page
        assert call_args[0][3] == 10  # page_size


class TestGetDialog:
    """Tests for GET /dialogs/{dialog_id} endpoint."""

    def test_get_dialog_success(self, client, mock_service, mock_dialog_response):
        """Test successful dialog retrieval."""
        dialog_id = mock_dialog_response.id
        mock_service.get_dialog = AsyncMock(return_value=mock_dialog_response)

        response = client.get(f"/api/v1/dialogs/{dialog_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(dialog_id)
        assert data["title"] == "Test Dialog"

    def test_get_dialog_not_found_returns_404(self, client, mock_service):
        """Test get dialog returns 404 for non-existent dialog."""
        dialog_id = uuid.uuid4()
        mock_service.get_dialog = AsyncMock(
            side_effect=NotFoundError(f"Dialog {dialog_id} not found")
        )

        response = client.get(f"/api/v1/dialogs/{dialog_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_dialog_forbidden_returns_403(self, client, mock_service):
        """Test get dialog returns 403 for other user's dialog."""
        dialog_id = uuid.uuid4()
        mock_service.get_dialog = AsyncMock(
            side_effect=ForbiddenError(f"Access denied to dialog {dialog_id}")
        )

        response = client.get(f"/api/v1/dialogs/{dialog_id}")

        assert response.status_code == 403
        assert "access denied" in response.json()["detail"].lower()

    def test_get_dialog_admin_can_access(self, mock_service, mock_dialog_response):
        """Test admin can access any dialog."""
        # Create app with admin user
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        async def mock_db():
            yield MagicMock()

        app.dependency_overrides[get_db_session] = mock_db
        app.dependency_overrides[get_dialog_service] = lambda: mock_service
        app.dependency_overrides[get_current_user_id] = lambda: 999  # Different user
        app.dependency_overrides[get_is_admin] = lambda: True  # But admin

        client = TestClient(app)
        dialog_id = mock_dialog_response.id
        mock_service.get_dialog = AsyncMock(return_value=mock_dialog_response)

        response = client.get(f"/api/v1/dialogs/{dialog_id}")

        assert response.status_code == 200
        # Verify is_admin was passed to service
        call_args = mock_service.get_dialog.call_args
        assert call_args[0][3] is True  # is_admin


class TestDialogsRouterWithAuth:
    """Tests for dialogs routes with full auth middleware."""

    def test_missing_auth_returns_401(self):
        """Test missing auth returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post("/api/v1/dialogs", json={"title": "Test"})

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"
        assert "Authorization header required" in data["message"]

    def test_invalid_token_returns_401(self):
        """Test invalid token returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.post(
            "/api/v1/dialogs",
            json={"title": "Test"},
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401
