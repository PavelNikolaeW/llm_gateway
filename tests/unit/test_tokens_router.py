"""Unit tests for tokens API routes with mocked dependencies."""
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.tokens import router
from src.api.dependencies import (
    get_db_session,
    get_token_service,
    get_current_user_id,
)
from src.shared.schemas import TokenStatsResponse


@pytest.fixture
def mock_token_stats():
    """Create mock token stats response."""
    return TokenStatsResponse(
        balance=5000,
        total_used=2500,
        limit=10000,
    )


@pytest.fixture
def mock_service():
    """Create mock token service."""
    return MagicMock()


@pytest.fixture
def test_app(mock_service):
    """Create test FastAPI app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override dependencies
    async def mock_db():
        yield MagicMock()

    def mock_user_id():
        return 123

    app.dependency_overrides[get_db_session] = mock_db
    app.dependency_overrides[get_token_service] = lambda: mock_service
    app.dependency_overrides[get_current_user_id] = mock_user_id

    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


class TestGetMyTokens:
    """Tests for GET /users/me/tokens endpoint."""

    def test_get_tokens_success(self, client, mock_service, mock_token_stats):
        """Test successful token stats retrieval."""
        mock_service.get_token_stats = AsyncMock(return_value=mock_token_stats)

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 5000
        assert data["total_used"] == 2500
        assert data["limit"] == 10000
        mock_service.get_token_stats.assert_called_once()

    def test_get_tokens_no_limit(self, client, mock_service):
        """Test token stats with no limit set."""
        mock_service.get_token_stats = AsyncMock(
            return_value=TokenStatsResponse(
                balance=1000,
                total_used=500,
                limit=None,
            )
        )

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 1000
        assert data["total_used"] == 500
        assert data["limit"] is None

    def test_get_tokens_zero_balance(self, client, mock_service):
        """Test token stats with zero balance."""
        mock_service.get_token_stats = AsyncMock(
            return_value=TokenStatsResponse(
                balance=0,
                total_used=10000,
                limit=10000,
            )
        )

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 0
        assert data["total_used"] == 10000

    def test_get_tokens_new_user(self, client, mock_service):
        """Test token stats for new user with no usage."""
        mock_service.get_token_stats = AsyncMock(
            return_value=TokenStatsResponse(
                balance=0,
                total_used=0,
                limit=None,
            )
        )

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 0
        assert data["total_used"] == 0
        assert data["limit"] is None

    def test_get_tokens_passes_user_id(self, client, mock_service, mock_token_stats):
        """Test that user_id from JWT is passed to service."""
        mock_service.get_token_stats = AsyncMock(return_value=mock_token_stats)

        client.get("/api/v1/users/me/tokens")

        # Verify user_id (123 from mock) was passed
        call_kwargs = mock_service.get_token_stats.call_args.kwargs
        assert call_kwargs["user_id"] == 123


class TestTokensRouterWithAuth:
    """Tests for tokens routes with full auth middleware."""

    def test_missing_auth_returns_401(self):
        """Test missing auth returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 401
