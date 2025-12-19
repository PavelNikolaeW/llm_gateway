"""Unit tests for admin API routes with mocked dependencies."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api.routes.admin import router
from src.api.dependencies import (
    get_db_session,
    get_admin_service,
    get_current_user_id,
    get_is_admin,
)
from src.shared.exceptions import ForbiddenError, NotFoundError
from src.shared.schemas import (
    TokenBalanceResponse,
    TokenTransactionResponse,
    UserDetailsResponse,
    UserStatsResponse,
)


@pytest.fixture
def mock_user_stats():
    """Create mock user stats."""
    return UserStatsResponse(
        user_id=456,
        dialog_count=5,
        total_tokens_used=1000,
        balance=5000,
        limit=10000,
    )


@pytest.fixture
def mock_user_details():
    """Create mock user details."""
    return UserDetailsResponse(
        user_id=456,
        dialog_count=5,
        total_tokens_used=1000,
        balance=5000,
        limit=10000,
        last_activity=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_balance_response():
    """Create mock balance response."""
    return TokenBalanceResponse(
        user_id=456,
        balance=6000,
        limit=10000,
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_transaction_response():
    """Create mock transaction response."""
    return TokenTransactionResponse(
        id=1,
        user_id=456,
        amount=1000,
        reason="admin_top_up",
        dialog_id=None,
        message_id=None,
        admin_user_id=123,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_service():
    """Create mock admin service."""
    return MagicMock()


@pytest.fixture
def test_app(mock_service):
    """Create test FastAPI app with mocked dependencies (admin user)."""
    app = FastAPI()

    # Add exception handlers
    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request, exc):
        return JSONResponse(status_code=403, content={"detail": exc.message})

    @app.exception_handler(NotFoundError)
    async def not_found_error_handler(request, exc):
        return JSONResponse(status_code=404, content={"detail": exc.message})

    app.include_router(router, prefix="/api/v1")

    # Override dependencies (admin user)
    async def mock_db():
        yield MagicMock()

    def mock_user_id():
        return 123  # Admin user ID

    def mock_is_admin():
        return True

    app.dependency_overrides[get_db_session] = mock_db
    app.dependency_overrides[get_admin_service] = lambda: mock_service
    app.dependency_overrides[get_current_user_id] = mock_user_id
    app.dependency_overrides[get_is_admin] = mock_is_admin

    return app


@pytest.fixture
def test_app_non_admin(mock_service):
    """Create test FastAPI app with mocked dependencies (non-admin user)."""
    app = FastAPI()

    # Add exception handlers
    @app.exception_handler(ForbiddenError)
    async def forbidden_error_handler(request, exc):
        return JSONResponse(status_code=403, content={"detail": exc.message})

    app.include_router(router, prefix="/api/v1")

    # Override dependencies (non-admin user)
    async def mock_db():
        yield MagicMock()

    def mock_user_id():
        return 999

    def mock_is_admin():
        return False

    app.dependency_overrides[get_db_session] = mock_db
    app.dependency_overrides[get_admin_service] = lambda: mock_service
    app.dependency_overrides[get_current_user_id] = mock_user_id
    app.dependency_overrides[get_is_admin] = mock_is_admin

    return app


@pytest.fixture
def client(test_app):
    """Create test client (admin)."""
    return TestClient(test_app)


@pytest.fixture
def client_non_admin(test_app_non_admin):
    """Create test client (non-admin)."""
    return TestClient(test_app_non_admin)


class TestListUsers:
    """Tests for GET /admin/users endpoint."""

    def test_list_users_success(self, client, mock_service, mock_user_stats):
        """Test successful user list."""
        mock_service.list_users = AsyncMock(return_value=[mock_user_stats])

        response = client.get("/api/v1/admin/users")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_id"] == 456
        assert data[0]["dialog_count"] == 5
        assert data[0]["total_tokens_used"] == 1000

    def test_list_users_empty(self, client, mock_service):
        """Test empty user list."""
        mock_service.list_users = AsyncMock(return_value=[])

        response = client.get("/api/v1/admin/users")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_users_with_pagination(self, client, mock_service, mock_user_stats):
        """Test user list with pagination."""
        mock_service.list_users = AsyncMock(return_value=[mock_user_stats])

        response = client.get("/api/v1/admin/users?skip=10&limit=5")

        assert response.status_code == 200
        call_kwargs = mock_service.list_users.call_args.kwargs
        assert call_kwargs["skip"] == 10
        assert call_kwargs["limit"] == 5

    def test_list_users_forbidden_non_admin(self, client_non_admin, mock_service):
        """Test list users returns 403 for non-admin."""
        mock_service.list_users = AsyncMock(
            side_effect=ForbiddenError("Admin access required")
        )

        response = client_non_admin.get("/api/v1/admin/users")

        assert response.status_code == 403


class TestGetUserDetails:
    """Tests for GET /admin/users/{user_id} endpoint."""

    def test_get_user_details_success(self, client, mock_service, mock_user_details):
        """Test successful user details retrieval."""
        mock_service.get_user_details = AsyncMock(return_value=mock_user_details)

        response = client.get("/api/v1/admin/users/456")

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 456
        assert data["dialog_count"] == 5
        assert "last_activity" in data

    def test_get_user_details_not_found(self, client, mock_service):
        """Test user details not found."""
        mock_service.get_user_details = AsyncMock(
            side_effect=NotFoundError("User 999 not found")
        )

        response = client.get("/api/v1/admin/users/999")

        assert response.status_code == 404

    def test_get_user_details_forbidden_non_admin(self, client_non_admin, mock_service):
        """Test get user details returns 403 for non-admin."""
        mock_service.get_user_details = AsyncMock(
            side_effect=ForbiddenError("Admin access required")
        )

        response = client_non_admin.get("/api/v1/admin/users/456")

        assert response.status_code == 403


class TestSetUserLimit:
    """Tests for PATCH /admin/users/{user_id}/limits endpoint."""

    def test_set_limit_success(self, client, mock_service, mock_balance_response):
        """Test successful limit setting."""
        mock_service.set_user_limit = AsyncMock(return_value=mock_balance_response)

        response = client.patch(
            "/api/v1/admin/users/456/limits",
            json={"limit": 20000},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 456
        mock_service.set_user_limit.assert_called_once()

    def test_set_limit_unlimited(self, client, mock_service, mock_balance_response):
        """Test setting unlimited (null) limit."""
        mock_balance_response.limit = None
        mock_service.set_user_limit = AsyncMock(return_value=mock_balance_response)

        response = client.patch(
            "/api/v1/admin/users/456/limits",
            json={"limit": None},
        )

        assert response.status_code == 200

    def test_set_limit_not_found(self, client, mock_service):
        """Test set limit for non-existent user."""
        mock_service.set_user_limit = AsyncMock(
            side_effect=NotFoundError("User 999 not found")
        )

        response = client.patch(
            "/api/v1/admin/users/999/limits",
            json={"limit": 10000},
        )

        assert response.status_code == 404

    def test_set_limit_forbidden_non_admin(self, client_non_admin, mock_service):
        """Test set limit returns 403 for non-admin."""
        mock_service.set_user_limit = AsyncMock(
            side_effect=ForbiddenError("Admin access required")
        )

        response = client_non_admin.patch(
            "/api/v1/admin/users/456/limits",
            json={"limit": 10000},
        )

        assert response.status_code == 403


class TestTopUpTokens:
    """Tests for POST /admin/users/{user_id}/tokens endpoint."""

    def test_top_up_success(self, client, mock_service, mock_balance_response, mock_transaction_response):
        """Test successful token top-up."""
        mock_service.top_up_tokens = AsyncMock(
            return_value=(mock_balance_response, mock_transaction_response)
        )

        response = client.post(
            "/api/v1/admin/users/456/tokens",
            json={"amount": 1000},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == 456

    def test_deduct_tokens_success(self, client, mock_service, mock_balance_response, mock_transaction_response):
        """Test successful token deduction (negative amount)."""
        mock_balance_response.balance = 4000
        mock_transaction_response.amount = -1000
        mock_service.top_up_tokens = AsyncMock(
            return_value=(mock_balance_response, mock_transaction_response)
        )

        response = client.post(
            "/api/v1/admin/users/456/tokens",
            json={"amount": -1000},
        )

        assert response.status_code == 200

    def test_top_up_not_found(self, client, mock_service):
        """Test top-up for non-existent user."""
        mock_service.top_up_tokens = AsyncMock(
            side_effect=NotFoundError("User 999 not found")
        )

        response = client.post(
            "/api/v1/admin/users/999/tokens",
            json={"amount": 1000},
        )

        assert response.status_code == 404

    def test_top_up_forbidden_non_admin(self, client_non_admin, mock_service):
        """Test top-up returns 403 for non-admin."""
        mock_service.top_up_tokens = AsyncMock(
            side_effect=ForbiddenError("Admin access required")
        )

        response = client_non_admin.post(
            "/api/v1/admin/users/456/tokens",
            json={"amount": 1000},
        )

        assert response.status_code == 403


class TestGetTokenHistory:
    """Tests for GET /admin/users/{user_id}/tokens/history endpoint."""

    def test_get_history_success(self, client, mock_service, mock_transaction_response):
        """Test successful history retrieval."""
        mock_service.get_token_history = AsyncMock(return_value=[mock_transaction_response])

        response = client.get("/api/v1/admin/users/456/tokens/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_id"] == 456
        assert data[0]["amount"] == 1000

    def test_get_history_empty(self, client, mock_service):
        """Test empty transaction history."""
        mock_service.get_token_history = AsyncMock(return_value=[])

        response = client.get("/api/v1/admin/users/456/tokens/history")

        assert response.status_code == 200
        assert response.json() == []

    def test_get_history_with_pagination(self, client, mock_service, mock_transaction_response):
        """Test history with pagination."""
        mock_service.get_token_history = AsyncMock(return_value=[mock_transaction_response])

        response = client.get("/api/v1/admin/users/456/tokens/history?skip=10&limit=50")

        assert response.status_code == 200
        call_kwargs = mock_service.get_token_history.call_args.kwargs
        assert call_kwargs["skip"] == 10
        assert call_kwargs["limit"] == 50

    def test_get_history_not_found(self, client, mock_service):
        """Test history for non-existent user."""
        mock_service.get_token_history = AsyncMock(
            side_effect=NotFoundError("User 999 not found")
        )

        response = client.get("/api/v1/admin/users/999/tokens/history")

        assert response.status_code == 404

    def test_get_history_forbidden_non_admin(self, client_non_admin, mock_service):
        """Test history returns 403 for non-admin."""
        mock_service.get_token_history = AsyncMock(
            side_effect=ForbiddenError("Admin access required")
        )

        response = client_non_admin.get("/api/v1/admin/users/456/tokens/history")

        assert response.status_code == 403


class TestAdminRouterWithAuth:
    """Tests for admin routes with full auth middleware."""

    def test_missing_auth_returns_401(self):
        """Test missing auth returns 401."""
        from src.api.app import create_app

        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/admin/users")

        assert response.status_code == 401
