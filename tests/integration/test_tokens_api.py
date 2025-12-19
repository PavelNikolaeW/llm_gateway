"""Integration tests for Tokens API endpoints.

Tests the HTTP layer including:
- Authentication middleware
- Response formatting
"""
import time

import jwt
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.app import create_app
from src.integrations.jwt_validator import JWTClaims


# Test JWT secret
TEST_SECRET = "test-secret-key-for-integration-tests"


def create_test_token(user_id: int, is_admin: bool = False) -> str:
    """Create a test JWT token."""
    payload = {
        "user_id": user_id,
        "is_admin": is_admin,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def mock_jwt_validator():
    """Mock JWT validator for all tests."""
    with patch("src.api.app.JWTValidator") as mock_class:
        mock_instance = MagicMock()

        def validate_side_effect(auth_header: str):
            token = auth_header.replace("Bearer ", "")
            try:
                payload = jwt.decode(token, TEST_SECRET, algorithms=["HS256"])
                return JWTClaims(
                    user_id=payload["user_id"],
                    is_admin=payload.get("is_admin", False),
                    exp=payload["exp"],
                    iat=payload["iat"],
                    nbf=None,
                    raw_claims=payload,
                )
            except jwt.InvalidTokenError as e:
                from src.shared.exceptions import UnauthorizedError
                raise UnauthorizedError(f"Invalid token: {e}")

        mock_instance.validate.side_effect = validate_side_effect
        mock_class.return_value = mock_instance
        yield mock_instance


class TestTokensAuthentication:
    """Tests for token endpoints authentication."""

    def test_get_tokens_no_auth_returns_401(self, mock_jwt_validator):
        """Test getting tokens without auth returns 401."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"

    def test_get_tokens_invalid_token_returns_401(self, mock_jwt_validator):
        """Test getting tokens with invalid token returns 401."""
        app = create_app()
        client = TestClient(app)

        response = client.get(
            "/api/v1/users/me/tokens",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401


class TestTokensResponseFormat:
    """Tests for response format on token endpoints."""

    def test_request_id_in_error_response(self, mock_jwt_validator):
        """Test request_id is in error response body."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/users/me/tokens")

        assert response.status_code == 401
        data = response.json()
        assert "request_id" in data

    def test_error_response_is_json(self):
        """Test error response is JSON."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/v1/users/me/tokens")

        assert response.headers["content-type"] == "application/json"


class TestTokensWithDatabase:
    """Integration tests that use the actual database."""

    @pytest.mark.asyncio
    async def test_token_balance_model(self, session, mock_jwt_validator):
        """Test creating and retrieving token balance from database."""
        import random
        from src.data.models import TokenBalance

        # Use random user_id to avoid conflicts with other tests
        user_id = random.randint(100000, 999999)
        balance = TokenBalance(
            user_id=user_id,
            balance=5000,
            limit=None,
        )
        session.add(balance)
        await session.commit()

        # Query back
        from sqlalchemy import select
        result = await session.execute(
            select(TokenBalance).where(TokenBalance.user_id == user_id)
        )
        retrieved = result.scalar_one()

        assert retrieved.balance == 5000
        assert retrieved.limit is None
        assert retrieved.user_id == user_id
