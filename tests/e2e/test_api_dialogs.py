"""E2E tests for dialog API endpoints.

Tests:
- POST /api/v1/dialogs - Create dialog
- GET /api/v1/dialogs - List dialogs
- GET /api/v1/dialogs/{id} - Get dialog
- POST /api/v1/dialogs/{id}/messages - Send message
- GET /api/v1/users/me/tokens - Get token balance
- Auth scenarios: valid JWT, invalid JWT, expired JWT
"""
import pytest
from httpx import AsyncClient

# API prefix for all endpoints
API_PREFIX = "/api/v1"


class TestDialogAuthScenarios:
    """Authentication tests for dialog endpoints."""

    @pytest.mark.asyncio
    async def test_create_dialog_no_auth(self, client: AsyncClient):
        """Test creating a dialog without auth fails with 401."""
        response = await client.post(
            f"{API_PREFIX}/dialogs",
            json={"model_name": "gpt-4"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_dialog_invalid_token(
        self, client: AsyncClient, invalid_token: str
    ):
        """Test creating a dialog with invalid token fails with 401."""
        response = await client.post(
            f"{API_PREFIX}/dialogs",
            json={"model_name": "gpt-4"},
            headers={"Authorization": f"Bearer {invalid_token}"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_dialog_expired_token(
        self, client: AsyncClient, expired_token: str
    ):
        """Test creating a dialog with expired token fails with 401."""
        response = await client.post(
            f"{API_PREFIX}/dialogs",
            json={"model_name": "gpt-4"},
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert response.status_code == 401


class TestDialogListAuth:
    """Authentication tests for GET /dialogs."""

    @pytest.mark.asyncio
    async def test_list_dialogs_no_auth(self, client: AsyncClient):
        """Test listing dialogs without auth fails with 401."""
        response = await client.get(f"{API_PREFIX}/dialogs")

        assert response.status_code == 401


class TestDialogGetAuth:
    """Authentication tests for GET /dialogs/{id}."""

    @pytest.mark.asyncio
    async def test_get_dialog_invalid_uuid(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test getting a dialog with invalid UUID returns 422."""
        response = await client.get(
            f"{API_PREFIX}/dialogs/invalid-uuid", headers=auth_headers
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_get_dialog_no_auth(self, client: AsyncClient):
        """Test getting a dialog without auth fails with 401."""
        import uuid

        response = await client.get(f"{API_PREFIX}/dialogs/{uuid.uuid4()}")

        assert response.status_code == 401


class TestMessageAuth:
    """Authentication tests for POST /dialogs/{id}/messages."""

    @pytest.mark.asyncio
    async def test_send_message_no_auth(self, client: AsyncClient):
        """Test sending a message without auth fails with 401."""
        import uuid

        response = await client.post(
            f"{API_PREFIX}/dialogs/{uuid.uuid4()}/messages",
            json={"content": "Hello"},
        )

        assert response.status_code == 401


class TestTokensAuth:
    """Authentication tests for GET /users/me/tokens."""

    @pytest.mark.asyncio
    async def test_get_tokens_no_auth(self, client: AsyncClient):
        """Test getting tokens without auth fails with 401."""
        response = await client.get(f"{API_PREFIX}/users/me/tokens")

        assert response.status_code == 401
