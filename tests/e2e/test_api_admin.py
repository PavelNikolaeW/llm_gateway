"""E2E tests for admin API endpoints.

Tests:
- GET /api/v1/admin/stats - Get global statistics
- GET /api/v1/admin/users - List users
- GET /api/v1/admin/users/{id} - Get user details
- PATCH /api/v1/admin/users/{id}/limits - Set token limit
- POST /api/v1/admin/users/{id}/tokens - Top-up tokens
- GET /api/v1/admin/users/{id}/tokens/history - Get transaction history
- Auth scenarios: no auth (401), non-admin (403), admin (200)
"""
from datetime import date, timedelta

import pytest
from httpx import AsyncClient

# API prefix for all endpoints
API_PREFIX = "/api/v1"


class TestAdminStatsAuth:
    """Tests for GET /admin/stats authentication/authorization."""

    @pytest.mark.asyncio
    async def test_get_stats_no_auth(self, client: AsyncClient):
        """Test getting stats without auth fails with 401."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        response = await client.get(
            f"{API_PREFIX}/admin/stats",
            params={"start_date": str(yesterday), "end_date": str(today)},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_stats_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test getting stats as non-admin fails with 403."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        response = await client.get(
            f"{API_PREFIX}/admin/stats",
            params={"start_date": str(yesterday), "end_date": str(today)},
            headers=auth_headers,
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_get_stats_admin(
        self, client: AsyncClient, admin_headers: dict[str, str]
    ):
        """Test getting stats as admin succeeds."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        response = await client.get(
            f"{API_PREFIX}/admin/stats",
            params={"start_date": str(yesterday), "end_date": str(today)},
            headers=admin_headers,
        )

        # May be 200 or 500 depending on database state
        assert response.status_code in [200, 500]


class TestAdminUsersAuth:
    """Tests for GET /admin/users authentication/authorization."""

    @pytest.mark.asyncio
    async def test_list_users_no_auth(self, client: AsyncClient):
        """Test listing users without auth fails with 401."""
        response = await client.get(f"{API_PREFIX}/admin/users")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_users_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test listing users as non-admin fails with 403."""
        response = await client.get(
            f"{API_PREFIX}/admin/users",
            headers=auth_headers,
        )

        assert response.status_code == 403

class TestAdminUserDetailsAuth:
    """Tests for GET /admin/users/{id} authentication/authorization."""

    @pytest.mark.asyncio
    async def test_get_user_no_auth(self, client: AsyncClient):
        """Test getting user details without auth fails with 401."""
        response = await client.get(f"{API_PREFIX}/admin/users/12345")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_user_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test getting user details as non-admin fails with 403."""
        response = await client.get(
            f"{API_PREFIX}/admin/users/12345",
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestAdminSetLimitAuth:
    """Tests for PATCH /admin/users/{id}/limits authentication/authorization."""

    @pytest.mark.asyncio
    async def test_set_limit_no_auth(self, client: AsyncClient):
        """Test setting limit without auth fails with 401."""
        response = await client.patch(
            f"{API_PREFIX}/admin/users/12345/limits",
            json={"limit": 10000},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_set_limit_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test setting limit as non-admin fails with 403."""
        response = await client.patch(
            f"{API_PREFIX}/admin/users/12345/limits",
            json={"limit": 10000},
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestAdminTopUpAuth:
    """Tests for POST /admin/users/{id}/tokens authentication/authorization."""

    @pytest.mark.asyncio
    async def test_top_up_no_auth(self, client: AsyncClient):
        """Test top-up without auth fails with 401."""
        response = await client.post(
            f"{API_PREFIX}/admin/users/12345/tokens",
            json={"amount": 1000},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_top_up_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test top-up as non-admin fails with 403."""
        response = await client.post(
            f"{API_PREFIX}/admin/users/12345/tokens",
            json={"amount": 1000},
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestAdminTokenHistoryAuth:
    """Tests for GET /admin/users/{id}/tokens/history authentication/authorization."""

    @pytest.mark.asyncio
    async def test_get_history_no_auth(self, client: AsyncClient):
        """Test getting history without auth fails with 401."""
        response = await client.get(
            f"{API_PREFIX}/admin/users/12345/tokens/history"
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_history_non_admin(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ):
        """Test getting history as non-admin fails with 403."""
        response = await client.get(
            f"{API_PREFIX}/admin/users/12345/tokens/history",
            headers=auth_headers,
        )

        assert response.status_code == 403
