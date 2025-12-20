"""Locust load testing file for LLM Gateway API.

Run with: locust -f tests/load/locustfile.py --host http://localhost:8000
"""
import random
import time
import uuid

import jwt
from locust import HttpUser, between, task

# JWT configuration for test tokens
JWT_SECRET = "test-secret-key-for-development-only"
JWT_ALGORITHM = "HS256"


def create_jwt_token(user_id: int, is_admin: bool = False) -> str:
    """Create a JWT token for testing.

    Args:
        user_id: User ID to embed in token
        is_admin: Whether user is admin

    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "is_admin": is_admin,
        "exp": int(time.time()) + 3600,  # 1 hour
        "iat": int(time.time()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


class DialogUser(HttpUser):
    """User that creates and interacts with dialogs."""

    wait_time = between(1, 3)
    weight = 3

    def on_start(self):
        """Initialize user with auth token."""
        self.user_id = random.randint(1000, 9999)
        self.token = create_jwt_token(self.user_id)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.dialog_ids: list[str] = []

    @task(3)
    def create_dialog(self):
        """Create a new dialog."""
        response = self.client.post(
            "/api/v1/dialogs",
            json={
                "title": f"Load test dialog {uuid.uuid4().hex[:8]}",
                "model": "gpt-4",
            },
            headers=self.headers,
        )
        if response.status_code == 201:
            data = response.json()
            self.dialog_ids.append(data["id"])

    @task(2)
    def list_dialogs(self):
        """List user's dialogs."""
        self.client.get("/api/v1/dialogs", headers=self.headers)

    @task(1)
    def get_dialog(self):
        """Get a specific dialog."""
        if self.dialog_ids:
            dialog_id = random.choice(self.dialog_ids)
            self.client.get(f"/api/v1/dialogs/{dialog_id}", headers=self.headers)


class TokenUser(HttpUser):
    """User that checks token balance."""

    wait_time = between(2, 5)
    weight = 2

    def on_start(self):
        """Initialize user with auth token."""
        self.user_id = random.randint(10000, 19999)
        self.token = create_jwt_token(self.user_id)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task
    def check_balance(self):
        """Check token balance."""
        self.client.get("/api/v1/users/me/tokens", headers=self.headers)


class HealthCheckUser(HttpUser):
    """User that performs health checks."""

    wait_time = between(5, 10)
    weight = 1

    @task
    def health_check(self):
        """Check API health."""
        self.client.get("/health")

    @task
    def models_list(self):
        """List available models (public endpoint)."""
        self.client.get("/api/v1/models")


class AdminUser(HttpUser):
    """Admin user for admin operations."""

    wait_time = between(3, 6)
    weight = 1

    def on_start(self):
        """Initialize admin user."""
        self.admin_id = random.randint(1, 10)
        self.token = create_jwt_token(self.admin_id, is_admin=True)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(2)
    def list_users(self):
        """List users."""
        self.client.get("/api/v1/admin/users", headers=self.headers)

    @task(1)
    def get_stats(self):
        """Get global stats."""
        self.client.get(
            "/api/v1/admin/stats",
            params={"start_date": "2024-01-01", "end_date": "2024-12-31"},
            headers=self.headers,
        )
