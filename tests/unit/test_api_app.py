"""Unit tests for FastAPI application setup and middleware."""
import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

from src.api.app import (
    app,
    create_app,
    get_current_user_id,
    get_is_admin,
    get_request_id,
    request_id_ctx,
    user_id_ctx,
    is_admin_ctx,
)
from src.shared.exceptions import (
    ForbiddenError,
    InsufficientTokensError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)


# Test secret for JWT
TEST_SECRET = "test-secret-key-for-api-tests"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_token():
    """Create a valid JWT token."""
    payload = {
        "user_id": 123,
        "is_admin": False,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def admin_token():
    """Create an admin JWT token."""
    payload = {
        "user_id": 456,
        "is_admin": True,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def expired_token():
    """Create an expired JWT token."""
    payload = {
        "user_id": 123,
        "is_admin": False,
        "exp": int(time.time()) - 3600,
        "iat": int(time.time()) - 7200,
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns 200 OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "components" in data

    def test_health_no_auth_required(self, client):
        """Test health endpoint doesn't require authentication."""
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_has_request_id_header(self, client):
        """Test health endpoint includes X-Request-ID header."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestJWTAuthMiddleware:
    """Tests for JWT authentication middleware."""

    def test_missing_auth_header_returns_401(self, client):
        """Test missing Authorization header returns 401."""
        response = client.get("/api/v1/dialogs")

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"
        assert "Authorization header required" in data["message"]

    def test_invalid_token_returns_401(self, client):
        """Test invalid token returns 401."""
        response = client.get(
            "/api/v1/dialogs",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401

    def test_expired_token_returns_401(self, client, expired_token):
        """Test expired token returns 401."""
        with patch("src.api.app.JWTValidator") as mock_validator_class:
            mock_validator = MagicMock()
            mock_validator.validate.side_effect = UnauthorizedError("Token has expired")
            mock_validator_class.return_value = mock_validator

            # Create new app with mocked validator
            test_app = create_app()
            test_client = TestClient(test_app)

            response = test_client.get(
                "/api/v1/dialogs",
                headers={"Authorization": f"Bearer {expired_token}"},
            )

            assert response.status_code == 401
            assert "expired" in response.json()["message"].lower()

    def test_valid_token_passes_auth(self, valid_token):
        """Test valid token passes authentication."""
        with patch("src.api.app.JWTValidator") as mock_validator_class:
            from src.integrations.jwt_validator import JWTClaims

            mock_validator = MagicMock()
            mock_validator.validate.return_value = JWTClaims(
                user_id=123,
                is_admin=False,
                exp=int(time.time()) + 3600,
                iat=int(time.time()),
                nbf=None,
                raw_claims={},
            )
            mock_validator_class.return_value = mock_validator

            test_app = create_app()

            # Add a test endpoint
            @test_app.get("/test-auth")
            async def test_auth():
                return {"user_id": user_id_ctx.get(), "is_admin": is_admin_ctx.get()}

            test_client = TestClient(test_app)

            response = test_client.get(
                "/test-auth",
                headers={"Authorization": f"Bearer {valid_token}"},
            )

            assert response.status_code == 200
            assert response.json()["user_id"] == 123
            assert response.json()["is_admin"] is False

    def test_admin_token_sets_is_admin(self, admin_token):
        """Test admin token sets is_admin to True."""
        with patch("src.api.app.JWTValidator") as mock_validator_class:
            from src.integrations.jwt_validator import JWTClaims

            mock_validator = MagicMock()
            mock_validator.validate.return_value = JWTClaims(
                user_id=456,
                is_admin=True,
                exp=int(time.time()) + 3600,
                iat=int(time.time()),
                nbf=None,
                raw_claims={},
            )
            mock_validator_class.return_value = mock_validator

            test_app = create_app()

            @test_app.get("/test-admin")
            async def test_admin():
                return {"is_admin": is_admin_ctx.get()}

            test_client = TestClient(test_app)

            response = test_client.get(
                "/test-admin",
                headers={"Authorization": f"Bearer {admin_token}"},
            )

            assert response.status_code == 200
            assert response.json()["is_admin"] is True


class TestExceptionHandlers:
    """Tests for global exception handlers."""

    def test_validation_error_returns_400(self):
        """Test ValidationError returns 400."""
        test_app = create_app()

        # Add test endpoint to public paths by using /health prefix
        @test_app.get("/health/test-validation-error")
        async def raise_validation():
            raise ValidationError("Invalid input")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-validation-error")

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert "Invalid input" in data["message"]

    def test_unauthorized_error_returns_401(self):
        """Test UnauthorizedError returns 401."""
        test_app = create_app()

        @test_app.get("/health/test-unauthorized")
        async def raise_unauthorized():
            raise UnauthorizedError("Not authenticated")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-unauthorized")

        assert response.status_code == 401
        data = response.json()
        assert data["code"] == "UNAUTHORIZED"
        assert "Not authenticated" in data["message"]

    def test_insufficient_tokens_error_returns_402(self):
        """Test InsufficientTokensError returns 402."""
        test_app = create_app()

        @test_app.get("/health/test-insufficient-tokens")
        async def raise_insufficient():
            raise InsufficientTokensError("Not enough tokens")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-insufficient-tokens")

        assert response.status_code == 402
        data = response.json()
        assert data["code"] == "INSUFFICIENT_TOKENS"
        assert "Not enough tokens" in data["message"]

    def test_forbidden_error_returns_403(self):
        """Test ForbiddenError returns 403."""
        test_app = create_app()

        @test_app.get("/health/test-forbidden")
        async def raise_forbidden():
            raise ForbiddenError("Access denied")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-forbidden")

        assert response.status_code == 403
        data = response.json()
        assert data["code"] == "FORBIDDEN"
        assert "Access denied" in data["message"]

    def test_not_found_error_returns_404(self):
        """Test NotFoundError returns 404."""
        test_app = create_app()

        @test_app.get("/health/test-not-found")
        async def raise_not_found():
            raise NotFoundError("Resource not found")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-not-found")

        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "NOT_FOUND"
        assert "Resource not found" in data["message"]

    def test_llm_error_returns_500(self):
        """Test LLMError returns 500."""
        test_app = create_app()

        @test_app.get("/health/test-llm-error")
        async def raise_llm_error():
            raise LLMError("LLM service failed")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-llm-error")

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "LLM_ERROR"
        assert "LLM service failed" in data["message"]

    def test_llm_timeout_error_returns_504(self):
        """Test LLMTimeoutError returns 504."""
        test_app = create_app()

        @test_app.get("/health/test-llm-timeout")
        async def raise_timeout():
            raise LLMTimeoutError("LLM request timed out")

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-llm-timeout")

        assert response.status_code == 504
        data = response.json()
        assert data["code"] == "LLM_TIMEOUT"
        assert "timed out" in data["message"].lower()

    def test_generic_exception_returns_500(self):
        """Test unhandled exception returns 500."""
        test_app = create_app()

        @test_app.get("/health/test-generic-error")
        async def raise_generic():
            raise RuntimeError("Unexpected error")

        # Use raise_server_exceptions=False to catch unhandled errors
        test_client = TestClient(test_app, raise_server_exceptions=False)
        response = test_client.get("/health/test-generic-error")

        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "INTERNAL_ERROR"
        assert "Internal server error" in data["message"]

    def test_error_with_details_includes_details(self):
        """Test error with details includes them in response."""
        test_app = create_app()

        @test_app.get("/health/test-error-details")
        async def raise_with_details():
            raise ValidationError(
                "Invalid input",
                details={"field": "email", "reason": "invalid format"}
            )

        test_client = TestClient(test_app)
        response = test_client.get("/health/test-error-details")

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "VALIDATION_ERROR"
        assert "details" in data
        assert data["details"]["field"] == "email"
        assert data["details"]["reason"] == "invalid format"

    def test_stack_trace_hidden_in_production(self):
        """Test stack trace is hidden when debug=False."""
        with patch("src.api.app.settings") as mock_settings:
            mock_settings.debug = False
            mock_settings.cors_origins = "*"
            mock_settings.log_level = "INFO"

            test_app = create_app()

            @test_app.get("/health/test-production-error")
            async def raise_error():
                raise RuntimeError("Production error")

            test_client = TestClient(test_app, raise_server_exceptions=False)
            response = test_client.get("/health/test-production-error")

            assert response.status_code == 500
            data = response.json()
            assert "details" not in data
            assert "traceback" not in str(data)

    def test_stack_trace_shown_in_debug_mode(self):
        """Test stack trace is shown when debug=True."""
        with patch("src.api.app.settings") as mock_settings:
            mock_settings.debug = True
            mock_settings.cors_origins = "*"
            mock_settings.log_level = "DEBUG"

            test_app = create_app()

            @test_app.get("/health/test-debug-error")
            async def raise_error():
                raise RuntimeError("Debug error")

            test_client = TestClient(test_app, raise_server_exceptions=False)
            response = test_client.get("/health/test-debug-error")

            assert response.status_code == 500
            data = response.json()
            assert "details" in data
            assert "traceback" in data["details"]
            assert "Debug error" in data["details"]["exception"]


class TestRequestContext:
    """Tests for request context middleware."""

    def test_request_id_generated(self, client):
        """Test each request gets a unique request_id."""
        response1 = client.get("/health")
        response2 = client.get("/health")

        req_id1 = response1.headers["X-Request-ID"]
        req_id2 = response2.headers["X-Request-ID"]

        assert req_id1 != req_id2
        assert len(req_id1) == 36  # UUID length

    def test_request_id_in_error_response(self, client):
        """Test request_id is included in error responses."""
        response = client.get("/api/v1/dialogs")

        assert response.status_code == 401
        assert "request_id" in response.json()


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in response."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.status_code == 200


class TestContextFunctions:
    """Tests for context helper functions."""

    def test_get_request_id(self):
        """Test get_request_id returns context value."""
        request_id_ctx.set("test-request-id")
        assert get_request_id() == "test-request-id"

    def test_get_current_user_id(self):
        """Test get_current_user_id returns context value."""
        user_id_ctx.set(123)
        assert get_current_user_id() == 123

    def test_get_is_admin(self):
        """Test get_is_admin returns context value."""
        is_admin_ctx.set(True)
        assert get_is_admin() is True


class TestAppCreation:
    """Tests for app factory function."""

    def test_create_app_returns_fastapi(self):
        """Test create_app returns FastAPI instance."""
        from fastapi import FastAPI

        test_app = create_app()
        assert isinstance(test_app, FastAPI)

    def test_create_app_has_title(self):
        """Test app has correct title."""
        test_app = create_app()
        assert test_app.title == "LLM Gateway API"

    def test_create_app_has_docs(self):
        """Test app has documentation endpoints."""
        test_app = create_app()
        assert test_app.docs_url == "/docs"
        assert test_app.redoc_url == "/redoc"
