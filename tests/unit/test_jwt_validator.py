"""Unit tests for JWT Validator with mocked tokens."""
import time
from unittest.mock import MagicMock, patch

import jwt
import pytest

from src.integrations.jwt_validator import (
    JWKSCache,
    JWTClaims,
    JWTValidator,
    get_jwt_validator,
    validate_jwt,
)
from src.shared.exceptions import UnauthorizedError


# Test secret for HS256
TEST_SECRET = "test-secret-key-for-jwt-validation-tests"


@pytest.fixture
def valid_token():
    """Create a valid HS256 token."""
    payload = {
        "user_id": 123,
        "is_admin": False,
        "exp": int(time.time()) + 3600,  # Expires in 1 hour
        "iat": int(time.time()),
        "nbf": int(time.time()) - 60,  # Valid since 1 minute ago
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def admin_token():
    """Create a valid HS256 token with admin rights."""
    payload = {
        "user_id": 456,
        "is_admin": True,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def expired_token():
    """Create an expired HS256 token."""
    payload = {
        "user_id": 123,
        "is_admin": False,
        "exp": int(time.time()) - 3600,  # Expired 1 hour ago
        "iat": int(time.time()) - 7200,
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def not_yet_valid_token():
    """Create a token with nbf in the future."""
    payload = {
        "user_id": 123,
        "is_admin": False,
        "exp": int(time.time()) + 7200,
        "iat": int(time.time()),
        "nbf": int(time.time()) + 3600,  # Not valid for another hour
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def missing_user_id_token():
    """Create a token without user_id."""
    payload = {
        "is_admin": False,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


@pytest.fixture
def sub_claim_token():
    """Create a token with 'sub' instead of 'user_id'."""
    payload = {
        "sub": "789",  # User ID as string in sub claim
        "is_admin": True,
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return jwt.encode(payload, TEST_SECRET, algorithm="HS256")


class TestJWTValidator:
    """Tests for JWTValidator class."""

    def test_validate_valid_token(self, valid_token):
        """Test validation of a valid token."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(valid_token)

        assert claims.user_id == 123
        assert claims.is_admin is False
        assert claims.exp > time.time()
        assert claims.iat <= time.time()

    def test_validate_admin_token(self, admin_token):
        """Test validation extracts is_admin correctly."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(admin_token)

        assert claims.user_id == 456
        assert claims.is_admin is True

    def test_validate_expired_token_raises(self, expired_token):
        """Test expired token raises UnauthorizedError."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate(expired_token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.message.lower()

    def test_validate_not_yet_valid_token_raises(self, not_yet_valid_token):
        """Test token with future nbf raises UnauthorizedError."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate(not_yet_valid_token)

        assert exc_info.value.status_code == 401
        assert "not yet valid" in exc_info.value.message.lower()

    def test_validate_invalid_signature_raises(self, valid_token):
        """Test token with wrong secret raises UnauthorizedError."""
        validator = JWTValidator(secret="wrong-secret", algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate(valid_token)

        assert exc_info.value.status_code == 401
        assert "signature" in exc_info.value.message.lower()

    def test_validate_malformed_token_raises(self):
        """Test malformed token raises UnauthorizedError."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate("not.a.valid.token")

        assert exc_info.value.status_code == 401
        assert "malformed" in exc_info.value.message.lower()

    def test_validate_empty_token_raises(self):
        """Test empty token raises UnauthorizedError."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate("")

        assert exc_info.value.status_code == 401

    def test_validate_missing_user_id_raises(self, missing_user_id_token):
        """Test token without user_id raises UnauthorizedError."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate(missing_user_id_token)

        assert exc_info.value.status_code == 401
        assert "user_id" in exc_info.value.message.lower()

    def test_validate_accepts_sub_claim(self, sub_claim_token):
        """Test validator accepts 'sub' claim as user_id."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(sub_claim_token)

        assert claims.user_id == 789
        assert claims.is_admin is True

    def test_validate_strips_bearer_prefix(self, valid_token):
        """Test validator strips 'Bearer ' prefix."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(f"Bearer {valid_token}")

        assert claims.user_id == 123

    def test_validate_no_secret_configured_raises(self):
        """Test validation without secret raises UnauthorizedError."""
        # Must patch settings.jwt_secret to prevent fallback to .env value
        with patch("src.integrations.jwt_validator.settings") as mock_settings:
            mock_settings.jwt_secret = None
            mock_settings.jwt_jwks_url = None
            mock_settings.jwt_algorithm = "HS256"
            validator = JWTValidator(secret=None, algorithm="HS256")

            with pytest.raises(UnauthorizedError) as exc_info:
                validator.validate("any.token.here")

            assert exc_info.value.status_code == 401
            assert "not configured" in exc_info.value.message.lower()

    def test_validate_is_admin_string_true(self):
        """Test is_admin as string 'true' is converted correctly."""
        payload = {
            "user_id": 123,
            "is_admin": "true",  # String instead of boolean
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = jwt.encode(payload, TEST_SECRET, algorithm="HS256")

        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(token)

        assert claims.is_admin is True

    def test_validate_is_admin_default_false(self, valid_token):
        """Test is_admin defaults to False when not present."""
        payload = {
            "user_id": 123,
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = jwt.encode(payload, TEST_SECRET, algorithm="HS256")

        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(token)

        assert claims.is_admin is False

    def test_raw_claims_contains_all_data(self, valid_token):
        """Test raw_claims contains all original claims."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")
        claims = validator.validate(valid_token)

        assert "user_id" in claims.raw_claims
        assert "is_admin" in claims.raw_claims
        assert "exp" in claims.raw_claims
        assert "iat" in claims.raw_claims


class TestJWKSCache:
    """Tests for JWKSCache class."""

    def test_cache_creates_client(self):
        """Test cache creates PyJWKClient on first access."""
        with patch("src.integrations.jwt_validator.PyJWKClient") as mock_client:
            cache = JWKSCache("https://example.com/.well-known/jwks.json")
            client = cache.get_client()

            mock_client.assert_called_once()
            assert cache._client is not None

    def test_cache_reuses_client(self):
        """Test cache reuses client within TTL."""
        with patch("src.integrations.jwt_validator.PyJWKClient") as mock_client:
            cache = JWKSCache("https://example.com/.well-known/jwks.json", ttl=3600)

            cache.get_client()
            cache.get_client()
            cache.get_client()

            # Should only create client once
            assert mock_client.call_count == 1

    def test_cache_refreshes_after_ttl(self):
        """Test cache refreshes client after TTL expires."""
        with patch("src.integrations.jwt_validator.PyJWKClient") as mock_client:
            cache = JWKSCache("https://example.com/.well-known/jwks.json", ttl=3600)

            # First access
            cache.get_client()
            assert mock_client.call_count == 1

            # Simulate TTL expiration by manipulating internal state
            cache._last_refresh = time.time() - 3601

            # Second access should refresh
            cache.get_client()
            assert mock_client.call_count == 2

    def test_cache_invalidate_forces_refresh(self):
        """Test invalidate forces refresh on next access."""
        with patch("src.integrations.jwt_validator.PyJWKClient") as mock_client:
            cache = JWKSCache("https://example.com/.well-known/jwks.json")

            cache.get_client()
            cache.invalidate()
            cache.get_client()

            assert mock_client.call_count == 2


class TestJWTClaims:
    """Tests for JWTClaims dataclass."""

    def test_claims_dataclass_creation(self):
        """Test JWTClaims dataclass creation."""
        claims = JWTClaims(
            user_id=123,
            is_admin=True,
            exp=1234567890,
            iat=1234567800,
            nbf=1234567790,
            raw_claims={"user_id": 123, "custom": "value"},
        )

        assert claims.user_id == 123
        assert claims.is_admin is True
        assert claims.exp == 1234567890
        assert claims.iat == 1234567800
        assert claims.nbf == 1234567790
        assert claims.raw_claims["custom"] == "value"

    def test_claims_nbf_optional(self):
        """Test nbf claim is optional."""
        claims = JWTClaims(
            user_id=123,
            is_admin=False,
            exp=1234567890,
            iat=1234567800,
            nbf=None,
            raw_claims={},
        )

        assert claims.nbf is None


class TestGlobalValidator:
    """Tests for global validator functions."""

    def test_get_jwt_validator_creates_singleton(self):
        """Test get_jwt_validator returns same instance."""
        # Reset global state
        import src.integrations.jwt_validator as mod
        mod._validator = None

        with patch.object(JWTValidator, "__init__", return_value=None):
            v1 = get_jwt_validator()
            v2 = get_jwt_validator()

            assert v1 is v2

    def test_validate_jwt_uses_global_validator(self, valid_token):
        """Test validate_jwt uses global validator."""
        # Reset global state
        import src.integrations.jwt_validator as mod
        mod._validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        claims = validate_jwt(valid_token)
        assert claims.user_id == 123


class TestRS256Validation:
    """Tests for RS256 validation with JWKS."""

    def test_rs256_without_jwks_url_raises(self):
        """Test RS256 without JWKS URL raises UnauthorizedError."""
        validator = JWTValidator(jwks_url=None, algorithm="RS256")

        with pytest.raises(UnauthorizedError) as exc_info:
            validator.validate("any.token.here")

        assert exc_info.value.status_code == 401
        assert "not configured" in exc_info.value.message.lower()

    def test_rs256_jwks_client_error_raises(self):
        """Test JWKS client error raises UnauthorizedError."""
        validator = JWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            algorithm="RS256",
        )

        with patch.object(validator._jwks_cache, "get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.get_signing_key_from_jwt.side_effect = (
                jwt.exceptions.PyJWKClientError("Network error")
            )
            mock_get.return_value = mock_client

            with pytest.raises(UnauthorizedError) as exc_info:
                validator.validate("any.token.here")

            assert exc_info.value.status_code == 401

    def test_refresh_jwks_invalidates_cache(self):
        """Test refresh_jwks invalidates the cache."""
        validator = JWTValidator(
            jwks_url="https://example.com/.well-known/jwks.json",
            algorithm="RS256",
        )

        with patch.object(validator._jwks_cache, "invalidate") as mock_invalidate:
            validator.refresh_jwks()
            mock_invalidate.assert_called_once()

    def test_refresh_jwks_no_cache_no_error(self):
        """Test refresh_jwks does nothing without cache."""
        validator = JWTValidator(secret=TEST_SECRET, algorithm="HS256")

        # Should not raise
        validator.refresh_jwks()
