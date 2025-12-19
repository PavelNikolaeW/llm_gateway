"""JWT Validator - Signature Verification & Claims Extraction.

Implements JWT validation with support for:
- HS256 (symmetric key from JWT_SECRET)
- RS256 (asymmetric key from JWKS endpoint)

Security:
    - Secret keys from environment variables, never logged
    - JWKS keys cached for 1 hour to reduce load on auth server
"""
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAudienceError,
    InvalidSignatureError,
    InvalidTokenError,
)

from src.config.settings import settings
from src.shared.exceptions import UnauthorizedError

logger = logging.getLogger(__name__)

# JWKS cache TTL in seconds (1 hour)
JWKS_CACHE_TTL = 3600


@dataclass
class JWTClaims:
    """Extracted claims from a validated JWT token."""

    user_id: int
    is_admin: bool
    exp: int  # Expiration time (Unix timestamp)
    iat: int  # Issued at time (Unix timestamp)
    nbf: int | None  # Not before time (Unix timestamp, optional)
    raw_claims: dict[str, Any]  # All claims for extension


class JWKSCache:
    """Cache for JWKS keys with TTL.

    Caches the PyJWKClient to avoid refetching keys on every request.
    Keys are refreshed automatically when TTL expires.
    """

    def __init__(self, jwks_url: str, ttl: int = JWKS_CACHE_TTL):
        """Initialize JWKS cache.

        Args:
            jwks_url: URL to fetch JWKS from
            ttl: Cache TTL in seconds (default 1 hour)
        """
        self._jwks_url = jwks_url
        self._ttl = ttl
        self._client: PyJWKClient | None = None
        self._last_refresh: float = 0

    def get_client(self) -> PyJWKClient:
        """Get JWKS client, refreshing if cache expired.

        Returns:
            PyJWKClient instance with cached keys
        """
        now = time.time()

        if self._client is None or (now - self._last_refresh) > self._ttl:
            logger.debug(f"Refreshing JWKS cache from {self._jwks_url}")
            self._client = PyJWKClient(self._jwks_url, cache_keys=True)
            self._last_refresh = now

        return self._client

    def invalidate(self) -> None:
        """Invalidate cache to force refresh on next access."""
        self._client = None
        self._last_refresh = 0


class JWTValidator:
    """JWT token validator supporting HS256 and RS256 algorithms.

    Features:
    - HS256: Uses shared secret from JWT_SECRET env var
    - RS256: Uses public keys from JWKS endpoint
    - Validates exp, iat, nbf claims
    - Extracts user_id and is_admin claims
    - Caches JWKS keys for 1 hour

    Usage:
        validator = JWTValidator()
        claims = validator.validate(token)
        print(claims.user_id, claims.is_admin)
    """

    def __init__(
        self,
        secret: str | None = None,
        jwks_url: str | None = None,
        algorithm: str | None = None,
    ):
        """Initialize JWT validator.

        Args:
            secret: Secret key for HS256 (defaults to settings.jwt_secret)
            jwks_url: JWKS URL for RS256 (defaults to settings.jwt_jwks_url)
            algorithm: Algorithm to use (defaults to settings.jwt_algorithm)
        """
        self._secret = secret or settings.jwt_secret
        self._jwks_url = jwks_url or settings.jwt_jwks_url
        self._algorithm = algorithm or settings.jwt_algorithm

        # Initialize JWKS cache if using RS256
        self._jwks_cache: JWKSCache | None = None
        if self._algorithm == "RS256" and self._jwks_url:
            self._jwks_cache = JWKSCache(self._jwks_url)

        # Validate configuration
        if self._algorithm == "HS256" and not self._secret:
            logger.warning("JWT_SECRET not configured for HS256 validation")
        elif self._algorithm == "RS256" and not self._jwks_url:
            logger.warning("JWT_JWKS_URL not configured for RS256 validation")

    def validate(self, token: str) -> JWTClaims:
        """Validate JWT token and extract claims.

        Args:
            token: JWT token string (without "Bearer " prefix)

        Returns:
            JWTClaims with extracted user_id, is_admin, exp, iat, nbf

        Raises:
            UnauthorizedError: If token is invalid, expired, or malformed (401)
        """
        # Clean token (remove "Bearer " prefix if present)
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            if self._algorithm == "RS256":
                claims = self._validate_rs256(token)
            else:  # Default to HS256
                claims = self._validate_hs256(token)

            return self._extract_claims(claims)

        except ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise UnauthorizedError("Token has expired")

        except ImmatureSignatureError:
            logger.warning("JWT token not yet valid (nbf)")
            raise UnauthorizedError("Token is not yet valid")

        except InvalidSignatureError:
            logger.warning("JWT token has invalid signature")
            raise UnauthorizedError("Invalid token signature")

        except InvalidAudienceError:
            logger.warning("JWT token has invalid audience")
            raise UnauthorizedError("Invalid token audience")

        except DecodeError as e:
            logger.warning(f"JWT token malformed: {e}")
            raise UnauthorizedError("Malformed token")

        except InvalidTokenError as e:
            logger.warning(f"JWT token invalid: {e}")
            raise UnauthorizedError("Invalid token")

        except UnauthorizedError:
            # Re-raise UnauthorizedError from inner methods
            raise

        except Exception as e:
            logger.error(f"JWT validation error: {e}")
            raise UnauthorizedError("Token validation failed")

    def _validate_hs256(self, token: str) -> dict[str, Any]:
        """Validate token with HS256 algorithm.

        Args:
            token: JWT token string

        Returns:
            Decoded claims dict

        Raises:
            UnauthorizedError: If secret not configured
            jwt exceptions: For invalid tokens
        """
        if not self._secret:
            raise UnauthorizedError("JWT validation not configured")

        return jwt.decode(
            token,
            self._secret,
            algorithms=["HS256"],
            options={
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": True,
                "require": ["exp", "iat"],
            },
        )

    def _validate_rs256(self, token: str) -> dict[str, Any]:
        """Validate token with RS256 algorithm using JWKS.

        Args:
            token: JWT token string

        Returns:
            Decoded claims dict

        Raises:
            UnauthorizedError: If JWKS not configured
            jwt exceptions: For invalid tokens
        """
        if not self._jwks_cache or not self._jwks_url:
            raise UnauthorizedError("JWT validation not configured")

        try:
            # Get signing key from JWKS
            jwks_client = self._jwks_cache.get_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "require": ["exp", "iat"],
                },
            )

        except jwt.exceptions.PyJWKClientError as e:
            logger.error(f"JWKS client error: {e}")
            raise UnauthorizedError("Unable to validate token")

    def _extract_claims(self, raw_claims: dict[str, Any]) -> JWTClaims:
        """Extract standard claims from decoded token.

        Args:
            raw_claims: Decoded JWT claims dict

        Returns:
            JWTClaims with extracted values

        Raises:
            UnauthorizedError: If required claims missing
        """
        # Extract user_id (may be in different fields)
        user_id = raw_claims.get("user_id") or raw_claims.get("sub")
        if user_id is None:
            raise UnauthorizedError("Token missing user_id claim")

        # Convert user_id to int if needed
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            raise UnauthorizedError("Invalid user_id in token")

        # Extract is_admin (default False if not present)
        is_admin = raw_claims.get("is_admin", False)
        if isinstance(is_admin, str):
            is_admin = is_admin.lower() in ("true", "1", "yes")

        # Extract exp and iat (required by validation)
        exp = raw_claims.get("exp", 0)
        iat = raw_claims.get("iat", 0)
        nbf = raw_claims.get("nbf")

        return JWTClaims(
            user_id=user_id,
            is_admin=bool(is_admin),
            exp=exp,
            iat=iat,
            nbf=nbf,
            raw_claims=raw_claims,
        )

    def refresh_jwks(self) -> None:
        """Force refresh of JWKS cache.

        Call this if keys have been rotated at the auth server.
        """
        if self._jwks_cache:
            self._jwks_cache.invalidate()
            logger.info("JWKS cache invalidated")


# Global validator instance for convenience
_validator: JWTValidator | None = None


def get_jwt_validator() -> JWTValidator:
    """Get global JWT validator instance.

    Creates validator on first call, reuses on subsequent calls.

    Returns:
        JWTValidator instance
    """
    global _validator
    if _validator is None:
        _validator = JWTValidator()
    return _validator


def validate_jwt(token: str) -> JWTClaims:
    """Validate JWT token using global validator.

    Convenience function for simple validation.

    Args:
        token: JWT token string

    Returns:
        JWTClaims with extracted claims

    Raises:
        UnauthorizedError: If token is invalid (401)
    """
    return get_jwt_validator().validate(token)
