"""JWT-based authentication for SQLAdmin.

Uses the same JWT validation as the API endpoints.
Only users with is_admin=True can access the admin panel.
"""

from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import RedirectResponse

from src.integrations.jwt_validator import JWTValidator
from src.config.logging import get_logger

logger = get_logger(__name__)


class JWTAdminAuth(AuthenticationBackend):
    """JWT-based authentication backend for SQLAdmin.

    Validates JWT tokens from cookies or Authorization header.
    Requires is_admin=True claim for access.
    """

    def __init__(self, secret_key: str):
        super().__init__(secret_key)
        self._validator = JWTValidator()

    async def login(self, request: Request) -> bool:
        """Handle login form submission.

        Accepts JWT token from form and validates it.
        If valid and is_admin=True, stores token in session.
        """
        form = await request.form()
        token = form.get("token", "")

        if not token:
            return False

        try:
            # Validate token (add Bearer prefix if not present)
            auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
            claims = self._validator.validate(auth_header)

            if not claims.is_admin:
                logger.warning(
                    "Non-admin user attempted admin panel access",
                    extra={"user_id": claims.user_id}
                )
                return False

            # Store token and user info in session
            request.session["token"] = token
            request.session["user_id"] = claims.user_id
            request.session["is_admin"] = claims.is_admin

            logger.info(
                "Admin user logged in to admin panel",
                extra={"user_id": claims.user_id}
            )
            return True

        except Exception as e:
            logger.warning(f"Admin login failed: {e}")
            return False

    async def logout(self, request: Request) -> bool:
        """Handle logout - clear session."""
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        """Check if current request is authenticated.

        Validates token stored in session is still valid.
        """
        token = request.session.get("token")

        if not token:
            return False

        try:
            auth_header = token if token.startswith("Bearer ") else f"Bearer {token}"
            claims = self._validator.validate(auth_header)

            if not claims.is_admin:
                return False

            # Update session with latest claims
            request.session["user_id"] = claims.user_id
            request.session["is_admin"] = claims.is_admin

            return True

        except Exception:
            # Token expired or invalid - clear session
            request.session.clear()
            return False
