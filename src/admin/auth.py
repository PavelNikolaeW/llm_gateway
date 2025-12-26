"""JWT-based authentication for SQLAdmin.

Authenticates via omnimap-back /api/v1/login/ endpoint.
Only users with is_staff=True can access the admin panel.
"""

from pathlib import Path

import httpx
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from src.config.logging import get_logger
from src.config.settings import settings
from src.integrations.jwt_validator import JWTValidator

logger = get_logger(__name__)

# Path to login template
TEMPLATES_DIR = Path(__file__).parent / "templates"


class JWTAdminAuth(AuthenticationBackend):
    """JWT-based authentication backend for SQLAdmin.

    Authenticates users via omnimap-back login endpoint.
    Requires is_staff=True (is_admin) claim for access.
    """

    def __init__(self, secret_key: str):
        super().__init__(secret_key)
        self._validator = JWTValidator()
        self._login_template = self._load_login_template()

    def _load_login_template(self) -> str:
        """Load login template from file."""
        template_path = TEMPLATES_DIR / "login.html"
        if template_path.exists():
            return template_path.read_text()
        # Fallback to simple template
        return """
        <html>
        <body>
            <h1>LLM Gateway Admin Login</h1>
            <form method="POST">
                <input type="text" name="username" placeholder="Username" required><br>
                <input type="password" name="password" placeholder="Password" required><br>
                <button type="submit">Login</button>
            </form>
        </body>
        </html>
        """

    async def login(self, request: Request) -> Response:
        """Handle login - show form or process credentials.

        GET: Render login form
        POST: Authenticate via omnimap-back and store JWT in session
        """
        if request.method == "GET":
            return HTMLResponse(self._login_template)

        # POST - process login
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")

        if not username or not password:
            return JSONResponse({"success": False, "error": "Username and password required"})

        try:
            # Authenticate via omnimap-back
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    settings.backend_auth_url,
                    json={"username": username, "password": password},
                )

            if response.status_code != 200:
                logger.warning(f"Backend auth failed with status {response.status_code}")
                return JSONResponse({"success": False, "error": "Invalid credentials"})

            data = response.json()
            access_token = data.get("access")
            is_staff = data.get("is_staff", False)
            user_id = data.get("user_id")

            if not access_token:
                return JSONResponse({"success": False, "error": "No token received"})

            if not is_staff:
                logger.warning(
                    "Non-admin user attempted admin panel access",
                    extra={"user_id": user_id, "username": username},
                )
                return JSONResponse({"success": False, "error": "Admin access required"})

            # Store token and user info in session
            request.session["token"] = access_token
            request.session["user_id"] = user_id
            request.session["is_admin"] = is_staff
            request.session["username"] = username

            logger.info(
                "Admin user logged in to admin panel",
                extra={"user_id": user_id, "username": username},
            )

            return JSONResponse({"success": True})

        except httpx.TimeoutException:
            logger.error("Backend auth timeout")
            return JSONResponse({"success": False, "error": "Authentication service timeout"})
        except httpx.RequestError as e:
            logger.error(f"Backend auth request error: {e}")
            return JSONResponse({"success": False, "error": "Authentication service unavailable"})
        except Exception as e:
            logger.exception(f"Admin login error: {e}")
            return JSONResponse({"success": False, "error": "Login failed"})

    async def logout(self, request: Request) -> bool:
        """Handle logout - clear session."""
        username = request.session.get("username", "unknown")
        user_id = request.session.get("user_id")
        request.session.clear()
        logger.info("Admin user logged out", extra={"user_id": user_id, "username": username})
        return True

    async def authenticate(self, request: Request) -> bool:
        """Check if current request is authenticated.

        Validates token stored in session is still valid.
        """
        token = request.session.get("token")

        if not token:
            return False

        try:
            auth_header = f"Bearer {token}"
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
