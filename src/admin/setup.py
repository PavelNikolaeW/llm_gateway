"""SQLAdmin setup and configuration."""

from pathlib import Path

from fastapi import FastAPI
from sqladmin import Admin
from starlette.middleware.sessions import SessionMiddleware

from src.admin.auth import JWTAdminAuth
from src.admin.views import (
    AuditLogAdmin,
    DialogAdmin,
    MessageAdmin,
    ModelAdmin,
    SystemConfigAdmin,
    TokenBalanceAdmin,
    TokenTransactionAdmin,
)
from src.config.logging import get_logger
from src.config.settings import settings
from src.data.database import get_engine

# Path to custom templates
TEMPLATES_DIR = Path(__file__).parent / "templates"

logger = get_logger(__name__)


def setup_admin(app: FastAPI) -> Admin:
    """Setup SQLAdmin panel on the FastAPI app.

    Args:
        app: FastAPI application instance

    Returns:
        Configured Admin instance
    """
    # Add session middleware (required for SQLAdmin auth)
    # Use JWT_SECRET as the session secret for consistency
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.jwt_secret,
        session_cookie="admin_session",
        max_age=3600,  # 1 hour session
        same_site="lax",
        https_only=not settings.debug,
    )

    # Create admin with JWT authentication
    authentication_backend = JWTAdminAuth(secret_key=settings.jwt_secret)

    admin = Admin(
        app,
        get_engine(),
        title="LLM Gateway Admin",
        base_url="/admin",
        authentication_backend=authentication_backend,
        templates_dir=str(TEMPLATES_DIR),
    )

    # Register model views
    admin.add_view(ModelAdmin)
    admin.add_view(TokenBalanceAdmin)
    admin.add_view(TokenTransactionAdmin)
    admin.add_view(DialogAdmin)
    admin.add_view(MessageAdmin)
    admin.add_view(AuditLogAdmin)
    admin.add_view(SystemConfigAdmin)

    logger.info("Admin panel configured at /admin")

    return admin
