"""API routes package."""

from src.api.routes.admin import router as admin_router
from src.api.routes.audit import router as audit_router
from src.api.routes.dialogs import router as dialogs_router
from src.api.routes.export import router as export_router
from src.api.routes.messages import router as messages_router
from src.api.routes.models import router as models_router
from src.api.routes.tokens import router as tokens_router

__all__ = [
    "admin_router",
    "audit_router",
    "dialogs_router",
    "export_router",
    "messages_router",
    "models_router",
    "tokens_router",
]
