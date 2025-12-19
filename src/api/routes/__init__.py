"""API routes package."""
from src.api.routes.dialogs import router as dialogs_router
from src.api.routes.messages import router as messages_router

__all__ = ["dialogs_router", "messages_router"]
