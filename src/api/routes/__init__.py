"""API routes package."""
from src.api.routes.dialogs import router as dialogs_router
from src.api.routes.messages import router as messages_router
from src.api.routes.tokens import router as tokens_router

__all__ = ["dialogs_router", "messages_router", "tokens_router"]
