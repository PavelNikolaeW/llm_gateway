"""FastAPI dependencies for dependency injection.

Provides:
- Database session dependency
- Current user dependency (from JWT)
- Service dependencies
"""
from typing import Annotated, AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.database import get_session_maker
from src.domain.dialog_service import DialogService
from src.domain.message_service import MessageService
from src.domain.model_registry import model_registry
from src.domain.token_service import TokenService
from src.integrations.jwt_validator import JWTClaims
from src.integrations.llm_factory import get_llm_provider


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency.

    Yields an async database session that is automatically closed
    after the request completes.
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# Type alias for database session dependency
DbSession = Annotated[AsyncSession, Depends(get_db_session)]


def get_current_user(request: Request) -> JWTClaims:
    """Get current user from JWT claims stored in request state.

    Args:
        request: FastAPI request object

    Returns:
        JWT claims with user_id, is_admin, etc.

    Raises:
        AttributeError: If JWT auth middleware hasn't run (should not happen)
    """
    return request.state.jwt_claims


def get_current_user_id(request: Request) -> int:
    """Get current user ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        User ID from JWT claims
    """
    return request.state.user_id


def get_is_admin(request: Request) -> bool:
    """Get admin status from request state.

    Args:
        request: FastAPI request object

    Returns:
        True if user is admin
    """
    return request.state.is_admin


# Type aliases for user dependencies
CurrentUser = Annotated[JWTClaims, Depends(get_current_user)]
CurrentUserId = Annotated[int, Depends(get_current_user_id)]
IsAdmin = Annotated[bool, Depends(get_is_admin)]


def get_dialog_service() -> DialogService:
    """Get DialogService instance.

    Uses global model_registry for model validation.
    """
    return DialogService(model_registry)


def get_token_service() -> TokenService:
    """Get TokenService instance."""
    return TokenService()


def get_message_service() -> MessageService:
    """Get MessageService instance.

    Creates service with TokenService and LLM provider.
    Uses OpenAI provider by default (can be changed based on model).
    """
    token_service = TokenService()
    # Default to OpenAI provider - in production, select based on model
    llm_provider = get_llm_provider("openai")
    return MessageService(token_service, llm_provider)


# Type aliases for service dependencies
DialogServiceDep = Annotated[DialogService, Depends(get_dialog_service)]
TokenServiceDep = Annotated[TokenService, Depends(get_token_service)]
MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]
