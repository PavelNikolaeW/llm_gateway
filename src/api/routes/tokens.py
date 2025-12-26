"""Token API routes for balance and usage stats.

Endpoints:
- GET /users/me/tokens - Get current user's token balance and usage
"""

import logging

from fastapi import APIRouter

from src.api.dependencies import (
    CurrentUserId,
    DbSession,
    TokenServiceDep,
)
from src.shared.schemas import TokenStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tokens"])


@router.get(
    "/users/me/tokens",
    response_model=TokenStatsResponse,
    summary="Get current user's token balance",
    description="Get the current user's token balance, total tokens used, and limit.",
)
async def get_my_tokens(
    session: DbSession,
    user_id: CurrentUserId,
    service: TokenServiceDep,
) -> TokenStatsResponse:
    """Get current user's token balance and usage stats.

    Args:
        session: Database session
        user_id: Current user ID from JWT
        service: Token service dependency

    Returns:
        Token stats with balance, total_used, and limit
    """
    return await service.get_token_stats(session=session, user_id=user_id)
