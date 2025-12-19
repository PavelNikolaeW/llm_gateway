"""Admin API routes for user and token management.

Endpoints:
- GET /admin/users - List all users with stats
- GET /admin/users/{user_id} - Get user details
- PATCH /admin/users/{user_id}/limits - Set token limit
- POST /admin/users/{user_id}/tokens - Top-up/deduct tokens
- GET /admin/users/{user_id}/tokens/history - Get transaction history
"""
import logging

from fastapi import APIRouter

from src.api.dependencies import (
    AdminServiceDep,
    CurrentUserId,
    DbSession,
    IsAdmin,
)
from src.shared.schemas import (
    SetLimitRequest,
    TokenBalanceResponse,
    TokenTransactionResponse,
    TopUpTokensRequest,
    UserDetailsResponse,
    UserStatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get(
    "/users",
    response_model=list[UserStatsResponse],
    summary="List all users with stats",
    description="Get a paginated list of all users with dialog count and token usage stats. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
    },
)
async def list_users(
    session: DbSession,
    is_admin: IsAdmin,
    service: AdminServiceDep,
    skip: int = 0,
    limit: int = 20,
) -> list[UserStatsResponse]:
    """List all users with stats.

    Args:
        session: Database session
        is_admin: Whether caller is admin
        service: Admin service dependency
        skip: Number of records to skip
        limit: Maximum number of records

    Returns:
        List of user stats

    Raises:
        ForbiddenError: If caller is not admin (403)
    """
    return await service.list_users(
        session=session,
        is_admin=is_admin,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/users/{user_id}",
    response_model=UserDetailsResponse,
    summary="Get user details",
    description="Get detailed information about a specific user. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
        404: {"description": "User not found"},
    },
)
async def get_user_details(
    user_id: int,
    session: DbSession,
    is_admin: IsAdmin,
    service: AdminServiceDep,
) -> UserDetailsResponse:
    """Get user details.

    Args:
        user_id: Target user ID
        session: Database session
        is_admin: Whether caller is admin
        service: Admin service dependency

    Returns:
        User details

    Raises:
        ForbiddenError: If caller is not admin (403)
        NotFoundError: If user not found (404)
    """
    return await service.get_user_details(
        session=session,
        user_id=user_id,
        is_admin=is_admin,
    )


@router.patch(
    "/users/{user_id}/limits",
    response_model=TokenBalanceResponse,
    summary="Set user token limit",
    description="Set the token limit for a user. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
        404: {"description": "User not found"},
    },
)
async def set_user_limit(
    user_id: int,
    data: SetLimitRequest,
    session: DbSession,
    admin_user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: AdminServiceDep,
) -> TokenBalanceResponse:
    """Set token limit for a user.

    Args:
        user_id: Target user ID
        data: Limit request with new limit value
        session: Database session
        admin_user_id: Admin performing the action
        is_admin: Whether caller is admin
        service: Admin service dependency

    Returns:
        Updated balance

    Raises:
        ForbiddenError: If caller is not admin (403)
        NotFoundError: If user not found (404)
    """
    return await service.set_user_limit(
        session=session,
        user_id=user_id,
        limit=data.limit,
        admin_user_id=admin_user_id,
        is_admin=is_admin,
    )


@router.post(
    "/users/{user_id}/tokens",
    response_model=TokenBalanceResponse,
    summary="Top-up or deduct user tokens",
    description="Add or remove tokens from a user's balance. Positive = add, negative = deduct. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
        404: {"description": "User not found"},
    },
)
async def top_up_tokens(
    user_id: int,
    data: TopUpTokensRequest,
    session: DbSession,
    admin_user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: AdminServiceDep,
) -> TokenBalanceResponse:
    """Top-up or deduct tokens for a user.

    Args:
        user_id: Target user ID
        data: Top-up request with amount
        session: Database session
        admin_user_id: Admin performing the action
        is_admin: Whether caller is admin
        service: Admin service dependency

    Returns:
        Updated balance

    Raises:
        ForbiddenError: If caller is not admin (403)
        NotFoundError: If user not found (404)
    """
    balance, _ = await service.top_up_tokens(
        session=session,
        user_id=user_id,
        amount=data.amount,
        admin_user_id=admin_user_id,
        is_admin=is_admin,
    )
    return balance


@router.get(
    "/users/{user_id}/tokens/history",
    response_model=list[TokenTransactionResponse],
    summary="Get user token transaction history",
    description="Get the token transaction history for a user. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
        404: {"description": "User not found"},
    },
)
async def get_token_history(
    user_id: int,
    session: DbSession,
    is_admin: IsAdmin,
    service: AdminServiceDep,
    skip: int = 0,
    limit: int = 100,
) -> list[TokenTransactionResponse]:
    """Get token transaction history for a user.

    Args:
        user_id: Target user ID
        session: Database session
        is_admin: Whether caller is admin
        service: Admin service dependency
        skip: Number of records to skip
        limit: Maximum number of records

    Returns:
        List of transactions ordered by created_at desc

    Raises:
        ForbiddenError: If caller is not admin (403)
        NotFoundError: If user not found (404)
    """
    return await service.get_token_history(
        session=session,
        user_id=user_id,
        is_admin=is_admin,
        skip=skip,
        limit=limit,
    )
