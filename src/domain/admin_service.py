"""Admin Service - business logic for admin operations."""

import logging
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.repositories import (
    DialogRepository,
    MessageRepository,
    TokenBalanceRepository,
    TokenTransactionRepository,
)
from src.shared.exceptions import ForbiddenError, NotFoundError
from src.shared.schemas import (
    AdminActionEvent,
    GlobalStatsResponse,
    ModelUsageStats,
    TokenBalanceResponse,
    TokenTransactionResponse,
    UserDetailsResponse,
    UserStatsResponse,
)

logger = logging.getLogger(__name__)

# Event handler type
EventHandler = Callable[[AdminActionEvent], None]


class AdminService:
    """Service for admin operations.

    Handles:
    - Listing users with stats
    - Viewing user details
    - Setting token limits
    - Token top-up/deduct operations
    - Transaction history retrieval
    """

    def __init__(self):
        """Initialize admin service with repositories."""
        self.balance_repo = TokenBalanceRepository()
        self.transaction_repo = TokenTransactionRepository()
        self.dialog_repo = DialogRepository()
        self.message_repo = MessageRepository()
        self._event_handlers: list[EventHandler] = []

    def register_event_handler(self, handler: EventHandler) -> None:
        """Register an event handler for admin actions."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: AdminActionEvent) -> None:
        """Emit event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def list_users(
        self,
        session: AsyncSession,
        is_admin: bool,
        skip: int = 0,
        limit: int = 20,
    ) -> list[UserStatsResponse]:
        """List all users with stats.

        Args:
            session: Database session
            is_admin: Whether caller is admin
            skip: Number of records to skip
            limit: Maximum number of records

        Returns:
            List of user stats

        Raises:
            ForbiddenError: If caller is not admin
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        balances = await self.balance_repo.list_all_users(session, skip, limit)

        result = []
        for balance in balances:
            dialog_count = await self.dialog_repo.count_by_user(session, balance.user_id)
            total_used = await self.transaction_repo.get_total_used(session, balance.user_id)

            result.append(
                UserStatsResponse(
                    user_id=balance.user_id,
                    dialog_count=dialog_count,
                    total_tokens_used=total_used,
                    balance=balance.balance,
                    limit=balance.limit,
                )
            )

        return result

    async def get_user_details(
        self,
        session: AsyncSession,
        user_id: int,
        is_admin: bool,
    ) -> UserDetailsResponse:
        """Get detailed user info.

        Args:
            session: Database session
            user_id: User to get details for
            is_admin: Whether caller is admin

        Returns:
            User details

        Raises:
            ForbiddenError: If caller is not admin
            NotFoundError: If user not found
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        balance = await self.balance_repo.get_by_user(session, user_id)
        if balance is None:
            raise NotFoundError(f"User {user_id} not found")

        dialog_count = await self.dialog_repo.count_by_user(session, user_id)
        total_used = await self.transaction_repo.get_total_used(session, user_id)
        last_activity = await self.dialog_repo.get_last_activity(session, user_id)

        return UserDetailsResponse(
            user_id=user_id,
            dialog_count=dialog_count,
            total_tokens_used=total_used,
            balance=balance.balance,
            limit=balance.limit,
            last_activity=last_activity,
        )

    async def set_user_limit(
        self,
        session: AsyncSession,
        user_id: int,
        limit: int | None,
        admin_user_id: int,
        is_admin: bool,
    ) -> TokenBalanceResponse:
        """Set token limit for a user.

        Args:
            session: Database session
            user_id: User to set limit for
            limit: Token limit (None = unlimited)
            admin_user_id: Admin performing the action
            is_admin: Whether caller is admin

        Returns:
            Updated balance

        Raises:
            ForbiddenError: If caller is not admin
            NotFoundError: If user not found
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        # Check if user exists
        balance = await self.balance_repo.get_by_user(session, user_id)
        if balance is None:
            raise NotFoundError(f"User {user_id} not found")

        # Set the limit
        updated_balance = await self.balance_repo.set_limit(session, user_id, limit)

        # Emit admin action event
        event = AdminActionEvent(
            event_type="admin_action",
            admin_user_id=admin_user_id,
            target_user_id=user_id,
            action="set_limit",
            details={"limit": limit},
            timestamp=datetime.now(timezone.utc),
        )
        self._emit_event(event)

        logger.info(f"Admin {admin_user_id} set limit for user {user_id}: limit={limit}")

        return TokenBalanceResponse.model_validate(updated_balance)

    async def top_up_tokens(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        admin_user_id: int,
        is_admin: bool,
    ) -> tuple[TokenBalanceResponse, TokenTransactionResponse]:
        """Top-up or deduct tokens for a user.

        Args:
            session: Database session
            user_id: User to modify tokens for
            amount: Amount to add (negative = deduct)
            admin_user_id: Admin performing the action
            is_admin: Whether caller is admin

        Returns:
            Tuple of (updated balance, transaction record)

        Raises:
            ForbiddenError: If caller is not admin
            NotFoundError: If user not found
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        # Check if user exists
        balance = await self.balance_repo.get_by_user(session, user_id)
        if balance is None:
            raise NotFoundError(f"User {user_id} not found")

        # Determine reason based on amount sign
        if amount >= 0:
            reason = "admin_top_up"
        else:
            reason = "admin_deduct"

        # Add tokens (can be negative)
        updated_balance = await self.balance_repo.add_tokens(session, user_id, amount)

        # Create transaction record
        transaction = await self.transaction_repo.create_admin_transaction(
            session, user_id, amount, admin_user_id, reason
        )

        # Emit admin action event
        event = AdminActionEvent(
            event_type="admin_action",
            admin_user_id=admin_user_id,
            target_user_id=user_id,
            action="top_up" if amount >= 0 else "deduct",
            details={"amount": amount, "new_balance": updated_balance.balance},
            timestamp=datetime.now(timezone.utc),
        )
        self._emit_event(event)

        logger.info(
            f"Admin {admin_user_id} {reason}: {amount} tokens for user {user_id}, "
            f"new_balance={updated_balance.balance}"
        )

        return (
            TokenBalanceResponse.model_validate(updated_balance),
            TokenTransactionResponse.model_validate(transaction),
        )

    async def get_token_history(
        self,
        session: AsyncSession,
        user_id: int,
        is_admin: bool,
        skip: int = 0,
        limit: int = 100,
    ) -> list[TokenTransactionResponse]:
        """Get token transaction history for a user.

        Args:
            session: Database session
            user_id: User to get history for
            is_admin: Whether caller is admin
            skip: Number of records to skip
            limit: Maximum number of records

        Returns:
            List of transactions ordered by created_at desc

        Raises:
            ForbiddenError: If caller is not admin
            NotFoundError: If user not found
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        # Check if user exists
        balance = await self.balance_repo.get_by_user(session, user_id)
        if balance is None:
            raise NotFoundError(f"User {user_id} not found")

        transactions = await self.transaction_repo.get_by_user(session, user_id, skip, limit)
        return [TokenTransactionResponse.model_validate(t) for t in transactions]

    async def get_global_stats(
        self,
        session: AsyncSession,
        is_admin: bool,
        start_date: datetime,
        end_date: datetime,
    ) -> GlobalStatsResponse:
        """Get global usage statistics for a date range.

        Args:
            session: Database session
            is_admin: Whether caller is admin
            start_date: Start of date range (inclusive)
            end_date: End of date range (exclusive)

        Returns:
            Global stats with total tokens, active users, top models, avg latency

        Raises:
            ForbiddenError: If caller is not admin
        """
        if not is_admin:
            raise ForbiddenError("Admin access required")

        # Get total tokens used
        total_tokens = await self.message_repo.get_total_tokens_in_range(
            session, start_date, end_date
        )

        # Get active users count
        active_users = await self.message_repo.get_active_users_in_range(
            session, start_date, end_date
        )

        # Get top models
        top_models_data = await self.dialog_repo.get_top_models_in_range(
            session, start_date, end_date, limit=5
        )
        top_models = [ModelUsageStats(model=model, usage=usage) for model, usage in top_models_data]

        # Average latency - not currently tracked in database
        # Could be added by storing latency in messages table
        avg_latency_ms = 0.0

        logger.info(
            f"Global stats for {start_date} to {end_date}: "
            f"tokens={total_tokens}, users={active_users}, models={len(top_models)}"
        )

        return GlobalStatsResponse(
            total_tokens=total_tokens,
            active_users=active_users,
            top_models=top_models,
            avg_latency_ms=avg_latency_ms,
        )
