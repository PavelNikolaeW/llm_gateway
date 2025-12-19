"""Token Service - business logic for token accounting."""
import logging
from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.repositories import TokenBalanceRepository, TokenTransactionRepository
from src.shared.exceptions import ForbiddenError, InsufficientTokensError
from src.shared.schemas import (
    TokenBalanceResponse,
    TokenEvent,
    TokenStatsResponse,
    TokenTransactionResponse,
)

logger = logging.getLogger(__name__)

# Event handler type
EventHandler = Callable[[TokenEvent], None]


class TokenService:
    """Service for token accounting with business logic.

    Handles:
    - Balance checking against estimated costs
    - Token deduction with transaction logging
    - Admin top-up/deduct operations
    - Event emission for token changes
    - Race condition handling via DB transactions
    """

    def __init__(self):
        """Initialize token service with repositories."""
        self.balance_repo = TokenBalanceRepository()
        self.transaction_repo = TokenTransactionRepository()
        self._event_handlers: list[EventHandler] = []

    def register_event_handler(self, handler: EventHandler) -> None:
        """Register an event handler for token events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: TokenEvent) -> None:
        """Emit event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def check_balance(
        self, session: AsyncSession, user_id: int, estimated_cost: int
    ) -> bool:
        """Check if user has sufficient balance for estimated cost.

        Args:
            session: Database session
            user_id: User to check balance for
            estimated_cost: Estimated token cost

        Returns:
            True if balance >= estimated_cost, False otherwise

        Note:
            Emits 'balance_exhausted' event if balance is insufficient
        """
        balance = await self.balance_repo.get_or_create(session, user_id)

        has_sufficient = balance.balance >= estimated_cost

        if not has_sufficient:
            # Emit balance exhausted event
            event = TokenEvent(
                event_type="balance_exhausted",
                user_id=user_id,
                amount=estimated_cost,
                new_balance=balance.balance,
                reason="check_failed",
                timestamp=datetime.now(timezone.utc),
            )
            self._emit_event(event)
            logger.warning(
                f"Balance check failed for user {user_id}: "
                f"balance={balance.balance}, required={estimated_cost}"
            )

        return has_sufficient

    async def get_balance(
        self, session: AsyncSession, user_id: int
    ) -> TokenBalanceResponse:
        """Get current token balance for user.

        Args:
            session: Database session
            user_id: User to get balance for

        Returns:
            Token balance response
        """
        balance = await self.balance_repo.get_or_create(session, user_id)
        return TokenBalanceResponse.model_validate(balance)

    async def get_token_stats(
        self, session: AsyncSession, user_id: int
    ) -> TokenStatsResponse:
        """Get token stats for user including total usage.

        Args:
            session: Database session
            user_id: User to get stats for

        Returns:
            Token stats with balance, total_used, and limit
        """
        balance = await self.balance_repo.get_or_create(session, user_id)
        total_used = await self.transaction_repo.get_total_used(session, user_id)

        return TokenStatsResponse(
            balance=balance.balance,
            total_used=total_used,
            limit=balance.limit,
        )

    async def deduct_tokens(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        dialog_id: UUID,
        message_id: UUID,
    ) -> tuple[TokenBalanceResponse, TokenTransactionResponse]:
        """Deduct tokens for LLM usage with atomic transaction.

        Args:
            session: Database session (should be in transaction)
            user_id: User to deduct tokens from
            amount: Amount of tokens to deduct (positive number)
            dialog_id: Associated dialog ID
            message_id: Associated message ID

        Returns:
            Tuple of (updated balance, transaction record)

        Raises:
            InsufficientTokensError: If balance is insufficient

        Note:
            - Creates transaction with reason='llm_usage'
            - Emits 'tokens_deducted' event
            - Emits 'balance_exhausted' if balance goes negative
            - Cache is invalidated automatically by repository
        """
        if amount <= 0:
            raise ValueError("Deduction amount must be positive")

        # Get current balance first to check
        balance = await self.balance_repo.get_or_create(session, user_id)

        if balance.balance < amount:
            # Emit balance exhausted event
            event = TokenEvent(
                event_type="balance_exhausted",
                user_id=user_id,
                amount=amount,
                new_balance=balance.balance,
                reason="llm_usage",
                dialog_id=dialog_id,
                message_id=message_id,
                timestamp=datetime.now(timezone.utc),
            )
            self._emit_event(event)
            raise InsufficientTokensError(
                f"Insufficient tokens: balance={balance.balance}, required={amount}"
            )

        # Deduct tokens (atomic operation in repository)
        updated_balance = await self.balance_repo.deduct_tokens(session, user_id, amount)

        # Create transaction record
        transaction = await self.transaction_repo.create_llm_usage_transaction(
            session, user_id, amount, dialog_id, message_id
        )

        # Emit tokens deducted event
        event = TokenEvent(
            event_type="tokens_deducted",
            user_id=user_id,
            amount=amount,
            new_balance=updated_balance.balance,
            reason="llm_usage",
            dialog_id=dialog_id,
            message_id=message_id,
            timestamp=datetime.now(timezone.utc),
        )
        self._emit_event(event)

        logger.info(
            f"Deducted {amount} tokens from user {user_id}: "
            f"new_balance={updated_balance.balance}, dialog={dialog_id}, message={message_id}"
        )

        # Check if balance went below zero after deduction
        if updated_balance.balance < 0:
            exhaust_event = TokenEvent(
                event_type="balance_exhausted",
                user_id=user_id,
                amount=amount,
                new_balance=updated_balance.balance,
                reason="llm_usage",
                dialog_id=dialog_id,
                message_id=message_id,
                timestamp=datetime.now(timezone.utc),
            )
            self._emit_event(exhaust_event)

        return (
            TokenBalanceResponse.model_validate(updated_balance),
            TokenTransactionResponse.model_validate(transaction),
        )

    async def admin_top_up(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        admin_user_id: int,
        is_admin: bool = False,
    ) -> tuple[TokenBalanceResponse, TokenTransactionResponse]:
        """Admin top-up tokens for a user.

        Args:
            session: Database session
            user_id: User to add tokens to
            amount: Amount of tokens to add (can be negative for deduction)
            admin_user_id: Admin performing the operation
            is_admin: Whether the caller is an admin

        Returns:
            Tuple of (updated balance, transaction record)

        Raises:
            ForbiddenError: If caller is not an admin

        Note:
            - Uses reason='admin_top_up' for positive amounts
            - Uses reason='admin_deduct' for negative amounts
            - Cache is invalidated automatically by repository
        """
        if not is_admin:
            raise ForbiddenError("Only admins can perform token top-up operations")

        # Determine reason based on amount sign
        if amount >= 0:
            reason = "admin_top_up"
            # Add tokens
            updated_balance = await self.balance_repo.add_tokens(session, user_id, amount)
        else:
            reason = "admin_deduct"
            # Deduct tokens (use absolute value)
            updated_balance = await self.balance_repo.add_tokens(session, user_id, amount)

        # Create transaction record
        transaction = await self.transaction_repo.create_admin_transaction(
            session, user_id, amount, admin_user_id, reason
        )

        logger.info(
            f"Admin {admin_user_id} {reason}: {amount} tokens for user {user_id}, "
            f"new_balance={updated_balance.balance}"
        )

        # Check if balance is exhausted after admin deduction
        if updated_balance.balance < 0:
            event = TokenEvent(
                event_type="balance_exhausted",
                user_id=user_id,
                amount=abs(amount),
                new_balance=updated_balance.balance,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
            )
            self._emit_event(event)

        return (
            TokenBalanceResponse.model_validate(updated_balance),
            TokenTransactionResponse.model_validate(transaction),
        )

    async def get_transaction_history(
        self, session: AsyncSession, user_id: int, skip: int = 0, limit: int = 100
    ) -> list[TokenTransactionResponse]:
        """Get transaction history for a user.

        Args:
            session: Database session
            user_id: User to get history for
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of transaction records, ordered by created_at desc
        """
        transactions = await self.transaction_repo.get_by_user(session, user_id, skip, limit)
        return [TokenTransactionResponse.model_validate(t) for t in transactions]
