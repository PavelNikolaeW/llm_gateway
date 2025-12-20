"""Test-specific services that work with test_ prefixed tables.

These services mirror the production services but use test repositories
to ensure complete isolation from production data.
"""
import logging
from datetime import datetime, timezone
from typing import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.exceptions import ForbiddenError, InsufficientTokensError
from src.shared.schemas import (
    TokenBalanceResponse,
    TokenEvent,
    TokenStatsResponse,
    TokenTransactionResponse,
)
from tests.test_repositories import TestTokenBalanceRepository, TestTokenTransactionRepository

logger = logging.getLogger(__name__)

EventHandler = Callable[[TokenEvent], None]


class TestTokenService:
    """Test version of TokenService that uses test tables.

    Uses test_token_balances and test_token_transactions tables
    to ensure complete isolation from production data.
    """

    def __init__(self):
        self.balance_repo = TestTokenBalanceRepository()
        self.transaction_repo = TestTokenTransactionRepository()
        self._event_handlers: list[EventHandler] = []

    def register_event_handler(self, handler: EventHandler) -> None:
        self._event_handlers.append(handler)

    def _emit_event(self, event: TokenEvent) -> None:
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def check_balance(
        self, session: AsyncSession, user_id: int, estimated_cost: int
    ) -> bool:
        balance = await self.balance_repo.get_or_create(session, user_id)
        has_sufficient = balance.balance >= estimated_cost

        if not has_sufficient:
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
        balance = await self.balance_repo.get_or_create(session, user_id)
        return TokenBalanceResponse.model_validate(balance)

    async def get_token_stats(
        self, session: AsyncSession, user_id: int
    ) -> TokenStatsResponse:
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
        if amount <= 0:
            raise ValueError("Deduction amount must be positive")

        balance = await self.balance_repo.get_or_create(session, user_id)

        if balance.balance < amount:
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

        updated_balance = await self.balance_repo.deduct_tokens(session, user_id, amount)

        transaction = await self.transaction_repo.create_llm_usage_transaction(
            session, user_id, amount, dialog_id, message_id
        )

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
        if not is_admin:
            raise ForbiddenError("Only admins can perform token top-up operations")

        if amount >= 0:
            reason = "admin_top_up"
            updated_balance = await self.balance_repo.add_tokens(session, user_id, amount)
        else:
            reason = "admin_deduct"
            updated_balance = await self.balance_repo.add_tokens(session, user_id, amount)

        transaction = await self.transaction_repo.create_admin_transaction(
            session, user_id, amount, admin_user_id, reason
        )

        logger.info(
            f"Admin {admin_user_id} {reason}: {amount} tokens for user {user_id}, "
            f"new_balance={updated_balance.balance}"
        )

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
        transactions = await self.transaction_repo.get_by_user(session, user_id, skip, limit)
        return [TokenTransactionResponse.model_validate(t) for t in transactions]
