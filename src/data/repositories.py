"""Specialized repositories for each model with custom query methods."""
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Dialog, Message, Model, TokenBalance, TokenTransaction
from src.data.repository import BaseRepository


class DialogRepository(BaseRepository[Dialog]):
    """Repository for Dialog model."""

    def __init__(self):
        super().__init__(Dialog)

    async def get_by_user(
        self, session: AsyncSession, user_id: int, skip: int = 0, limit: int = 20
    ) -> list[Dialog]:
        """Get all dialogs for a user, ordered by created_at desc."""
        result = await session.execute(
            select(Dialog)
            .where(Dialog.user_id == user_id)
            .order_by(Dialog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_id_and_user(
        self, session: AsyncSession, dialog_id: UUID, user_id: int
    ) -> Dialog | None:
        """Get dialog by ID and verify ownership."""
        result = await session.execute(
            select(Dialog).where(Dialog.id == dialog_id, Dialog.user_id == user_id)
        )
        return result.scalar_one_or_none()


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model."""

    def __init__(self):
        super().__init__(Message)

    async def get_by_dialog(
        self, session: AsyncSession, dialog_id: UUID, skip: int = 0, limit: int = 100
    ) -> list[Message]:
        """Get all messages for a dialog, ordered by created_at asc."""
        result = await session.execute(
            select(Message)
            .where(Message.dialog_id == dialog_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def create_user_message(
        self, session: AsyncSession, dialog_id: UUID, content: str
    ) -> Message:
        """Create a user message."""
        return await self.create(
            session, dialog_id=dialog_id, role="user", content=content
        )

    async def create_assistant_message(
        self,
        session: AsyncSession,
        dialog_id: UUID,
        content: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Message:
        """Create an assistant message with token counts."""
        return await self.create(
            session,
            dialog_id=dialog_id,
            role="assistant",
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class TokenBalanceRepository(BaseRepository[TokenBalance]):
    """Repository for TokenBalance model."""

    def __init__(self):
        super().__init__(TokenBalance)

    async def get_by_user(self, session: AsyncSession, user_id: int) -> TokenBalance | None:
        """Get token balance for a user."""
        return await session.get(TokenBalance, user_id)

    async def get_or_create(
        self, session: AsyncSession, user_id: int, initial_balance: int = 0
    ) -> TokenBalance:
        """Get existing balance or create new one with initial balance."""
        balance = await self.get_by_user(session, user_id)
        if balance is None:
            balance = await self.create(session, user_id=user_id, balance=initial_balance)
        return balance

    async def deduct_tokens(
        self, session: AsyncSession, user_id: int, amount: int
    ) -> TokenBalance:
        """Deduct tokens from user balance (atomic operation).

        Raises ValueError if insufficient balance.
        """
        balance = await self.get_or_create(session, user_id)
        if balance.balance < amount:
            raise ValueError(
                f"Insufficient tokens: balance={balance.balance}, required={amount}"
            )
        balance.balance -= amount
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)
        return balance

    async def add_tokens(self, session: AsyncSession, user_id: int, amount: int) -> TokenBalance:
        """Add tokens to user balance (top-up)."""
        balance = await self.get_or_create(session, user_id)
        balance.balance += amount
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)
        return balance


class TokenTransactionRepository(BaseRepository[TokenTransaction]):
    """Repository for TokenTransaction model."""

    def __init__(self):
        super().__init__(TokenTransaction)

    async def create_llm_usage_transaction(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        dialog_id: UUID,
        message_id: UUID,
    ) -> TokenTransaction:
        """Create LLM usage transaction (negative amount for deduction)."""
        return await self.create(
            session,
            user_id=user_id,
            amount=-amount,  # Negative for deduction
            reason="llm_usage",
            dialog_id=dialog_id,
            message_id=message_id,
        )

    async def create_admin_transaction(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        admin_user_id: int,
        reason: str = "admin_top_up",
    ) -> TokenTransaction:
        """Create admin transaction (top-up or deduct)."""
        return await self.create(
            session, user_id=user_id, amount=amount, reason=reason, admin_user_id=admin_user_id
        )

    async def get_by_user(
        self, session: AsyncSession, user_id: int, skip: int = 0, limit: int = 100
    ) -> list[TokenTransaction]:
        """Get transaction history for a user, ordered by created_at desc."""
        result = await session.execute(
            select(TokenTransaction)
            .where(TokenTransaction.user_id == user_id)
            .order_by(TokenTransaction.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class ModelRepository(BaseRepository[Model]):
    """Repository for Model model."""

    def __init__(self):
        super().__init__(Model)

    async def get_by_name(self, session: AsyncSession, name: str) -> Model | None:
        """Get model by name."""
        return await session.get(Model, name)

    async def get_enabled_models(self, session: AsyncSession) -> list[Model]:
        """Get all enabled models."""
        result = await session.execute(select(Model).where(Model.enabled == True))
        return list(result.scalars().all())

    async def get_by_provider(
        self, session: AsyncSession, provider: str, enabled_only: bool = True
    ) -> list[Model]:
        """Get models by provider."""
        query = select(Model).where(Model.provider == provider)
        if enabled_only:
            query = query.where(Model.enabled == True)
        result = await session.execute(query)
        return list(result.scalars().all())
