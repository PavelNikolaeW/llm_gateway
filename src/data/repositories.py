"""Specialized repositories for each model with custom query methods."""
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.cache import cache_service
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

    async def count_by_user(self, session: AsyncSession, user_id: int) -> int:
        """Count dialogs for a user."""
        result = await session.execute(
            select(func.count(Dialog.id)).where(Dialog.user_id == user_id)
        )
        return int(result.scalar() or 0)

    async def get_last_activity(self, session: AsyncSession, user_id: int) -> datetime | None:
        """Get last activity time for a user (most recent dialog update)."""
        result = await session.execute(
            select(func.max(Dialog.updated_at)).where(Dialog.user_id == user_id)
        )
        return result.scalar()

    async def get_top_models_in_range(
        self,
        session: AsyncSession,
        start_date: datetime,
        end_date: datetime,
        limit: int = 5,
    ) -> list[tuple[str, int]]:
        """Get top models by usage (message count) in date range.

        Returns list of (model_name, usage_count) tuples.
        """
        result = await session.execute(
            select(Dialog.model_name, func.count(Message.id).label("usage"))
            .join(Message, Dialog.id == Message.dialog_id)
            .where(Message.created_at >= start_date)
            .where(Message.created_at < end_date)
            .group_by(Dialog.model_name)
            .order_by(func.count(Message.id).desc())
            .limit(limit)
        )
        return [(row[0], row[1]) for row in result.all()]

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

    async def get_total_tokens_in_range(
        self, session: AsyncSession, start_date: datetime, end_date: datetime
    ) -> int:
        """Get total tokens used in a date range.

        Sums prompt_tokens + completion_tokens for all messages.
        """
        result = await session.execute(
            select(
                func.coalesce(func.sum(Message.prompt_tokens), 0)
                + func.coalesce(func.sum(Message.completion_tokens), 0)
            )
            .where(Message.created_at >= start_date)
            .where(Message.created_at < end_date)
        )
        return int(result.scalar() or 0)

    async def get_active_users_in_range(
        self, session: AsyncSession, start_date: datetime, end_date: datetime
    ) -> int:
        """Get count of unique users who sent messages in date range."""
        result = await session.execute(
            select(func.count(func.distinct(Dialog.user_id)))
            .select_from(Message)
            .join(Dialog, Message.dialog_id == Dialog.id)
            .where(Message.created_at >= start_date)
            .where(Message.created_at < end_date)
        )
        return int(result.scalar() or 0)


class TokenBalanceRepository(BaseRepository[TokenBalance]):
    """Repository for TokenBalance model."""

    def __init__(self):
        super().__init__(TokenBalance)

    async def get_by_user(self, session: AsyncSession, user_id: int) -> TokenBalance | None:
        """Get token balance for a user.

        Uses cache with 5 min TTL, falls back to database on miss.
        """
        # Try cache first
        cached = await cache_service.get_balance(user_id)
        if cached:
            # Reconstruct TokenBalance from cached data
            balance = TokenBalance(
                user_id=cached["user_id"],
                balance=cached["balance"],
                limit=cached.get("limit"),
                updated_at=datetime.fromisoformat(cached["updated_at"]),
            )
            return balance

        # Cache miss - fetch from database
        balance = await session.get(TokenBalance, user_id)

        # Cache the result for next time
        if balance:
            await cache_service.set_balance(
                user_id,
                {
                    "user_id": balance.user_id,
                    "balance": balance.balance,
                    "limit": balance.limit,
                    "updated_at": balance.updated_at.isoformat(),
                },
            )

        return balance

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
        Invalidates cache after update.
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

        # Invalidate cache after balance update
        await cache_service.invalidate_balance(user_id)

        return balance

    async def add_tokens(self, session: AsyncSession, user_id: int, amount: int) -> TokenBalance:
        """Add tokens to user balance (top-up).

        Invalidates cache after update.
        """
        balance = await self.get_or_create(session, user_id)
        balance.balance += amount
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)

        # Invalidate cache after balance update
        await cache_service.invalidate_balance(user_id)

        return balance

    async def set_limit(
        self, session: AsyncSession, user_id: int, limit: int | None
    ) -> TokenBalance:
        """Set token limit for user.

        Args:
            session: Database session
            user_id: User to set limit for
            limit: Token limit (None = unlimited)

        Returns:
            Updated balance record
        """
        balance = await self.get_or_create(session, user_id)
        balance.limit = limit
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)

        # Invalidate cache after limit update
        await cache_service.invalidate_balance(user_id)

        return balance

    async def list_all_users(
        self, session: AsyncSession, skip: int = 0, limit: int = 20
    ) -> list[TokenBalance]:
        """List all users with token balances, paginated.

        Returns users ordered by user_id.
        """
        result = await session.execute(
            select(TokenBalance)
            .order_by(TokenBalance.user_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_all_users(self, session: AsyncSession) -> int:
        """Count total users with token balances."""
        result = await session.execute(
            select(func.count(TokenBalance.user_id))
        )
        return int(result.scalar() or 0)


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

    async def get_total_used(self, session: AsyncSession, user_id: int) -> int:
        """Get total tokens used by a user (sum of all negative transactions).

        Args:
            session: Database session
            user_id: User to get total usage for

        Returns:
            Total tokens used (as positive number)
        """
        result = await session.execute(
            select(func.coalesce(func.sum(-TokenTransaction.amount), 0))
            .where(TokenTransaction.user_id == user_id)
            .where(TokenTransaction.amount < 0)
        )
        return int(result.scalar() or 0)


class ModelRepository(BaseRepository[Model]):
    """Repository for Model model."""

    def __init__(self):
        super().__init__(Model)

    async def get_by_name(self, session: AsyncSession, name: str) -> Model | None:
        """Get model by name.

        Uses cache with 1 hour TTL, falls back to database on miss.
        """
        # Try cache first
        cached = await cache_service.get_model(name)
        if cached:
            # Reconstruct Model from cached data
            model = Model(
                name=cached["name"],
                provider=cached["provider"],
                cost_per_1k_prompt_tokens=cached["cost_per_1k_prompt_tokens"],
                cost_per_1k_completion_tokens=cached["cost_per_1k_completion_tokens"],
                context_window=cached["context_window"],
                enabled=cached["enabled"],
                created_at=datetime.fromisoformat(cached["created_at"]),
                updated_at=datetime.fromisoformat(cached["updated_at"]),
            )
            return model

        # Cache miss - fetch from database
        model = await session.get(Model, name)

        # Cache the result for next time
        if model:
            await cache_service.set_model(
                name,
                {
                    "name": model.name,
                    "provider": model.provider,
                    "cost_per_1k_prompt_tokens": float(model.cost_per_1k_prompt_tokens),
                    "cost_per_1k_completion_tokens": float(model.cost_per_1k_completion_tokens),
                    "context_window": model.context_window,
                    "enabled": model.enabled,
                    "created_at": model.created_at.isoformat(),
                    "updated_at": model.updated_at.isoformat(),
                },
            )

        return model

    async def get_enabled_models(self, session: AsyncSession) -> list[Model]:
        """Get all enabled models."""
        result = await session.execute(select(Model).where(Model.enabled.is_(True)))
        return list(result.scalars().all())

    async def get_by_provider(
        self, session: AsyncSession, provider: str, enabled_only: bool = True
    ) -> list[Model]:
        """Get models by provider."""
        query = select(Model).where(Model.provider == provider)
        if enabled_only:
            query = query.where(Model.enabled.is_(True))
        result = await session.execute(query)
        return list(result.scalars().all())
