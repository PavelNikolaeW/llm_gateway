"""Test-specific repositories that work with test_ prefixed tables.

These repositories mirror the production repositories but use test models
to ensure complete isolation from production data.
"""
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_models import (
    TestBase,
    TestDialog,
    TestMessage,
    TestModel,
    TestTokenBalance,
    TestTokenTransaction,
)

TestModelType = TypeVar("TestModelType", bound=TestBase)


class TestBaseRepository(Generic[TestModelType]):
    """Base repository for test models."""

    def __init__(self, model: type[TestModelType]):
        self.model = model

    async def get_by_id(
        self, session: AsyncSession, id_value: UUID | int | str
    ) -> TestModelType | None:
        return await session.get(self.model, id_value)

    async def get_all(
        self, session: AsyncSession, skip: int = 0, limit: int = 100
    ) -> list[TestModelType]:
        result = await session.execute(select(self.model).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, session: AsyncSession, **kwargs: Any) -> TestModelType:
        instance = self.model(**kwargs)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update(
        self, session: AsyncSession, instance: TestModelType, **kwargs: Any
    ) -> TestModelType:
        for key, value in kwargs.items():
            setattr(instance, key, value)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def delete(self, session: AsyncSession, instance: TestModelType) -> None:
        await session.delete(instance)
        await session.flush()


class TestDialogRepository(TestBaseRepository[TestDialog]):
    """Repository for TestDialog model."""

    def __init__(self):
        super().__init__(TestDialog)

    async def get_by_user(
        self, session: AsyncSession, user_id: int, skip: int = 0, limit: int = 20
    ) -> list[TestDialog]:
        result = await session.execute(
            select(TestDialog)
            .where(TestDialog.user_id == user_id)
            .order_by(TestDialog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class TestMessageRepository(TestBaseRepository[TestMessage]):
    """Repository for TestMessage model."""

    def __init__(self):
        super().__init__(TestMessage)

    async def get_by_dialog(
        self, session: AsyncSession, dialog_id: UUID, skip: int = 0, limit: int = 100
    ) -> list[TestMessage]:
        result = await session.execute(
            select(TestMessage)
            .where(TestMessage.dialog_id == dialog_id)
            .order_by(TestMessage.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())


class TestTokenBalanceRepository(TestBaseRepository[TestTokenBalance]):
    """Repository for TestTokenBalance model."""

    def __init__(self):
        super().__init__(TestTokenBalance)

    async def get_by_user(self, session: AsyncSession, user_id: int) -> TestTokenBalance | None:
        return await session.get(TestTokenBalance, user_id)

    async def get_or_create(
        self, session: AsyncSession, user_id: int, initial_balance: int = 0
    ) -> TestTokenBalance:
        balance = await self.get_by_user(session, user_id)
        if balance is None:
            balance = await self.create(session, user_id=user_id, balance=initial_balance)
        return balance

    async def deduct_tokens(
        self, session: AsyncSession, user_id: int, amount: int
    ) -> TestTokenBalance:
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

    async def add_tokens(self, session: AsyncSession, user_id: int, amount: int) -> TestTokenBalance:
        balance = await self.get_or_create(session, user_id)
        balance.balance += amount
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)
        return balance

    async def set_limit(
        self, session: AsyncSession, user_id: int, limit: int | None
    ) -> TestTokenBalance:
        balance = await self.get_or_create(session, user_id)
        balance.limit = limit
        balance.updated_at = datetime.utcnow()
        await session.flush()
        await session.refresh(balance)
        return balance

    async def list_all_users(
        self, session: AsyncSession, skip: int = 0, limit: int = 20
    ) -> list[TestTokenBalance]:
        result = await session.execute(
            select(TestTokenBalance)
            .order_by(TestTokenBalance.user_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_all_users(self, session: AsyncSession) -> int:
        result = await session.execute(
            select(func.count(TestTokenBalance.user_id))
        )
        return int(result.scalar() or 0)


class TestTokenTransactionRepository(TestBaseRepository[TestTokenTransaction]):
    """Repository for TestTokenTransaction model."""

    def __init__(self):
        super().__init__(TestTokenTransaction)

    async def create_llm_usage_transaction(
        self,
        session: AsyncSession,
        user_id: int,
        amount: int,
        dialog_id: UUID,
        message_id: UUID,
    ) -> TestTokenTransaction:
        return await self.create(
            session,
            user_id=user_id,
            amount=-amount,
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
    ) -> TestTokenTransaction:
        return await self.create(
            session, user_id=user_id, amount=amount, reason=reason, admin_user_id=admin_user_id
        )

    async def get_by_user(
        self, session: AsyncSession, user_id: int, skip: int = 0, limit: int = 100
    ) -> list[TestTokenTransaction]:
        result = await session.execute(
            select(TestTokenTransaction)
            .where(TestTokenTransaction.user_id == user_id)
            .order_by(TestTokenTransaction.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_total_used(self, session: AsyncSession, user_id: int) -> int:
        result = await session.execute(
            select(func.coalesce(func.sum(-TestTokenTransaction.amount), 0))
            .where(TestTokenTransaction.user_id == user_id)
            .where(TestTokenTransaction.amount < 0)
        )
        return int(result.scalar() or 0)


class TestModelRepository(TestBaseRepository[TestModel]):
    """Repository for TestModel model."""

    def __init__(self):
        super().__init__(TestModel)

    async def get_by_name(self, session: AsyncSession, name: str) -> TestModel | None:
        return await session.get(TestModel, name)

    async def get_enabled_models(self, session: AsyncSession) -> list[TestModel]:
        result = await session.execute(select(TestModel).where(TestModel.enabled == True))
        return list(result.scalars().all())
