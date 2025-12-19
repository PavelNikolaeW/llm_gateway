"""Base repository with common CRUD operations."""
from typing import Any, Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations.

    All queries use SQLAlchemy parameterized queries for SQL injection protection.
    """

    def __init__(self, model: type[ModelType]):
        """Initialize repository with model class."""
        self.model = model

    async def get_by_id(
        self, session: AsyncSession, id_value: UUID | int | str
    ) -> ModelType | None:
        """Get entity by primary key."""
        return await session.get(self.model, id_value)

    async def get_all(
        self, session: AsyncSession, skip: int = 0, limit: int = 100
    ) -> list[ModelType]:
        """Get all entities with pagination."""
        result = await session.execute(select(self.model).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, session: AsyncSession, **kwargs: Any) -> ModelType:
        """Create new entity."""
        instance = self.model(**kwargs)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def update(
        self, session: AsyncSession, instance: ModelType, **kwargs: Any
    ) -> ModelType:
        """Update existing entity."""
        for key, value in kwargs.items():
            setattr(instance, key, value)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def delete(self, session: AsyncSession, instance: ModelType) -> None:
        """Delete entity."""
        await session.delete(instance)
        await session.flush()
