"""Pytest configuration and fixtures."""
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from src.config.settings import settings


@pytest.fixture(scope="session")
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine."""
    test_engine = create_async_engine(settings.database_url, echo=False)
    yield test_engine
    await test_engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for a test."""
    async with AsyncSession(engine, expire_on_commit=False) as sess:
        yield sess
