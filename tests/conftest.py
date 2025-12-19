"""Pytest configuration and fixtures."""
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config.settings import settings


@pytest.fixture(scope="function")
async def session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test.

    Uses function scope to ensure complete isolation between tests,
    preventing 'another operation is in progress' errors with asyncpg.
    """
    # Create engine for this test
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    # Create session factory
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with async_session() as sess:
        yield sess

    # Dispose engine after test
    await engine.dispose()
