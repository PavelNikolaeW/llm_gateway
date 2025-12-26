"""Database configuration and session management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config.settings import settings

# Global engine instance (lazily initialized)
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the global async database engine.

    Connection pool configured with max 20 connections.
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=10,  # Total max 20 connections
            pool_pre_ping=True,
            echo=settings.debug,
        )
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get or create the global session maker."""
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_maker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting a database session.

    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(session: AsyncSession = Depends(get_session)):
            ...
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_transaction_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for explicit transaction control.

    Usage:
        async with get_transaction_session() as session:
            # Do multiple operations
            await repo1.create(session, data1)
            await repo2.update(session, data2)
            # Automatically commits on success, rolls back on exception
    """
    session_maker = get_session_maker()
    async with session_maker() as session:
        async with session.begin():
            yield session


async def close_engine() -> None:
    """Close the database engine and cleanup connections."""
    global _engine, _session_maker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None
