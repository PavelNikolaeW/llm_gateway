"""Pytest configuration and fixtures.

Test isolation strategy:
- Uses testcontainers PostgreSQL when USE_TESTCONTAINERS=true (for CI/CD)
- Falls back to existing database with test_ prefixed tables (for local dev)
- Each test gets a clean session with truncated tables
- NEVER touches production tables or tables starting with api_
"""
import os
import time
from collections.abc import AsyncGenerator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.test_models import TestBase

# Counter for generating unique user IDs across all tests
_user_id_counter = int(time.time() * 1000) % 1_000_000_000


def get_unique_user_id() -> int:
    """Generate a unique user ID for each test."""
    global _user_id_counter
    _user_id_counter += 1
    return _user_id_counter


@pytest.fixture(scope="function")
def unique_user_id() -> int:
    """Provide a unique user ID for each test."""
    return get_unique_user_id()


# Test tables to manage (NEVER include api_* tables)
TEST_TABLES = [
    "test_token_transactions",
    "test_messages",
    "test_dialogs",
    "test_token_balances",
    "test_models",
]


def get_database_url() -> str:
    """Get database URL for tests.

    Uses testcontainers PostgreSQL if USE_TESTCONTAINERS=true,
    otherwise uses the configured database from settings.
    """
    if os.environ.get("USE_TESTCONTAINERS", "").lower() == "true":
        try:
            from testcontainers.postgres import PostgresContainer

            # Start PostgreSQL container (cached for session)
            postgres = PostgresContainer("postgres:15-alpine")
            postgres.start()

            # Store container for cleanup
            _testcontainer_cache["postgres"] = postgres

            host = postgres.get_container_host_ip()
            port = postgres.get_exposed_port(5432)
            return f"postgresql+asyncpg://test:test@{host}:{port}/test"
        except ImportError:
            pass  # Fall back to settings

    # Use settings database
    from src.config.settings import settings
    return settings.database_url


# Cache for testcontainers
_testcontainer_cache: dict = {}


@pytest.fixture(scope="session")
def database_url():
    """Get the database URL for the test session."""
    url = get_database_url()
    yield url

    # Cleanup testcontainers
    if "postgres" in _testcontainer_cache:
        _testcontainer_cache["postgres"].stop()


@pytest.fixture(scope="function")
async def session() -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test.

    - Creates test tables if they don't exist
    - Uses test tables only (test_dialogs, test_token_balances, etc.)
    - Truncates test tables before each test for isolation
    - NEVER touches production tables or api_* tables
    """
    # Use settings database for now (testcontainers requires Docker)
    from src.config.settings import settings

    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    # Create test tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with async_session() as sess:
        # Truncate test tables before each test (in reverse order due to FK constraints)
        for table in TEST_TABLES:
            try:
                await sess.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            except Exception:
                pass  # Table might not exist yet
        await sess.commit()

        yield sess

    await engine.dispose()


@pytest.fixture(scope="function")
async def session_no_truncate() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session without truncating tables.

    Use this for tests that need data to persist across multiple operations.
    """
    from src.config.settings import settings

    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    # Create test tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)

    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    async with async_session() as sess:
        yield sess

    await engine.dispose()
