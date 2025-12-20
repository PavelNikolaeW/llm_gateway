"""E2E test configuration with FastAPI TestClient.

Provides:
- FastAPI TestClient with test database
- Mocked LLM providers
- Test data factories (users, dialogs, models)
- JWT token generation for auth testing
"""
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import create_app
from src.config.settings import settings
from src.integrations.jwt_validator import JWTValidator, JWTClaims
from tests.test_models import TestBase, TestDialog, TestMessage, TestTokenBalance, TestModel


# Test tables to manage
TEST_TABLES = [
    "test_token_transactions",
    "test_messages",
    "test_dialogs",
    "test_token_balances",
    "test_models",
]


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_user_token(
        user_id: int,
        is_admin: bool = False,
        exp_hours: int = 24,
    ) -> str:
        """Create a JWT token for testing."""
        payload = {
            "sub": str(user_id),
            "user_id": user_id,
            "is_admin": is_admin,
            "exp": datetime.now(timezone.utc) + timedelta(hours=exp_hours),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, settings.django_secret_key, algorithm="HS256")

    @staticmethod
    def create_expired_token(user_id: int) -> str:
        """Create an expired JWT token for testing."""
        payload = {
            "sub": str(user_id),
            "user_id": user_id,
            "is_admin": False,
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        }
        return jwt.encode(payload, settings.django_secret_key, algorithm="HS256")

    @staticmethod
    def create_invalid_token() -> str:
        """Create an invalid JWT token."""
        return "invalid.jwt.token"


class MockLLMProvider:
    """Mock LLM provider for E2E testing."""

    def __init__(
        self,
        response: str = "Hello! I'm a helpful AI assistant.",
        prompt_tokens: int = 50,
        completion_tokens: int = 100,
    ):
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_count = 0

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate mock response."""
        self.call_count += 1
        return self.response, self.prompt_tokens, self.completion_tokens

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        """Generate streaming mock response."""
        self.call_count += 1
        words = self.response.split()
        for i, word in enumerate(words):
            is_final = i == len(words) - 1
            if is_final:
                yield word, True, self.prompt_tokens, self.completion_tokens
            else:
                yield word + " ", False, None, None


@pytest.fixture(scope="function")
def test_data_factory():
    """Provide test data factory."""
    return TestDataFactory()


@pytest.fixture(scope="function")
def mock_llm():
    """Provide mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture(scope="function")
def regular_user_id():
    """Regular test user ID."""
    return 100001


@pytest.fixture(scope="function")
def admin_user_id():
    """Admin test user ID."""
    return 999999


@pytest.fixture(scope="function")
def regular_user_token(regular_user_id: int, test_data_factory: TestDataFactory):
    """JWT token for regular user."""
    return test_data_factory.create_user_token(regular_user_id, is_admin=False)


@pytest.fixture(scope="function")
def admin_user_token(admin_user_id: int, test_data_factory: TestDataFactory):
    """JWT token for admin user."""
    return test_data_factory.create_user_token(admin_user_id, is_admin=True)


@pytest.fixture(scope="function")
def expired_token(regular_user_id: int, test_data_factory: TestDataFactory):
    """Expired JWT token."""
    return test_data_factory.create_expired_token(regular_user_id)


@pytest.fixture(scope="function")
def invalid_token(test_data_factory: TestDataFactory):
    """Invalid JWT token."""
    return test_data_factory.create_invalid_token()


@pytest.fixture(scope="function")
async def test_session() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )

    # Create test tables
    async with engine.begin() as conn:
        await conn.run_sync(TestBase.metadata.create_all)

    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        # Truncate test tables
        for table in TEST_TABLES:
            try:
                await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            except Exception:
                pass
        await session.commit()

        yield session

    await engine.dispose()


class MockModel:
    """Mock model for testing."""

    def __init__(self, name: str, provider: str = "openai"):
        self.name = name
        self.provider = provider
        self.display_name = name
        self.context_window = 128000
        self.cost_per_1k_input = 0.01
        self.cost_per_1k_output = 0.03
        self.enabled = True
        self.agent_config = {}


class MockModelRegistry:
    """Mock model registry that always returns valid models."""

    def __init__(self):
        self._models = {
            "gpt-4": MockModel("gpt-4", "openai"),
            "gpt-3.5-turbo": MockModel("gpt-3.5-turbo", "openai"),
            "claude-3-opus-20240229": MockModel("claude-3-opus-20240229", "anthropic"),
        }
        self._loaded = True

    async def load_models(self, session) -> None:
        pass

    def get_model(self, name: str):
        return self._models.get(name)

    def get_all_models(self):
        return list(self._models.values())

    def is_loaded(self) -> bool:
        return True

    def model_exists(self, name: str) -> bool:
        return name in self._models

    def validate_model(self, name: str) -> None:
        from src.shared.exceptions import ValidationError
        if not self._models:
            raise ValidationError("No models available in registry")
        if name not in self._models:
            model_names = ", ".join(self._models.keys())
            raise ValidationError(f"Invalid model_name '{name}'. Available models: {model_names}")


class TestJWTValidator:
    """JWT Validator configured for testing with django_secret_key."""

    def __init__(self, *args, **kwargs):
        """Initialize with django_secret_key regardless of args."""
        self._secret = settings.django_secret_key
        self._algorithm = "HS256"

    def validate(self, token: str) -> JWTClaims:
        """Validate JWT token and extract claims."""
        import jwt as pyjwt
        from src.shared.exceptions import UnauthorizedError

        # Clean token (remove "Bearer " prefix if present)
        if token.startswith("Bearer "):
            token = token[7:]

        try:
            claims = pyjwt.decode(
                token,
                self._secret,
                algorithms=["HS256"],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "require": ["exp", "iat"],
                },
            )

            # Extract user_id
            user_id = claims.get("user_id") or claims.get("sub")
            if user_id is None:
                raise UnauthorizedError("Token missing user_id claim")
            user_id = int(user_id)

            # Extract is_admin
            is_admin = claims.get("is_admin", False)

            return JWTClaims(
                user_id=user_id,
                is_admin=bool(is_admin),
                exp=claims.get("exp", 0),
                iat=claims.get("iat", 0),
                nbf=claims.get("nbf"),
                raw_claims=claims,
            )
        except pyjwt.ExpiredSignatureError:
            raise UnauthorizedError("Token has expired")
        except pyjwt.InvalidTokenError as e:
            raise UnauthorizedError(f"Invalid token: {e}")
        except Exception as e:
            raise UnauthorizedError(f"Token validation failed: {e}")


@pytest.fixture(scope="function")
async def client(mock_llm: MockLLMProvider) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with mocked dependencies."""
    # Create mock registry instance
    mock_registry = MockModelRegistry()

    # Patch JWTValidator, LLM provider factory, and model_registry global instance
    with patch("src.api.app.JWTValidator", TestJWTValidator), \
         patch("src.integrations.llm_factory.LLMProviderFactory.get_provider") as mock_factory, \
         patch("src.api.dependencies.model_registry", mock_registry), \
         patch("src.domain.model_registry.model_registry", mock_registry):
        mock_factory.return_value = mock_llm

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture(scope="function")
def auth_headers(regular_user_token: str) -> dict[str, str]:
    """Authorization headers for regular user."""
    return {"Authorization": f"Bearer {regular_user_token}"}


@pytest.fixture(scope="function")
def admin_headers(admin_user_token: str) -> dict[str, str]:
    """Authorization headers for admin user."""
    return {"Authorization": f"Bearer {admin_user_token}"}
