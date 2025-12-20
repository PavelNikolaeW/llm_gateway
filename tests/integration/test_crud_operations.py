"""Integration tests for CRUD operations on all models.

Tests Data Access layer with real database using test tables.
Uses test-specific tables (test_dialogs, test_token_balances, etc.)
to ensure complete isolation from production data.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import get_unique_user_id
from tests.test_models import (
    TestDialog,
    TestMessage,
    TestModel,
    TestTokenBalance,
    TestTokenTransaction,
)
from tests.test_repositories import (
    TestDialogRepository,
    TestMessageRepository,
    TestModelRepository,
    TestTokenBalanceRepository,
    TestTokenTransactionRepository,
)


class TestDialogCRUD:
    """CRUD tests for Dialog model."""

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.mark.asyncio
    async def test_create_dialog(self, session: AsyncSession, dialog_repo: TestDialogRepository):
        """Test creating a new dialog."""
        user_id = get_unique_user_id()

        dialog = await dialog_repo.create(
            session,
            user_id=user_id,
            title="Test Dialog",
            model_name="gpt-4",
            system_prompt="You are a helpful assistant.",
        )
        await session.commit()

        assert dialog.id is not None
        assert dialog.user_id == user_id
        assert dialog.title == "Test Dialog"
        assert dialog.model_name == "gpt-4"
        assert dialog.system_prompt == "You are a helpful assistant."
        assert dialog.created_at is not None

    @pytest.mark.asyncio
    async def test_read_dialog(self, session: AsyncSession, dialog_repo: TestDialogRepository):
        """Test reading a dialog by ID."""
        user_id = get_unique_user_id()

        # Create
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Read Test", model_name="gpt-4"
        )
        await session.commit()

        # Read
        fetched = await dialog_repo.get_by_id(session, dialog.id)

        assert fetched is not None
        assert fetched.id == dialog.id
        assert fetched.title == "Read Test"

    @pytest.mark.asyncio
    async def test_update_dialog(self, session: AsyncSession, dialog_repo: TestDialogRepository):
        """Test updating a dialog."""
        user_id = get_unique_user_id()

        # Create
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Original", model_name="gpt-4"
        )
        await session.commit()

        # Update
        updated = await dialog_repo.update(session, dialog, title="Updated Title")
        await session.commit()

        assert updated.title == "Updated Title"

        # Verify in DB
        fetched = await dialog_repo.get_by_id(session, dialog.id)
        assert fetched.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_dialog(self, session: AsyncSession, dialog_repo: TestDialogRepository):
        """Test deleting a dialog."""
        user_id = get_unique_user_id()

        # Create
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="To Delete", model_name="gpt-4"
        )
        await session.commit()
        dialog_id = dialog.id

        # Delete
        await dialog_repo.delete(session, dialog)
        await session.commit()

        # Verify deleted
        fetched = await dialog_repo.get_by_id(session, dialog_id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_list_dialogs_by_user(
        self, session: AsyncSession, dialog_repo: TestDialogRepository
    ):
        """Test listing dialogs by user."""
        user_id = get_unique_user_id()
        other_user_id = get_unique_user_id()

        # Create dialogs for user
        for i in range(3):
            await dialog_repo.create(
                session, user_id=user_id, title=f"Dialog {i}", model_name="gpt-4"
            )

        # Create dialog for other user
        await dialog_repo.create(
            session, user_id=other_user_id, title="Other User Dialog", model_name="gpt-4"
        )
        await session.commit()

        # List user's dialogs
        dialogs = await dialog_repo.get_by_user(session, user_id)

        assert len(dialogs) == 3
        assert all(d.user_id == user_id for d in dialogs)


class TestMessageCRUD:
    """CRUD tests for Message model."""

    @pytest.fixture
    def message_repo(self):
        return TestMessageRepository()

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.mark.asyncio
    async def test_create_message(
        self,
        session: AsyncSession,
        message_repo: TestMessageRepository,
        dialog_repo: TestDialogRepository,
    ):
        """Test creating a new message."""
        user_id = get_unique_user_id()

        # Create dialog first
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.flush()

        # Create message
        message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="user",
            content="Hello, how are you?",
        )
        await session.commit()

        assert message.id is not None
        assert message.dialog_id == dialog.id
        assert message.role == "user"
        assert message.content == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_create_assistant_message_with_tokens(
        self,
        session: AsyncSession,
        message_repo: TestMessageRepository,
        dialog_repo: TestDialogRepository,
    ):
        """Test creating assistant message with token counts."""
        user_id = get_unique_user_id()

        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.flush()

        message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content="I'm doing well, thank you!",
            prompt_tokens=15,
            completion_tokens=8,
        )
        await session.commit()

        assert message.role == "assistant"
        assert message.prompt_tokens == 15
        assert message.completion_tokens == 8

    @pytest.mark.asyncio
    async def test_get_messages_by_dialog(
        self,
        session: AsyncSession,
        message_repo: TestMessageRepository,
        dialog_repo: TestDialogRepository,
    ):
        """Test getting messages by dialog."""
        user_id = get_unique_user_id()

        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.flush()

        # Create multiple messages
        await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Message 1"
        )
        await message_repo.create(
            session, dialog_id=dialog.id, role="assistant", content="Response 1"
        )
        await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Message 2"
        )
        await session.commit()

        messages = await message_repo.get_by_dialog(session, dialog.id)

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Response 1"
        assert messages[2].content == "Message 2"


class TestTokenBalanceCRUD:
    """CRUD tests for TokenBalance model."""

    @pytest.fixture
    def balance_repo(self):
        return TestTokenBalanceRepository()

    @pytest.mark.asyncio
    async def test_create_balance(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test creating a new token balance."""
        user_id = get_unique_user_id()

        balance = await balance_repo.create(
            session, user_id=user_id, balance=1000, limit=5000
        )
        await session.commit()

        assert balance.user_id == user_id
        assert balance.balance == 1000
        assert balance.limit == 5000

    @pytest.mark.asyncio
    async def test_get_or_create_balance(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test get_or_create creates balance if not exists."""
        user_id = get_unique_user_id()

        # First call creates
        balance1 = await balance_repo.get_or_create(session, user_id, initial_balance=500)
        await session.commit()

        # Second call gets existing
        balance2 = await balance_repo.get_or_create(session, user_id, initial_balance=999)

        assert balance1.user_id == user_id
        assert balance1.balance == 500
        assert balance2.balance == 500  # Should not change

    @pytest.mark.asyncio
    async def test_add_tokens(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test adding tokens to balance."""
        user_id = get_unique_user_id()

        await balance_repo.create(session, user_id=user_id, balance=1000)
        await session.commit()

        updated = await balance_repo.add_tokens(session, user_id, 500)
        await session.commit()

        assert updated.balance == 1500

    @pytest.mark.asyncio
    async def test_deduct_tokens(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test deducting tokens from balance."""
        user_id = get_unique_user_id()

        await balance_repo.create(session, user_id=user_id, balance=1000)
        await session.commit()

        updated = await balance_repo.deduct_tokens(session, user_id, 300)
        await session.commit()

        assert updated.balance == 700

    @pytest.mark.asyncio
    async def test_deduct_tokens_insufficient(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test deducting more tokens than available raises error."""
        user_id = get_unique_user_id()

        await balance_repo.create(session, user_id=user_id, balance=100)
        await session.commit()

        with pytest.raises(ValueError, match="Insufficient tokens"):
            await balance_repo.deduct_tokens(session, user_id, 500)

    @pytest.mark.asyncio
    async def test_set_limit(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test setting token limit."""
        user_id = get_unique_user_id()

        await balance_repo.create(session, user_id=user_id, balance=1000)
        await session.commit()

        updated = await balance_repo.set_limit(session, user_id, 10000)
        await session.commit()

        assert updated.limit == 10000


class TestTokenTransactionCRUD:
    """CRUD tests for TokenTransaction model."""

    @pytest.fixture
    def transaction_repo(self):
        return TestTokenTransactionRepository()

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.fixture
    def message_repo(self):
        return TestMessageRepository()

    @pytest.mark.asyncio
    async def test_create_admin_transaction(
        self, session: AsyncSession, transaction_repo: TestTokenTransactionRepository
    ):
        """Test creating admin transaction."""
        user_id = get_unique_user_id()
        admin_id = 999

        transaction = await transaction_repo.create_admin_transaction(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id
        )
        await session.commit()

        assert transaction.user_id == user_id
        assert transaction.amount == 1000
        assert transaction.reason == "admin_top_up"
        assert transaction.admin_user_id == admin_id

    @pytest.mark.asyncio
    async def test_create_llm_usage_transaction(
        self,
        session: AsyncSession,
        transaction_repo: TestTokenTransactionRepository,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
    ):
        """Test creating LLM usage transaction."""
        user_id = get_unique_user_id()

        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.flush()

        message = await message_repo.create(
            session, dialog_id=dialog.id, role="assistant", content="Response"
        )
        await session.flush()

        transaction = await transaction_repo.create_llm_usage_transaction(
            session, user_id=user_id, amount=150, dialog_id=dialog.id, message_id=message.id
        )
        await session.commit()

        assert transaction.user_id == user_id
        assert transaction.amount == -150  # Negative for deduction
        assert transaction.reason == "llm_usage"
        assert transaction.dialog_id == dialog.id
        assert transaction.message_id == message.id

    @pytest.mark.asyncio
    async def test_get_transactions_by_user(
        self, session: AsyncSession, transaction_repo: TestTokenTransactionRepository
    ):
        """Test getting transactions by user."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Create multiple transactions
        await transaction_repo.create_admin_transaction(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id
        )
        await transaction_repo.create_admin_transaction(
            session, user_id=user_id, amount=-100, admin_user_id=admin_id, reason="admin_deduct"
        )
        await session.commit()

        transactions = await transaction_repo.get_by_user(session, user_id)

        assert len(transactions) == 2

    @pytest.mark.asyncio
    async def test_get_total_used(
        self,
        session: AsyncSession,
        transaction_repo: TestTokenTransactionRepository,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
    ):
        """Test getting total tokens used."""
        user_id = get_unique_user_id()
        admin_id = 999

        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        message = await message_repo.create(
            session, dialog_id=dialog.id, role="assistant", content="Response"
        )
        await session.flush()

        # Top up (positive) - should not count
        await transaction_repo.create_admin_transaction(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id
        )

        # Usage transactions (negative) - should count
        await transaction_repo.create_llm_usage_transaction(
            session, user_id=user_id, amount=100, dialog_id=dialog.id, message_id=message.id
        )
        await session.commit()

        total_used = await transaction_repo.get_total_used(session, user_id)

        assert total_used == 100


class TestModelCRUD:
    """CRUD tests for Model model."""

    @pytest.fixture
    def model_repo(self):
        return TestModelRepository()

    @pytest.mark.asyncio
    async def test_create_model(self, session: AsyncSession, model_repo: TestModelRepository):
        """Test creating a new model."""
        model = await model_repo.create(
            session,
            name="gpt-4-test",
            provider="openai",
            cost_per_1k_prompt_tokens=0.03,
            cost_per_1k_completion_tokens=0.06,
            context_window=128000,
            enabled=True,
        )
        await session.commit()

        assert model.name == "gpt-4-test"
        assert model.provider == "openai"
        assert float(model.cost_per_1k_prompt_tokens) == 0.03
        assert model.context_window == 128000
        assert model.enabled is True

    @pytest.mark.asyncio
    async def test_get_model_by_name(
        self, session: AsyncSession, model_repo: TestModelRepository
    ):
        """Test getting model by name."""
        await model_repo.create(
            session,
            name="claude-3-test",
            provider="anthropic",
            cost_per_1k_prompt_tokens=0.015,
            cost_per_1k_completion_tokens=0.075,
            context_window=200000,
        )
        await session.commit()

        model = await model_repo.get_by_name(session, "claude-3-test")

        assert model is not None
        assert model.name == "claude-3-test"
        assert model.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_get_enabled_models(
        self, session: AsyncSession, model_repo: TestModelRepository
    ):
        """Test getting only enabled models."""
        await model_repo.create(
            session,
            name="enabled-model",
            provider="openai",
            cost_per_1k_prompt_tokens=0.01,
            cost_per_1k_completion_tokens=0.02,
            context_window=4096,
            enabled=True,
        )
        await model_repo.create(
            session,
            name="disabled-model",
            provider="openai",
            cost_per_1k_prompt_tokens=0.01,
            cost_per_1k_completion_tokens=0.02,
            context_window=4096,
            enabled=False,
        )
        await session.commit()

        enabled_models = await model_repo.get_enabled_models(session)

        assert len(enabled_models) == 1
        assert enabled_models[0].name == "enabled-model"

    @pytest.mark.asyncio
    async def test_update_model(self, session: AsyncSession, model_repo: TestModelRepository):
        """Test updating a model."""
        model = await model_repo.create(
            session,
            name="update-test-model",
            provider="openai",
            cost_per_1k_prompt_tokens=0.01,
            cost_per_1k_completion_tokens=0.02,
            context_window=4096,
        )
        await session.commit()

        updated = await model_repo.update(
            session, model, cost_per_1k_prompt_tokens=0.02, enabled=False
        )
        await session.commit()

        assert float(updated.cost_per_1k_prompt_tokens) == 0.02
        assert updated.enabled is False
