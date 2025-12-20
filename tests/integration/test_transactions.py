"""Integration tests for database transactions.

Tests atomic commit/rollback behavior for complex operations like:
- Save messages + deduct tokens (should be atomic)
- Rollback on failure
- Concurrent operations
"""
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.conftest import get_unique_user_id
from tests.test_models import (
    TestDialog,
    TestMessage,
    TestTokenBalance,
    TestTokenTransaction,
)
from tests.test_repositories import (
    TestDialogRepository,
    TestMessageRepository,
    TestTokenBalanceRepository,
    TestTokenTransactionRepository,
)
from tests.test_services import TestTokenService


class TestAtomicOperations:
    """Tests for atomic transaction behavior."""

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.fixture
    def message_repo(self):
        return TestMessageRepository()

    @pytest.fixture
    def balance_repo(self):
        return TestTokenBalanceRepository()

    @pytest.fixture
    def transaction_repo(self):
        return TestTokenTransactionRepository()

    @pytest.fixture
    def token_service(self):
        return TestTokenService()

    @pytest.mark.asyncio
    async def test_save_message_and_deduct_tokens_atomic_success(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
    ):
        """Test that saving message and deducting tokens is atomic on success."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup: create dialog and add balance
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Atomic operation: save message + deduct tokens
        message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content="Test response",
            prompt_tokens=50,
            completion_tokens=100,
        )
        await session.flush()

        balance, transaction = await token_service.deduct_tokens(
            session, user_id=user_id, amount=150, dialog_id=dialog.id, message_id=message.id
        )
        await session.commit()

        # Verify both operations succeeded
        assert message.id is not None
        assert balance.balance == 850  # 1000 - 150

        # Verify in database
        result = await session.execute(
            select(TestMessage).where(TestMessage.id == message.id)
        )
        saved_message = result.scalar_one()
        assert saved_message is not None

    @pytest.mark.asyncio
    async def test_rollback_on_insufficient_tokens(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
    ):
        """Test that message is not saved if token deduction fails."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup: create dialog with small balance
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await token_service.admin_top_up(
            session, user_id=user_id, amount=50, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Try atomic operation that should fail
        message_id = uuid.uuid4()
        try:
            message = TestMessage(
                id=message_id,
                dialog_id=dialog.id,
                role="assistant",
                content="Test response",
            )
            session.add(message)
            await session.flush()

            # This should fail (insufficient tokens)
            await token_service.deduct_tokens(
                session, user_id=user_id, amount=150, dialog_id=dialog.id, message_id=message_id
            )
            await session.commit()
        except Exception:
            await session.rollback()

        # Verify message was not saved due to rollback
        result = await session.execute(
            select(TestMessage).where(TestMessage.id == message_id)
        )
        assert result.scalar_one_or_none() is None

        # Verify balance unchanged
        balance = await token_service.get_balance(session, user_id)
        assert balance.balance == 50

    @pytest.mark.asyncio
    async def test_multiple_operations_in_single_transaction(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
    ):
        """Test multiple related operations commit together."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Multiple operations in single transaction
        # User message
        user_msg = await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Hello"
        )
        await session.flush()

        # Assistant message with token deduction
        assistant_msg = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content="Hi there!",
            prompt_tokens=10,
            completion_tokens=5,
        )
        await session.flush()

        await token_service.deduct_tokens(
            session, user_id=user_id, amount=15, dialog_id=dialog.id, message_id=assistant_msg.id
        )
        await session.commit()

        # Verify all operations committed
        messages = await message_repo.get_by_dialog(session, dialog.id)
        assert len(messages) == 2

        balance = await token_service.get_balance(session, user_id)
        assert balance.balance == 985  # 1000 - 15


class TestTransactionIsolation:
    """Tests for transaction isolation behavior."""

    @pytest.fixture
    def balance_repo(self):
        return TestTokenBalanceRepository()

    @pytest.mark.asyncio
    async def test_uncommitted_changes_not_visible(
        self, session: AsyncSession, balance_repo: TestTokenBalanceRepository
    ):
        """Test that uncommitted changes are not visible to other queries."""
        user_id = get_unique_user_id()

        # Create initial balance
        await balance_repo.create(session, user_id=user_id, balance=1000)
        await session.commit()

        # Start modifying but don't commit
        balance = await balance_repo.get_by_user(session, user_id)
        balance.balance = 500
        await session.flush()  # Write to DB but don't commit

        # Rollback
        await session.rollback()

        # Verify original value is restored
        balance = await balance_repo.get_by_user(session, user_id)
        assert balance.balance == 1000


class TestCascadeOperations:
    """Tests for cascade delete and update operations."""

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.fixture
    def message_repo(self):
        return TestMessageRepository()

    @pytest.mark.asyncio
    async def test_cascade_delete_messages_when_dialog_deleted(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
    ):
        """Test that messages are deleted when dialog is deleted (cascade)."""
        user_id = get_unique_user_id()

        # Create dialog with messages
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.flush()

        msg1 = await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Hello"
        )
        msg2 = await message_repo.create(
            session, dialog_id=dialog.id, role="assistant", content="Hi"
        )
        await session.commit()

        msg1_id = msg1.id
        msg2_id = msg2.id

        # Delete dialog
        await dialog_repo.delete(session, dialog)
        await session.commit()

        # Verify messages are also deleted
        result = await session.execute(
            select(TestMessage).where(TestMessage.id.in_([msg1_id, msg2_id]))
        )
        messages = result.scalars().all()
        assert len(messages) == 0


class TestTransactionHistory:
    """Tests for transaction history tracking."""

    @pytest.fixture
    def token_service(self):
        return TestTokenService()

    @pytest.fixture
    def dialog_repo(self):
        return TestDialogRepository()

    @pytest.fixture
    def message_repo(self):
        return TestMessageRepository()

    @pytest.mark.asyncio
    async def test_all_token_changes_recorded(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
    ):
        """Test that all token changes create transaction records."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Admin top-up
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Create dialog and message for deduction
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        message = await message_repo.create(
            session, dialog_id=dialog.id, role="assistant", content="Response"
        )
        await session.flush()

        # LLM usage deduction
        await token_service.deduct_tokens(
            session, user_id=user_id, amount=100, dialog_id=dialog.id, message_id=message.id
        )
        await session.commit()

        # Admin deduction
        await token_service.admin_top_up(
            session, user_id=user_id, amount=-50, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Verify all transactions recorded
        history = await token_service.get_transaction_history(session, user_id)

        assert len(history) == 3
        reasons = [t.reason for t in history]
        assert "admin_top_up" in reasons
        assert "llm_usage" in reasons
        assert "admin_deduct" in reasons

        # Verify amounts
        top_up = next(t for t in history if t.reason == "admin_top_up")
        usage = next(t for t in history if t.reason == "llm_usage")
        deduct = next(t for t in history if t.reason == "admin_deduct")

        assert top_up.amount == 1000
        assert usage.amount == -100
        assert deduct.amount == -50
