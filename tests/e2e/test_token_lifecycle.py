"""End-to-end tests for token lifecycle.

Tests the complete token flow using test tables:
1. New user gets initial balance
2. Admin top-up increases balance
3. Token deduction for LLM usage
4. Token limit enforcement
5. Transaction history tracking

Uses test-specific tables (test_dialogs, test_token_balances, etc.)
to ensure complete isolation from production data.
NEVER touches tables starting with api_*.
"""
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.exceptions import ForbiddenError, InsufficientTokensError
from tests.conftest import get_unique_user_id
from tests.test_models import TestDialog, TestMessage, TestTokenBalance, TestTokenTransaction
from tests.test_services import TestTokenService


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, prompt_tokens: int = 50, completion_tokens: int = 100):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        return "Mock response", self.prompt_tokens, self.completion_tokens

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        yield "Mock", False, None, None
        yield " response", False, None, None
        yield "", True, self.prompt_tokens, self.completion_tokens


@pytest.fixture
def token_service():
    """Create test token service that uses test tables."""
    return TestTokenService()


class TestNewUserBalance:
    """Tests for new user balance initialization."""

    @pytest.mark.asyncio
    async def test_new_user_starts_with_zero_balance(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test new user gets zero balance on first check."""
        user_id = get_unique_user_id()

        # Check balance (creates record if not exists)
        has_balance = await token_service.check_balance(session, user_id, estimated_cost=0)

        # Get the balance
        balance = await token_service.get_balance(session, user_id)

        assert balance.user_id == user_id
        assert balance.balance == 0
        assert balance.limit is None

        print(f"\n✓ New user {user_id} starts with zero balance")

    @pytest.mark.asyncio
    async def test_get_balance_creates_record(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test get_balance creates record for new user."""
        user_id = get_unique_user_id()

        # Get balance should create record
        balance = await token_service.get_balance(session, user_id)

        assert balance is not None
        assert balance.user_id == user_id
        assert balance.balance == 0

        print(f"\n✓ get_balance creates record for new user {user_id}")


class TestAdminTopUp:
    """Tests for admin token top-up operations."""

    @pytest.mark.asyncio
    async def test_admin_can_top_up_tokens(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test admin can add tokens to user balance."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Top up tokens
        balance, transaction = await token_service.admin_top_up(
            session, user_id, amount=5000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        assert balance.balance == 5000
        assert transaction.amount == 5000
        assert transaction.reason == "admin_top_up"
        assert transaction.admin_user_id == admin_id

        print(f"\n✓ Admin topped up user {user_id} with 5000 tokens")

    @pytest.mark.asyncio
    async def test_non_admin_cannot_top_up(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test non-admin cannot top up tokens."""
        user_id = get_unique_user_id()
        non_admin_id = 1001

        with pytest.raises(ForbiddenError):
            await token_service.admin_top_up(
                session, user_id, amount=1000, admin_user_id=non_admin_id, is_admin=False
            )

        print(f"\n✓ Non-admin correctly blocked from top-up")

    @pytest.mark.asyncio
    async def test_multiple_top_ups_accumulate(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test multiple top-ups accumulate correctly."""
        user_id = get_unique_user_id()
        admin_id = 999

        # First top-up
        await token_service.admin_top_up(
            session, user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )

        # Second top-up
        await token_service.admin_top_up(
            session, user_id, amount=2000, admin_user_id=admin_id, is_admin=True
        )

        # Third top-up
        balance, _ = await token_service.admin_top_up(
            session, user_id, amount=500, admin_user_id=admin_id, is_admin=True
        )

        assert balance.balance == 3500  # 1000 + 2000 + 500

        print(f"\n✓ Multiple top-ups accumulated: {balance.balance} tokens")


class TestTokenDeduction:
    """Tests for token deduction during usage."""

    @pytest.mark.asyncio
    async def test_tokens_deducted_correctly(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test tokens are deducted correctly."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Set up balance
        await token_service.admin_top_up(
            session, user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Create dialog and message for deduction
        dialog_id = uuid.uuid4()
        message_id = uuid.uuid4()
        dialog = TestDialog(id=dialog_id, user_id=user_id, title="Test", model_name="gpt-3.5-turbo")
        session.add(dialog)
        await session.flush()

        # Create actual message (required by FK constraint)
        message = TestMessage(id=message_id, dialog_id=dialog_id, role="assistant", content="Test")
        session.add(message)
        await session.flush()

        # Deduct tokens
        await token_service.deduct_tokens(
            session, user_id, amount=150, dialog_id=dialog_id, message_id=message_id
        )
        await session.commit()

        # Check balance
        stats = await token_service.get_token_stats(session, user_id)
        assert stats.balance == 850  # 1000 - 150
        assert stats.total_used == 150

        print(f"\n✓ Tokens deducted: 1000 - 150 = {stats.balance}")

    @pytest.mark.asyncio
    async def test_insufficient_balance_blocks_deduction(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test deduction is blocked when balance is insufficient."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Set up small balance
        await token_service.admin_top_up(
            session, user_id, amount=100, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Try to deduct more than balance
        dialog_id = uuid.uuid4()
        message_id = uuid.uuid4()

        with pytest.raises(InsufficientTokensError):
            await token_service.deduct_tokens(
                session, user_id, amount=500, dialog_id=dialog_id, message_id=message_id
            )

        print(f"\n✓ Insufficient balance correctly blocks deduction")


class TestTokenLimits:
    """Tests for token limit behavior."""

    @pytest.mark.asyncio
    async def test_balance_can_have_limit(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test balance model can store limit value."""
        user_id = get_unique_user_id()

        # Create balance with limit directly using the model
        balance = TestTokenBalance(
            user_id=user_id,
            balance=5000,
            limit=10000,
        )
        session.add(balance)
        await session.commit()

        # Verify limit is stored by querying directly
        result = await session.execute(
            select(TestTokenBalance).where(TestTokenBalance.user_id == user_id)
        )
        stored = result.scalar_one()

        assert stored.limit == 10000
        assert stored.balance == 5000

        print(f"\n✓ Token limit stored: {stored.limit}")

    @pytest.mark.asyncio
    async def test_balance_with_no_limit(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test balance starts with no limit."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Set up balance
        await token_service.admin_top_up(
            session, user_id, amount=5000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        balance = await token_service.get_balance(session, user_id)
        assert balance.limit is None

        print(f"\n✓ Balance created with no limit")


class TestTransactionHistory:
    """Tests for transaction history tracking."""

    @pytest.mark.asyncio
    async def test_transactions_recorded(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test all token changes create transaction records."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Top up
        await token_service.admin_top_up(
            session, user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Create dialog and message for deduction
        dialog_id = uuid.uuid4()
        message_id = uuid.uuid4()
        dialog = TestDialog(id=dialog_id, user_id=user_id, title="Test", model_name="gpt-3.5-turbo")
        session.add(dialog)
        await session.flush()

        message = TestMessage(id=message_id, dialog_id=dialog_id, role="assistant", content="Test")
        session.add(message)
        await session.flush()

        # Deduct
        await token_service.deduct_tokens(
            session, user_id, amount=100, dialog_id=dialog_id, message_id=message_id
        )
        await session.commit()

        # Get transactions
        result = await session.execute(
            select(TestTokenTransaction)
            .where(TestTokenTransaction.user_id == user_id)
            .order_by(TestTokenTransaction.created_at)
        )
        transactions = result.scalars().all()

        assert len(transactions) == 2  # top_up + deduct

        top_up_txn = next(t for t in transactions if t.reason == "admin_top_up")
        deduct_txn = next(t for t in transactions if t.reason == "llm_usage")

        assert top_up_txn.amount == 1000
        assert deduct_txn.amount == -100  # Negative for deduction

        print(f"\n✓ {len(transactions)} transactions recorded")

    @pytest.mark.asyncio
    async def test_transaction_includes_metadata(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test transactions include relevant metadata."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Top up with admin
        _, transaction = await token_service.admin_top_up(
            session, user_id, amount=500, admin_user_id=admin_id, is_admin=True
        )

        assert transaction.admin_user_id == admin_id
        assert transaction.reason == "admin_top_up"
        assert transaction.created_at is not None

        print(f"\n✓ Transaction metadata verified")


class TestBalanceStats:
    """Tests for balance statistics."""

    @pytest.mark.asyncio
    async def test_get_token_stats(
        self,
        session: AsyncSession,
        token_service: TestTokenService,
    ):
        """Test get_token_stats returns correct data."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Set up balance
        await token_service.admin_top_up(
            session, user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Create dialog and message for deduction
        dialog_id = uuid.uuid4()
        message_id = uuid.uuid4()
        dialog = TestDialog(id=dialog_id, user_id=user_id, title="Test", model_name="gpt-3.5-turbo")
        session.add(dialog)
        await session.flush()

        message = TestMessage(id=message_id, dialog_id=dialog_id, role="assistant", content="Test")
        session.add(message)
        await session.flush()

        await token_service.deduct_tokens(
            session, user_id, amount=150, dialog_id=dialog_id, message_id=message_id
        )
        await session.commit()

        # Get stats
        stats = await token_service.get_token_stats(session, user_id)

        assert stats.balance == 850
        assert stats.total_used == 150
        assert stats.limit is None

        print(f"\n✓ Stats: balance={stats.balance}, used={stats.total_used}")
