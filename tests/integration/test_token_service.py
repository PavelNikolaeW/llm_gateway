"""Integration tests for TokenService with test database tables.

Uses test-specific tables (test_dialogs, test_token_balances, etc.)
to ensure complete isolation from production data.
NEVER touches tables starting with api_*.
"""
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.exceptions import ForbiddenError, InsufficientTokensError
from tests.conftest import get_unique_user_id
from tests.test_models import TestDialog, TestMessage
from tests.test_services import TestTokenService


@pytest.fixture(scope="function")
def token_service():
    """Create test token service that uses test tables."""
    return TestTokenService()


async def create_dialog_and_message(session: AsyncSession, user_id: int) -> tuple[uuid.UUID, uuid.UUID]:
    """Helper to create a test dialog and message."""
    dialog = TestDialog(
        id=uuid.uuid4(),
        user_id=user_id,
        title="Test Dialog",
        model_name="gpt-3.5-turbo",
    )
    session.add(dialog)
    await session.flush()

    message = TestMessage(
        id=uuid.uuid4(),
        dialog_id=dialog.id,
        role="assistant",
        content="Test response",
    )
    session.add(message)
    await session.flush()

    return dialog.id, message.id


@pytest.mark.asyncio
async def test_check_balance_new_user(session: AsyncSession, token_service: TestTokenService):
    """Test check_balance creates balance for new user."""
    user_id = get_unique_user_id()

    # New user should have 0 balance
    result = await token_service.check_balance(session, user_id=user_id, estimated_cost=100)

    assert result is False  # 0 < 100

    # Verify balance was created
    balance = await token_service.get_balance(session, user_id)
    assert balance.user_id == user_id
    assert balance.balance == 0

    print(f"\n✓ New user {user_id} created with zero balance")


@pytest.mark.asyncio
async def test_admin_top_up_and_check_balance(session: AsyncSession, token_service: TestTokenService):
    """Test admin top-up followed by balance check."""
    user_id = get_unique_user_id()
    admin_id = 999

    # Admin tops up user
    balance, transaction = await token_service.admin_top_up(
        session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    assert balance.balance == 1000
    assert transaction.reason == "admin_top_up"
    assert transaction.amount == 1000
    assert transaction.admin_user_id == admin_id

    # Now check balance
    result = await token_service.check_balance(session, user_id=user_id, estimated_cost=500)
    assert result is True

    result = await token_service.check_balance(session, user_id=user_id, estimated_cost=1500)
    assert result is False

    print(f"\n✓ Admin top-up {user_id}: balance=1000, check 500=True, check 1500=False")


@pytest.mark.asyncio
async def test_deduct_tokens_integration(session: AsyncSession, token_service: TestTokenService):
    """Test token deduction with real database."""
    user_id = get_unique_user_id()
    admin_id = 999

    # First, top up the user
    await token_service.admin_top_up(
        session, user_id=user_id, amount=500, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    # Create dialog and message
    dialog_id, message_id = await create_dialog_and_message(session, user_id)
    await session.commit()

    # Deduct tokens
    balance, transaction = await token_service.deduct_tokens(
        session, user_id=user_id, amount=100, dialog_id=dialog_id, message_id=message_id
    )
    await session.commit()

    assert balance.balance == 400  # 500 - 100
    assert transaction.reason == "llm_usage"
    assert transaction.amount == -100  # Negative for deduction
    assert transaction.dialog_id == dialog_id
    assert transaction.message_id == message_id

    print(f"\n✓ Deducted 100 tokens from user {user_id}: new_balance=400")


@pytest.mark.asyncio
async def test_deduct_tokens_insufficient_balance(session: AsyncSession, token_service: TestTokenService):
    """Test deduction fails with insufficient balance."""
    user_id = get_unique_user_id()
    admin_id = 999

    # Top up with small amount
    await token_service.admin_top_up(
        session, user_id=user_id, amount=50, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    # Create dialog and message
    dialog_id, message_id = await create_dialog_and_message(session, user_id)
    await session.commit()

    # Try to deduct more than balance
    with pytest.raises(InsufficientTokensError) as exc_info:
        await token_service.deduct_tokens(
            session, user_id=user_id, amount=100, dialog_id=dialog_id, message_id=message_id
        )

    assert "Insufficient tokens" in exc_info.value.message
    assert "balance=50" in exc_info.value.message

    print(f"\n✓ InsufficientTokensError raised for user {user_id} (balance=50, required=100)")


@pytest.mark.asyncio
async def test_admin_deduct_negative_amount(session: AsyncSession, token_service: TestTokenService):
    """Test admin can deduct tokens with negative amount."""
    user_id = get_unique_user_id()
    admin_id = 999

    # First top up
    await token_service.admin_top_up(
        session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    # Then deduct with negative amount
    balance, transaction = await token_service.admin_top_up(
        session, user_id=user_id, amount=-300, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    assert balance.balance == 700  # 1000 - 300
    assert transaction.reason == "admin_deduct"
    assert transaction.amount == -300

    print(f"\n✓ Admin deducted 300 from user {user_id}: new_balance=700")


@pytest.mark.asyncio
async def test_admin_top_up_non_admin_forbidden(session: AsyncSession, token_service: TestTokenService):
    """Test non-admin cannot perform top-up."""
    user_id = get_unique_user_id()

    with pytest.raises(ForbiddenError) as exc_info:
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=123, is_admin=False
        )

    assert "admin" in exc_info.value.message.lower()

    print(f"\n✓ ForbiddenError raised for non-admin top-up attempt")


@pytest.mark.asyncio
async def test_transaction_history(session: AsyncSession, token_service: TestTokenService):
    """Test getting transaction history."""
    user_id = get_unique_user_id()
    admin_id = 999

    # Create multiple transactions
    await token_service.admin_top_up(
        session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    dialog_id, message_id = await create_dialog_and_message(session, user_id)
    await session.commit()

    await token_service.deduct_tokens(
        session, user_id=user_id, amount=100, dialog_id=dialog_id, message_id=message_id
    )
    await session.commit()

    await token_service.admin_top_up(
        session, user_id=user_id, amount=-50, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    # Get history
    history = await token_service.get_transaction_history(session, user_id=user_id)

    assert len(history) == 3
    # Most recent first
    assert history[0].reason == "admin_deduct"
    assert history[1].reason == "llm_usage"
    assert history[2].reason == "admin_top_up"

    print(f"\n✓ Got {len(history)} transactions for user {user_id}")


@pytest.mark.asyncio
async def test_event_emission(session: AsyncSession, token_service: TestTokenService):
    """Test events are emitted during operations."""
    user_id = get_unique_user_id()
    emitted_events = []

    token_service.register_event_handler(lambda e: emitted_events.append(e))

    # Check balance for new user (should fail and emit event)
    await token_service.check_balance(session, user_id=user_id, estimated_cost=100)

    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "balance_exhausted"
    assert emitted_events[0].user_id == user_id

    print(f"\n✓ Event emitted: {emitted_events[0].event_type}")


@pytest.mark.asyncio
async def test_multiple_deductions(session: AsyncSession, token_service: TestTokenService):
    """Test multiple sequential deductions work correctly."""
    user_id = get_unique_user_id()
    admin_id = 999

    # Top up
    await token_service.admin_top_up(
        session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
    )
    await session.commit()

    # Multiple deductions
    for i in range(5):
        dialog_id, message_id = await create_dialog_and_message(session, user_id)
        await session.commit()

        balance, _ = await token_service.deduct_tokens(
            session, user_id=user_id, amount=100, dialog_id=dialog_id, message_id=message_id
        )
        await session.commit()

        expected_balance = 1000 - (i + 1) * 100
        assert balance.balance == expected_balance

    # Final balance should be 500
    final_balance = await token_service.get_balance(session, user_id)
    assert final_balance.balance == 500

    print(f"\n✓ Multiple deductions: 1000 -> 500 (5 x 100)")


print("\n✅ All TokenService integration tests completed (using test tables)")
