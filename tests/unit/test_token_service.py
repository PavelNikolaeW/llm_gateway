"""Unit tests for TokenService with mocked repositories."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import TokenBalance, TokenTransaction
from src.domain.token_service import TokenService
from src.shared.exceptions import ForbiddenError, InsufficientTokensError
from src.shared.schemas import TokenEvent


@pytest.fixture
def token_service():
    """Create TokenService with mocked repositories."""
    service = TokenService()
    service.balance_repo = AsyncMock()
    service.transaction_repo = AsyncMock()
    return service


@pytest.fixture
def mock_balance():
    """Create a mock token balance."""
    balance = MagicMock(spec=TokenBalance)
    balance.user_id = 1
    balance.balance = 1000
    balance.limit = None
    balance.updated_at = datetime.now(timezone.utc)
    return balance


@pytest.fixture
def mock_transaction():
    """Create a mock token transaction."""
    transaction = MagicMock(spec=TokenTransaction)
    transaction.id = 1
    transaction.user_id = 1
    transaction.amount = -100
    transaction.reason = "llm_usage"
    transaction.dialog_id = uuid.uuid4()
    transaction.message_id = uuid.uuid4()
    transaction.admin_user_id = None
    transaction.created_at = datetime.now(timezone.utc)
    return transaction


# Balance Check Tests


@pytest.mark.asyncio
async def test_check_balance_sufficient(token_service, mock_balance):
    """Test check_balance returns True when balance >= estimated_cost."""
    session = AsyncMock()
    mock_balance.balance = 1000
    token_service.balance_repo.get_or_create.return_value = mock_balance

    result = await token_service.check_balance(session, user_id=1, estimated_cost=500)

    assert result is True
    token_service.balance_repo.get_or_create.assert_called_once_with(session, 1)


@pytest.mark.asyncio
async def test_check_balance_insufficient(token_service, mock_balance):
    """Test check_balance returns False when balance < estimated_cost."""
    session = AsyncMock()
    mock_balance.balance = 100
    token_service.balance_repo.get_or_create.return_value = mock_balance

    # Track emitted events
    emitted_events = []
    token_service.register_event_handler(lambda e: emitted_events.append(e))

    result = await token_service.check_balance(session, user_id=1, estimated_cost=500)

    assert result is False
    # Should emit balance_exhausted event
    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "balance_exhausted"
    assert emitted_events[0].user_id == 1
    assert emitted_events[0].new_balance == 100


@pytest.mark.asyncio
async def test_check_balance_exact_amount(token_service, mock_balance):
    """Test check_balance returns True when balance == estimated_cost."""
    session = AsyncMock()
    mock_balance.balance = 500
    token_service.balance_repo.get_or_create.return_value = mock_balance

    result = await token_service.check_balance(session, user_id=1, estimated_cost=500)

    assert result is True


# Deduct Tokens Tests


@pytest.mark.asyncio
async def test_deduct_tokens_success(token_service, mock_balance, mock_transaction):
    """Test successful token deduction."""
    session = AsyncMock()
    dialog_id = uuid.uuid4()
    message_id = uuid.uuid4()

    mock_balance.balance = 1000
    token_service.balance_repo.get_or_create.return_value = mock_balance

    updated_balance = MagicMock(spec=TokenBalance)
    updated_balance.user_id = 1
    updated_balance.balance = 900  # After deduction
    updated_balance.limit = None
    updated_balance.updated_at = datetime.now(timezone.utc)
    token_service.balance_repo.deduct_tokens.return_value = updated_balance

    mock_transaction.dialog_id = dialog_id
    mock_transaction.message_id = message_id
    token_service.transaction_repo.create_llm_usage_transaction.return_value = mock_transaction

    # Track emitted events
    emitted_events = []
    token_service.register_event_handler(lambda e: emitted_events.append(e))

    balance_resp, transaction_resp = await token_service.deduct_tokens(
        session, user_id=1, amount=100, dialog_id=dialog_id, message_id=message_id
    )

    assert balance_resp.balance == 900
    assert transaction_resp.amount == -100
    assert transaction_resp.reason == "llm_usage"

    # Should emit tokens_deducted event
    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "tokens_deducted"
    assert emitted_events[0].amount == 100
    assert emitted_events[0].new_balance == 900


@pytest.mark.asyncio
async def test_deduct_tokens_insufficient_balance(token_service, mock_balance):
    """Test deduct_tokens raises InsufficientTokensError when balance is low."""
    session = AsyncMock()
    dialog_id = uuid.uuid4()
    message_id = uuid.uuid4()

    mock_balance.balance = 50
    token_service.balance_repo.get_or_create.return_value = mock_balance

    # Track emitted events
    emitted_events = []
    token_service.register_event_handler(lambda e: emitted_events.append(e))

    with pytest.raises(InsufficientTokensError) as exc_info:
        await token_service.deduct_tokens(
            session, user_id=1, amount=100, dialog_id=dialog_id, message_id=message_id
        )

    assert "Insufficient tokens" in str(exc_info.value.message)
    assert "balance=50" in str(exc_info.value.message)
    assert "required=100" in str(exc_info.value.message)

    # Should emit balance_exhausted event
    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "balance_exhausted"


@pytest.mark.asyncio
async def test_deduct_tokens_negative_amount_rejected(token_service):
    """Test deduct_tokens raises ValueError for non-positive amounts."""
    session = AsyncMock()
    dialog_id = uuid.uuid4()
    message_id = uuid.uuid4()

    with pytest.raises(ValueError) as exc_info:
        await token_service.deduct_tokens(
            session, user_id=1, amount=-100, dialog_id=dialog_id, message_id=message_id
        )

    assert "positive" in str(exc_info.value)


@pytest.mark.asyncio
async def test_deduct_tokens_zero_amount_rejected(token_service):
    """Test deduct_tokens raises ValueError for zero amount."""
    session = AsyncMock()
    dialog_id = uuid.uuid4()
    message_id = uuid.uuid4()

    with pytest.raises(ValueError):
        await token_service.deduct_tokens(
            session, user_id=1, amount=0, dialog_id=dialog_id, message_id=message_id
        )


# Admin Top-Up Tests


@pytest.mark.asyncio
async def test_admin_top_up_positive_amount(token_service, mock_balance, mock_transaction):
    """Test admin top-up with positive amount."""
    session = AsyncMock()

    updated_balance = MagicMock(spec=TokenBalance)
    updated_balance.user_id = 1
    updated_balance.balance = 1500  # After top-up
    updated_balance.limit = None
    updated_balance.updated_at = datetime.now(timezone.utc)
    token_service.balance_repo.add_tokens.return_value = updated_balance

    mock_transaction.amount = 500
    mock_transaction.reason = "admin_top_up"
    mock_transaction.admin_user_id = 999
    token_service.transaction_repo.create_admin_transaction.return_value = mock_transaction

    balance_resp, transaction_resp = await token_service.admin_top_up(
        session, user_id=1, amount=500, admin_user_id=999, is_admin=True
    )

    assert balance_resp.balance == 1500
    assert transaction_resp.reason == "admin_top_up"
    token_service.transaction_repo.create_admin_transaction.assert_called_with(
        session, 1, 500, 999, "admin_top_up"
    )


@pytest.mark.asyncio
async def test_admin_top_up_negative_amount_deducts(token_service, mock_balance, mock_transaction):
    """Test admin top-up with negative amount becomes deduction."""
    session = AsyncMock()

    updated_balance = MagicMock(spec=TokenBalance)
    updated_balance.user_id = 1
    updated_balance.balance = 500  # After deduction
    updated_balance.limit = None
    updated_balance.updated_at = datetime.now(timezone.utc)
    token_service.balance_repo.add_tokens.return_value = updated_balance

    mock_transaction.amount = -500
    mock_transaction.reason = "admin_deduct"
    mock_transaction.admin_user_id = 999
    token_service.transaction_repo.create_admin_transaction.return_value = mock_transaction

    balance_resp, transaction_resp = await token_service.admin_top_up(
        session, user_id=1, amount=-500, admin_user_id=999, is_admin=True
    )

    assert balance_resp.balance == 500
    assert transaction_resp.reason == "admin_deduct"
    token_service.transaction_repo.create_admin_transaction.assert_called_with(
        session, 1, -500, 999, "admin_deduct"
    )


@pytest.mark.asyncio
async def test_admin_top_up_non_admin_forbidden(token_service):
    """Test admin top-up raises ForbiddenError for non-admin users."""
    session = AsyncMock()

    with pytest.raises(ForbiddenError) as exc_info:
        await token_service.admin_top_up(
            session, user_id=1, amount=500, admin_user_id=2, is_admin=False
        )

    assert "admin" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_admin_deduct_emits_exhausted_event_when_negative(
    token_service, mock_balance, mock_transaction
):
    """Test admin deduction emits balance_exhausted when balance goes negative."""
    session = AsyncMock()

    updated_balance = MagicMock(spec=TokenBalance)
    updated_balance.user_id = 1
    updated_balance.balance = -100  # Negative after deduction
    updated_balance.limit = None
    updated_balance.updated_at = datetime.now(timezone.utc)
    token_service.balance_repo.add_tokens.return_value = updated_balance

    mock_transaction.amount = -500
    mock_transaction.reason = "admin_deduct"
    mock_transaction.admin_user_id = 999
    token_service.transaction_repo.create_admin_transaction.return_value = mock_transaction

    # Track emitted events
    emitted_events = []
    token_service.register_event_handler(lambda e: emitted_events.append(e))

    await token_service.admin_top_up(
        session, user_id=1, amount=-500, admin_user_id=999, is_admin=True
    )

    # Should emit balance_exhausted event
    assert len(emitted_events) == 1
    assert emitted_events[0].event_type == "balance_exhausted"
    assert emitted_events[0].new_balance == -100


# Get Balance Tests


@pytest.mark.asyncio
async def test_get_balance(token_service, mock_balance):
    """Test getting current balance."""
    session = AsyncMock()
    mock_balance.balance = 1000
    token_service.balance_repo.get_or_create.return_value = mock_balance

    result = await token_service.get_balance(session, user_id=1)

    assert result.user_id == 1
    assert result.balance == 1000


# Transaction History Tests


@pytest.mark.asyncio
async def test_get_transaction_history(token_service, mock_transaction):
    """Test getting transaction history."""
    session = AsyncMock()

    transactions = [mock_transaction]
    token_service.transaction_repo.get_by_user.return_value = transactions

    result = await token_service.get_transaction_history(session, user_id=1)

    assert len(result) == 1
    assert result[0].user_id == 1
    token_service.transaction_repo.get_by_user.assert_called_once_with(session, 1, 0, 100)


@pytest.mark.asyncio
async def test_get_transaction_history_with_pagination(token_service, mock_transaction):
    """Test getting transaction history with pagination."""
    session = AsyncMock()

    token_service.transaction_repo.get_by_user.return_value = []

    await token_service.get_transaction_history(session, user_id=1, skip=10, limit=20)

    token_service.transaction_repo.get_by_user.assert_called_once_with(session, 1, 10, 20)


# Event Handler Tests


@pytest.mark.asyncio
async def test_event_handler_registration(token_service, mock_balance):
    """Test event handlers can be registered and called."""
    session = AsyncMock()
    mock_balance.balance = 50
    token_service.balance_repo.get_or_create.return_value = mock_balance

    # Register multiple handlers
    events_handler1 = []
    events_handler2 = []
    token_service.register_event_handler(lambda e: events_handler1.append(e))
    token_service.register_event_handler(lambda e: events_handler2.append(e))

    await token_service.check_balance(session, user_id=1, estimated_cost=100)

    # Both handlers should receive the event
    assert len(events_handler1) == 1
    assert len(events_handler2) == 1
    assert events_handler1[0].event_type == "balance_exhausted"


@pytest.mark.asyncio
async def test_event_handler_error_does_not_propagate(token_service, mock_balance):
    """Test that errors in event handlers don't propagate to caller."""
    session = AsyncMock()
    mock_balance.balance = 50
    token_service.balance_repo.get_or_create.return_value = mock_balance

    # Register a handler that raises an exception
    def failing_handler(event):
        raise RuntimeError("Handler error")

    token_service.register_event_handler(failing_handler)

    # Should not raise exception
    result = await token_service.check_balance(session, user_id=1, estimated_cost=100)
    assert result is False
