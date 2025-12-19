"""Unit tests for MessageService with mocked dependencies."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models import Dialog, Message
from src.domain.message_service import MessageService
from src.domain.token_service import TokenService
from src.shared.exceptions import (
    ForbiddenError,
    InsufficientTokensError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
)
from src.shared.schemas import MessageCreate


@pytest.fixture
def mock_token_service():
    """Create mock token service."""
    service = MagicMock(spec=TokenService)
    service.check_balance = AsyncMock(return_value=True)
    service.deduct_tokens = AsyncMock(return_value=(MagicMock(), MagicMock()))
    return service


@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=("Hello! How can I help?", 10, 20))
    return provider


@pytest.fixture
def mock_dialog():
    """Create mock dialog."""
    dialog = MagicMock(spec=Dialog)
    dialog.id = uuid.uuid4()
    dialog.user_id = 1
    dialog.model_name = "gpt-3.5-turbo"
    dialog.system_prompt = "You are a helpful assistant."
    return dialog


@pytest.fixture
def mock_message():
    """Create mock message."""
    message = MagicMock(spec=Message)
    message.id = uuid.uuid4()
    message.dialog_id = uuid.uuid4()
    message.role = "assistant"
    message.content = "Hello! How can I help?"
    message.prompt_tokens = 10
    message.completion_tokens = 20
    message.created_at = datetime.now(timezone.utc)
    return message


@pytest.fixture
def message_service(mock_token_service, mock_llm_provider):
    """Create MessageService with mocked dependencies."""
    service = MessageService(mock_token_service, mock_llm_provider)
    service.dialog_repo = AsyncMock()
    service.message_repo = AsyncMock()
    return service


# Send Message Tests


@pytest.mark.asyncio
async def test_send_message_success(message_service, mock_dialog, mock_message):
    """Test successful message send."""
    session = AsyncMock()

    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    data = MessageCreate(content="Hello")
    result = await message_service.send_message(
        session, mock_dialog.id, user_id=1, data=data
    )

    assert result.id == mock_message.id
    assert result.role == "assistant"
    message_service.llm_provider.generate.assert_called_once()
    message_service.token_service.deduct_tokens.assert_called_once()


@pytest.mark.asyncio
async def test_send_message_dialog_not_found(message_service):
    """Test send message raises NotFoundError for missing dialog."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = None

    data = MessageCreate(content="Hello")

    with pytest.raises(NotFoundError) as exc_info:
        await message_service.send_message(
            session, uuid.uuid4(), user_id=1, data=data
        )

    assert "not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_send_message_access_denied(message_service, mock_dialog):
    """Test send message raises ForbiddenError for wrong user."""
    session = AsyncMock()
    mock_dialog.user_id = 1
    message_service.dialog_repo.get_by_id.return_value = mock_dialog

    data = MessageCreate(content="Hello")

    with pytest.raises(ForbiddenError) as exc_info:
        await message_service.send_message(
            session, mock_dialog.id, user_id=999, data=data, is_admin=False
        )

    assert "Access denied" in exc_info.value.message


@pytest.mark.asyncio
async def test_send_message_admin_can_access_any(message_service, mock_dialog, mock_message):
    """Test admin can access any dialog."""
    session = AsyncMock()
    mock_dialog.user_id = 1
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    data = MessageCreate(content="Hello")

    # Admin (user 999) accessing user 1's dialog
    result = await message_service.send_message(
        session, mock_dialog.id, user_id=999, data=data, is_admin=True
    )

    assert result is not None


@pytest.mark.asyncio
async def test_send_message_insufficient_tokens(message_service, mock_dialog):
    """Test send message raises InsufficientTokensError."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.token_service.check_balance.return_value = False

    data = MessageCreate(content="Hello")

    with pytest.raises(InsufficientTokensError) as exc_info:
        await message_service.send_message(
            session, mock_dialog.id, user_id=1, data=data
        )

    assert "Insufficient tokens" in exc_info.value.message


@pytest.mark.asyncio
async def test_send_message_llm_timeout(message_service, mock_dialog, mock_message):
    """Test send message raises LLMTimeoutError."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []
    message_service.llm_provider.generate.side_effect = LLMTimeoutError()

    data = MessageCreate(content="Hello")

    with pytest.raises(LLMTimeoutError):
        await message_service.send_message(
            session, mock_dialog.id, user_id=1, data=data
        )

    # Should rollback on LLM failure
    session.rollback.assert_called()


@pytest.mark.asyncio
async def test_send_message_llm_error(message_service, mock_dialog, mock_message):
    """Test send message raises LLMError."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []
    message_service.llm_provider.generate.side_effect = LLMError("API error")

    data = MessageCreate(content="Hello")

    with pytest.raises(LLMError):
        await message_service.send_message(
            session, mock_dialog.id, user_id=1, data=data
        )

    session.rollback.assert_called()


@pytest.mark.asyncio
async def test_send_message_no_llm_provider():
    """Test send message raises LLMError when no provider."""
    token_service = MagicMock(spec=TokenService)
    token_service.check_balance = AsyncMock(return_value=True)

    service = MessageService(token_service, llm_provider=None)
    service.dialog_repo = AsyncMock()
    service.message_repo = AsyncMock()

    mock_dialog = MagicMock(spec=Dialog)
    mock_dialog.id = uuid.uuid4()
    mock_dialog.user_id = 1
    service.dialog_repo.get_by_id.return_value = mock_dialog

    mock_message = MagicMock(spec=Message)
    mock_message.id = uuid.uuid4()
    service.message_repo.create_user_message.return_value = mock_message
    service.message_repo.get_by_dialog.return_value = []

    session = AsyncMock()
    data = MessageCreate(content="Hello")

    with pytest.raises(LLMError) as exc_info:
        await service.send_message(session, mock_dialog.id, user_id=1, data=data)

    assert "not configured" in exc_info.value.message


# Event Emission Tests


@pytest.mark.asyncio
async def test_send_message_emits_events(message_service, mock_dialog, mock_message):
    """Test send message emits proper events."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    emitted_events = []
    message_service.register_event_handler(lambda e: emitted_events.append(e))

    data = MessageCreate(content="Hello")
    await message_service.send_message(
        session, mock_dialog.id, user_id=1, data=data
    )

    # Should emit MessageSentEvent and LLMResponseEvent
    assert len(emitted_events) == 2
    assert emitted_events[0].event_type == "message_sent"
    assert emitted_events[1].event_type == "llm_response_received"


@pytest.mark.asyncio
async def test_event_handler_error_does_not_propagate(message_service, mock_dialog, mock_message):
    """Test event handler errors don't stop message flow."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    def failing_handler(event):
        raise RuntimeError("Handler error")

    message_service.register_event_handler(failing_handler)

    data = MessageCreate(content="Hello")

    # Should not raise
    result = await message_service.send_message(
        session, mock_dialog.id, user_id=1, data=data
    )

    assert result is not None


# Get Messages Tests


@pytest.mark.asyncio
async def test_get_messages_success(message_service, mock_dialog, mock_message):
    """Test getting messages for a dialog."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.get_by_dialog.return_value = [mock_message]

    result = await message_service.get_messages(
        session, mock_dialog.id, user_id=1
    )

    assert len(result) == 1
    assert result[0].id == mock_message.id


@pytest.mark.asyncio
async def test_get_messages_dialog_not_found(message_service):
    """Test get messages raises NotFoundError."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = None

    with pytest.raises(NotFoundError):
        await message_service.get_messages(session, uuid.uuid4(), user_id=1)


# Build Messages Tests


@pytest.mark.asyncio
async def test_build_messages_includes_system_prompt(message_service, mock_dialog):
    """Test built messages include system prompt."""
    session = AsyncMock()
    mock_dialog.system_prompt = "You are a helpful assistant."
    message_service.message_repo.get_by_dialog.return_value = []

    messages = await message_service._build_messages_for_llm(
        session, mock_dialog, "Hello"
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_build_messages_includes_history(message_service, mock_dialog):
    """Test built messages include conversation history."""
    session = AsyncMock()
    mock_dialog.system_prompt = None

    history_message = MagicMock(spec=Message)
    history_message.role = "user"
    history_message.content = "Previous message"

    message_service.message_repo.get_by_dialog.return_value = [history_message]

    messages = await message_service._build_messages_for_llm(
        session, mock_dialog, "New message"
    )

    assert len(messages) == 2
    assert messages[0]["content"] == "Previous message"
    assert messages[1]["content"] == "New message"


# Token Deduction Tests


@pytest.mark.asyncio
async def test_tokens_deducted_after_success(message_service, mock_dialog, mock_message):
    """Test tokens are deducted after successful LLM response."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    # LLM returns 10 prompt + 20 completion = 30 total tokens
    message_service.llm_provider.generate.return_value = ("Response", 10, 20)

    data = MessageCreate(content="Hello")
    await message_service.send_message(
        session, mock_dialog.id, user_id=1, data=data
    )

    # Should deduct 30 tokens
    message_service.token_service.deduct_tokens.assert_called_once()
    call_args = message_service.token_service.deduct_tokens.call_args
    assert call_args[0][2] == 30  # amount


@pytest.mark.asyncio
async def test_no_tokens_deducted_on_llm_failure(message_service, mock_dialog, mock_message):
    """Test no tokens deducted when LLM fails."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []
    message_service.llm_provider.generate.side_effect = LLMError()

    data = MessageCreate(content="Hello")

    with pytest.raises(LLMError):
        await message_service.send_message(
            session, mock_dialog.id, user_id=1, data=data
        )

    # Should NOT deduct tokens
    message_service.token_service.deduct_tokens.assert_not_called()


# Transaction Tests


@pytest.mark.asyncio
async def test_transaction_committed_on_success(message_service, mock_dialog, mock_message):
    """Test transaction is committed on success."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.create_assistant_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []

    data = MessageCreate(content="Hello")
    await message_service.send_message(
        session, mock_dialog.id, user_id=1, data=data
    )

    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_transaction_rolled_back_on_llm_failure(message_service, mock_dialog, mock_message):
    """Test transaction is rolled back on LLM failure."""
    session = AsyncMock()
    message_service.dialog_repo.get_by_id.return_value = mock_dialog
    message_service.message_repo.create_user_message.return_value = mock_message
    message_service.message_repo.get_by_dialog.return_value = []
    message_service.llm_provider.generate.side_effect = LLMError()

    data = MessageCreate(content="Hello")

    with pytest.raises(LLMError):
        await message_service.send_message(
            session, mock_dialog.id, user_id=1, data=data
        )

    session.rollback.assert_called()
    session.commit.assert_not_called()
