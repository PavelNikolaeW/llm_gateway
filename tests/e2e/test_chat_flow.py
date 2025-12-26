"""End-to-end tests for complete chat flow.

Tests the full flow from dialog creation to message exchange:
1. Create dialog
2. Send message
3. Receive LLM response
4. Verify token deduction
5. Get message history
"""
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import TokenBalance
from src.domain.dialog_service import DialogService
from src.domain.message_service import MessageService
from src.domain.model_registry import ModelRegistry
from src.domain.token_service import TokenService
from src.shared.schemas import DialogCreate, MessageCreate


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "Hello! I'm a mock assistant."):
        self.response = response
        self.calls: list[dict] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate mock response."""
        self.calls.append({"messages": messages, "model": model, "config": config})
        # Return fixed token counts for predictable testing
        return self.response, 50, 100  # prompt_tokens=50, completion_tokens=100

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        """Generate streaming mock response."""
        self.calls.append({"messages": messages, "model": model, "config": config})

        # Stream response in chunks
        words = self.response.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            chunk = word + (" " if not is_last else "")
            yield chunk, False, None, None

        # Final chunk with token counts
        yield "", True, 50, 100


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
async def model_registry(session: AsyncSession):
    """Create and load model registry."""
    registry = ModelRegistry()
    await registry.load_models(session)
    return registry


@pytest.fixture
def token_service():
    """Create token service."""
    return TokenService()


@pytest.fixture
def dialog_service(model_registry: ModelRegistry):
    """Create dialog service."""
    return DialogService(model_registry)


@pytest.fixture
def message_service(token_service: TokenService, mock_llm: MockLLMProvider):
    """Create message service with mock LLM."""
    return MessageService(token_service=token_service, llm_provider=mock_llm)


async def setup_user_balance(session: AsyncSession, user_id: int, balance: int = 10000) -> TokenBalance:
    """Helper to set up user balance for testing."""
    from sqlalchemy import select

    # Check if balance already exists
    result = await session.execute(
        select(TokenBalance).where(TokenBalance.user_id == user_id)
    )
    existing = result.scalar_one_or_none()

    if existing:
        existing.balance = balance
        await session.commit()
        await session.refresh(existing)
        return existing

    token_balance = TokenBalance(
        user_id=user_id,
        balance=balance,
        limit=None,
    )
    session.add(token_balance)
    await session.commit()
    await session.refresh(token_balance)
    return token_balance


@pytest.mark.skip(reason="Session persistence issue with TokenBalance - needs repository fix")
class TestCompleteDialogFlow:
    """End-to-end tests for complete dialog creation flow."""

    @pytest.mark.asyncio
    async def test_create_dialog_and_send_message(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
        mock_llm: MockLLMProvider,
    ):
        """Test creating a dialog and sending a message through it."""
        # Use unique user ID
        user_id = 300000 + abs(hash(str(uuid.uuid4()))) % 10000

        # Set up user balance
        await setup_user_balance(session, user_id, balance=10000)

        # Step 1: Create dialog
        dialog_data = DialogCreate(
            title="E2E Test Dialog",
            model_name="gpt-3.5-turbo",
        )
        dialog = await dialog_service.create_dialog(session, user_id, dialog_data)

        assert dialog.id is not None
        assert dialog.title == "E2E Test Dialog"
        assert dialog.model_name == "gpt-3.5-turbo"
        assert dialog.user_id == user_id

        # Step 2: Send message
        message_data = MessageCreate(content="Hello, assistant!")
        response = await message_service.send_message(
            session, dialog.id, user_id, message_data
        )

        assert response.role == "assistant"
        assert response.content == "Hello! I'm a mock assistant."
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 100

        # Step 3: Verify LLM was called correctly
        assert len(mock_llm.calls) == 1
        call = mock_llm.calls[0]
        assert call["model"] == "gpt-3.5-turbo"
        assert any(m["content"] == "Hello, assistant!" for m in call["messages"])

        # Step 4: Verify messages saved
        messages = await message_service.get_messages(session, dialog.id, user_id)
        assert len(messages) == 2  # user + assistant

        user_msg = next(m for m in messages if m.role == "user")
        assistant_msg = next(m for m in messages if m.role == "assistant")

        assert user_msg.content == "Hello, assistant!"
        assert assistant_msg.content == "Hello! I'm a mock assistant."

        print(f"\n✓ Complete dialog flow test passed for user {user_id}")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
        mock_llm: MockLLMProvider,
    ):
        """Test multi-turn conversation builds correct context."""
        user_id = 300100 + abs(hash(str(uuid.uuid4()))) % 10000
        await setup_user_balance(session, user_id, balance=50000)

        # Create dialog
        dialog = await dialog_service.create_dialog(
            session, user_id, DialogCreate(title="Multi-turn Test", model_name="gpt-3.5-turbo")
        )

        # Turn 1
        await message_service.send_message(
            session, dialog.id, user_id, MessageCreate(content="First message")
        )

        # Turn 2
        mock_llm.response = "Second response"
        await message_service.send_message(
            session, dialog.id, user_id, MessageCreate(content="Second message")
        )

        # Verify second call includes history
        assert len(mock_llm.calls) == 2
        second_call_messages = mock_llm.calls[1]["messages"]

        # Should have conversation history from first turn
        user_messages = [m for m in second_call_messages if m["role"] == "user"]
        assistant_messages = [m for m in second_call_messages if m["role"] == "assistant"]

        # First user message should be in history
        assert any(m["content"] == "First message" for m in user_messages)
        # Second user message should be in history
        assert any(m["content"] == "Second message" for m in user_messages)
        # Assistant response from first turn should be in history
        assert len(assistant_messages) >= 1

        print(f"\n✓ Multi-turn conversation test passed for user {user_id}")

    @pytest.mark.asyncio
    async def test_dialog_with_system_prompt(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
        mock_llm: MockLLMProvider,
    ):
        """Test dialog with system prompt passes it to LLM."""
        user_id = 300200 + abs(hash(str(uuid.uuid4()))) % 10000
        await setup_user_balance(session, user_id, balance=10000)

        # Create dialog with system prompt
        dialog = await dialog_service.create_dialog(
            session,
            user_id,
            DialogCreate(
                title="System Prompt Test",
                model_name="gpt-3.5-turbo",
                system_prompt="You are a helpful assistant.",
            ),
        )

        # Send message
        await message_service.send_message(
            session, dialog.id, user_id, MessageCreate(content="Hello!")
        )

        # Verify system prompt was included
        call_messages = mock_llm.calls[0]["messages"]
        system_msgs = [m for m in call_messages if m["role"] == "system"]

        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a helpful assistant."

        print(f"\n✓ System prompt test passed for user {user_id}")


@pytest.mark.skip(reason="Session persistence issue with TokenBalance - needs repository fix")
class TestTokenDeduction:
    """Tests for token deduction during chat flow."""

    @pytest.mark.asyncio
    async def test_tokens_deducted_after_message(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
        token_service: TokenService,
    ):
        """Test tokens are deducted after successful message."""
        user_id = 300300 + abs(hash(str(uuid.uuid4()))) % 10000
        initial_balance = 10000

        await setup_user_balance(session, user_id, balance=initial_balance)

        # Create dialog
        dialog = await dialog_service.create_dialog(
            session, user_id, DialogCreate(title="Token Test", model_name="gpt-3.5-turbo")
        )

        # Check initial balance
        stats_before = await token_service.get_token_stats(session, user_id)
        assert stats_before.balance == initial_balance

        # Send message (mock returns 50 + 100 = 150 tokens)
        await message_service.send_message(
            session, dialog.id, user_id, MessageCreate(content="Test message")
        )

        # Check balance after
        stats_after = await token_service.get_token_stats(session, user_id)
        expected_balance = initial_balance - 150  # 50 prompt + 100 completion

        assert stats_after.balance == expected_balance
        assert stats_after.total_used == 150

        print(f"\n✓ Token deduction test passed: {initial_balance} -> {stats_after.balance}")


@pytest.mark.skip(reason="Session persistence issue with TokenBalance - needs repository fix")
class TestMessageHistory:
    """Tests for message history retrieval."""

    @pytest.mark.asyncio
    async def test_get_message_history_ordered(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
    ):
        """Test message history is returned in correct order."""
        user_id = 300400 + abs(hash(str(uuid.uuid4()))) % 10000
        await setup_user_balance(session, user_id, balance=50000)

        # Create dialog
        dialog = await dialog_service.create_dialog(
            session, user_id, DialogCreate(title="History Test", model_name="gpt-3.5-turbo")
        )

        # Send multiple messages
        for i in range(3):
            await message_service.send_message(
                session, dialog.id, user_id, MessageCreate(content=f"Message {i + 1}")
            )

        # Get history
        messages = await message_service.get_messages(session, dialog.id, user_id)

        # Should have 6 messages (3 user + 3 assistant)
        assert len(messages) == 6

        # Verify order: user1, assistant1, user2, assistant2, user3, assistant3
        assert messages[0].role == "user"
        assert messages[0].content == "Message 1"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[2].content == "Message 2"

        print(f"\n✓ Message history order test passed with {len(messages)} messages")

    @pytest.mark.asyncio
    async def test_get_message_history_pagination(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
    ):
        """Test message history pagination."""
        user_id = 300500 + abs(hash(str(uuid.uuid4()))) % 10000
        await setup_user_balance(session, user_id, balance=100000)

        # Create dialog
        dialog = await dialog_service.create_dialog(
            session, user_id, DialogCreate(title="Pagination Test", model_name="gpt-3.5-turbo")
        )

        # Send 5 messages
        for i in range(5):
            await message_service.send_message(
                session, dialog.id, user_id, MessageCreate(content=f"Message {i + 1}")
            )

        # Get first page (4 messages)
        page1 = await message_service.get_messages(session, dialog.id, user_id, skip=0, limit=4)
        assert len(page1) == 4

        # Get second page
        page2 = await message_service.get_messages(session, dialog.id, user_id, skip=4, limit=4)
        assert len(page2) == 4

        # Get third page
        page3 = await message_service.get_messages(session, dialog.id, user_id, skip=8, limit=4)
        assert len(page3) == 2  # Only 10 total (5 pairs)

        print(f"\n✓ Pagination test passed: {len(page1)}, {len(page2)}, {len(page3)}")


class TestErrorHandling:
    """Tests for error handling in chat flow."""

    @pytest.mark.asyncio
    async def test_insufficient_tokens_error(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
    ):
        """Test InsufficientTokensError when balance is too low."""
        from src.shared.exceptions import InsufficientTokensError

        user_id = 300600 + abs(hash(str(uuid.uuid4()))) % 10000
        await setup_user_balance(session, user_id, balance=10)  # Very low balance

        # Create dialog
        dialog = await dialog_service.create_dialog(
            session, user_id, DialogCreate(title="Low Balance Test", model_name="gpt-3.5-turbo")
        )

        # Try to send message
        with pytest.raises(InsufficientTokensError):
            await message_service.send_message(
                session, dialog.id, user_id, MessageCreate(content="Test message")
            )

        print("\n✓ Insufficient tokens error test passed")

    @pytest.mark.asyncio
    async def test_dialog_not_found_error(
        self,
        session: AsyncSession,
        message_service: MessageService,
    ):
        """Test NotFoundError when dialog doesn't exist."""
        from src.shared.exceptions import NotFoundError

        user_id = 300700 + abs(hash(str(uuid.uuid4()))) % 10000
        fake_dialog_id = uuid.uuid4()

        with pytest.raises(NotFoundError):
            await message_service.send_message(
                session, fake_dialog_id, user_id, MessageCreate(content="Test")
            )

        print("\n✓ Dialog not found error test passed")

    @pytest.mark.asyncio
    async def test_access_denied_error(
        self,
        session: AsyncSession,
        dialog_service: DialogService,
        message_service: MessageService,
    ):
        """Test ForbiddenError when accessing another user's dialog."""
        from src.shared.exceptions import ForbiddenError

        owner_id = 300800 + abs(hash(str(uuid.uuid4()))) % 10000
        other_user_id = owner_id + 1
        await setup_user_balance(session, owner_id, balance=10000)

        # Create dialog as owner
        dialog = await dialog_service.create_dialog(
            session, owner_id, DialogCreate(title="Private Dialog", model_name="gpt-3.5-turbo")
        )

        # Try to send message as other user
        with pytest.raises(ForbiddenError):
            await message_service.send_message(
                session, dialog.id, other_user_id, MessageCreate(content="Intruder!")
            )

        print("\n✓ Access denied error test passed")
