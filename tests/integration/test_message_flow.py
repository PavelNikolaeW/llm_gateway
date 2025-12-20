"""Integration tests for end-to-end message flow.

Tests the complete flow:
1. Create dialog
2. Send message
3. LLM response (mocked)
4. Save messages
5. Deduct tokens

Uses test-specific tables to ensure isolation from production data.
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, response: str = "Hello! I'm an AI assistant.", prompt_tokens: int = 15, completion_tokens: int = 10):
        self.response = response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_count = 0

    async def generate(self, messages: list, model: str, config: dict = None):
        """Generate non-streaming response."""
        self.call_count += 1
        return self.response, self.prompt_tokens, self.completion_tokens

    async def generate_stream(self, messages: list, model: str, config: dict = None):
        """Generate streaming response."""
        self.call_count += 1
        words = self.response.split()
        for i, word in enumerate(words):
            is_final = i == len(words) - 1
            if is_final:
                yield word, True, self.prompt_tokens, self.completion_tokens
            else:
                yield word + " ", False, None, None


class TestEndToEndMessageFlow:
    """End-to-end tests for message flow."""

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

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_complete_message_flow(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
        mock_llm: MockLLMClient,
    ):
        """Test complete flow: create dialog -> send message -> LLM -> save -> deduct."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Step 1: Setup - top up user tokens
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        await session.commit()

        # Step 2: Create dialog
        dialog = await dialog_repo.create(
            session,
            user_id=user_id,
            title="Test Conversation",
            model_name="gpt-4",
            system_prompt="You are a helpful assistant.",
        )
        await session.commit()

        # Step 3: Save user message
        user_message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="user",
            content="Hello, can you help me?",
        )
        await session.flush()

        # Step 4: Call LLM (mocked)
        messages_for_llm = [
            {"role": "system", "content": dialog.system_prompt},
            {"role": "user", "content": user_message.content},
        ]
        response, prompt_tokens, completion_tokens = await mock_llm.generate(
            messages=messages_for_llm,
            model=dialog.model_name,
        )

        # Step 5: Save assistant message
        assistant_message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await session.flush()

        # Step 6: Deduct tokens
        total_tokens = prompt_tokens + completion_tokens
        balance, transaction = await token_service.deduct_tokens(
            session,
            user_id=user_id,
            amount=total_tokens,
            dialog_id=dialog.id,
            message_id=assistant_message.id,
        )
        await session.commit()

        # Verify results
        assert dialog.id is not None
        assert user_message.id is not None
        assert assistant_message.id is not None
        assert assistant_message.content == "Hello! I'm an AI assistant."
        assert assistant_message.prompt_tokens == 15
        assert assistant_message.completion_tokens == 10
        assert balance.balance == 1000 - 25  # 975
        assert transaction.amount == -25

        # Verify messages in dialog
        messages = await message_repo.get_by_dialog(session, dialog.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_message_flow_with_streaming(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
    ):
        """Test message flow with streaming LLM response."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Streaming Test", model_name="gpt-4"
        )
        await session.commit()

        # User message
        user_message = await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Tell me a story"
        )
        await session.flush()

        # Streaming LLM call (mocked)
        mock_llm = MockLLMClient(
            response="Once upon a time in a land far away",
            prompt_tokens=20,
            completion_tokens=10,
        )

        collected_response = []
        final_prompt_tokens = 0
        final_completion_tokens = 0

        async for chunk, is_final, prompt_tokens, completion_tokens in mock_llm.generate_stream(
            messages=[{"role": "user", "content": user_message.content}],
            model=dialog.model_name,
        ):
            collected_response.append(chunk)
            if is_final:
                final_prompt_tokens = prompt_tokens
                final_completion_tokens = completion_tokens

        full_response = "".join(collected_response).strip()

        # Save assistant message
        assistant_message = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content=full_response,
            prompt_tokens=final_prompt_tokens,
            completion_tokens=final_completion_tokens,
        )
        await session.flush()

        # Deduct tokens
        total_tokens = final_prompt_tokens + final_completion_tokens
        await token_service.deduct_tokens(
            session,
            user_id=user_id,
            amount=total_tokens,
            dialog_id=dialog.id,
            message_id=assistant_message.id,
        )
        await session.commit()

        # Verify
        assert "Once upon a time" in full_response
        balance = await token_service.get_balance(session, user_id)
        assert balance.balance == 1000 - 30

    @pytest.mark.asyncio
    async def test_message_flow_insufficient_tokens_aborts(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
        mock_llm: MockLLMClient,
    ):
        """Test that message flow aborts if user has insufficient tokens."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup with small balance
        await token_service.admin_top_up(
            session, user_id=user_id, amount=10, admin_user_id=admin_id, is_admin=True
        )
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Test", model_name="gpt-4"
        )
        await session.commit()

        # User message
        user_message = await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Hello"
        )
        await session.flush()

        # LLM response
        response, prompt_tokens, completion_tokens = await mock_llm.generate(
            messages=[{"role": "user", "content": user_message.content}],
            model=dialog.model_name,
        )

        # Try to save and deduct (should fail)
        assistant_message_id = uuid.uuid4()
        try:
            assistant_message = TestMessage(
                id=assistant_message_id,
                dialog_id=dialog.id,
                role="assistant",
                content=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            session.add(assistant_message)
            await session.flush()

            # This should fail (25 > 10)
            await token_service.deduct_tokens(
                session,
                user_id=user_id,
                amount=prompt_tokens + completion_tokens,
                dialog_id=dialog.id,
                message_id=assistant_message_id,
            )
            await session.commit()
            pytest.fail("Should have raised InsufficientTokensError")
        except Exception:
            await session.rollback()

        # Verify message was not saved
        result = await session.execute(
            select(TestMessage).where(TestMessage.id == assistant_message_id)
        )
        assert result.scalar_one_or_none() is None

        # Verify balance unchanged
        balance = await token_service.get_balance(session, user_id)
        assert balance.balance == 10

    @pytest.mark.asyncio
    async def test_multiple_messages_in_conversation(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
    ):
        """Test multiple back-and-forth messages in a conversation."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup
        await token_service.admin_top_up(
            session, user_id=user_id, amount=5000, admin_user_id=admin_id, is_admin=True
        )
        dialog = await dialog_repo.create(
            session, user_id=user_id, title="Long Conversation", model_name="gpt-4"
        )
        await session.commit()

        # Simulate 5 rounds of conversation
        for i in range(5):
            # User message
            await message_repo.create(
                session, dialog_id=dialog.id, role="user", content=f"Message {i+1}"
            )
            await session.flush()

            # Mock LLM call
            mock_llm = MockLLMClient(
                response=f"Response {i+1}",
                prompt_tokens=10 * (i + 1),  # Increasing context
                completion_tokens=5,
            )
            response, prompt_tokens, completion_tokens = await mock_llm.generate(
                messages=[], model=dialog.model_name
            )

            # Assistant message
            assistant_msg = await message_repo.create(
                session,
                dialog_id=dialog.id,
                role="assistant",
                content=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            await session.flush()

            # Deduct tokens
            await token_service.deduct_tokens(
                session,
                user_id=user_id,
                amount=prompt_tokens + completion_tokens,
                dialog_id=dialog.id,
                message_id=assistant_msg.id,
            )
            await session.commit()

        # Verify all messages saved
        messages = await message_repo.get_by_dialog(session, dialog.id)
        assert len(messages) == 10  # 5 user + 5 assistant

        # Verify tokens deducted
        # Total: (10+5) + (20+5) + (30+5) + (40+5) + (50+5) = 175
        balance = await token_service.get_balance(session, user_id)
        assert balance.balance == 5000 - 175

    @pytest.mark.asyncio
    async def test_message_flow_with_agent_config(
        self,
        session: AsyncSession,
        dialog_repo: TestDialogRepository,
        message_repo: TestMessageRepository,
        token_service: TestTokenService,
        mock_llm: MockLLMClient,
    ):
        """Test message flow with agent configuration."""
        user_id = get_unique_user_id()
        admin_id = 999

        # Setup
        await token_service.admin_top_up(
            session, user_id=user_id, amount=1000, admin_user_id=admin_id, is_admin=True
        )

        # Create dialog with agent config
        agent_config = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
        }
        dialog = await dialog_repo.create(
            session,
            user_id=user_id,
            title="Agent Test",
            model_name="gpt-4",
            system_prompt="You are a creative writer.",
            agent_config=agent_config,
        )
        await session.commit()

        # Verify agent config stored
        fetched = await dialog_repo.get_by_id(session, dialog.id)
        assert fetched.agent_config == agent_config

        # Continue with message flow
        user_msg = await message_repo.create(
            session, dialog_id=dialog.id, role="user", content="Write a poem"
        )
        await session.flush()

        response, prompt_tokens, completion_tokens = await mock_llm.generate(
            messages=[],
            model=dialog.model_name,
            config=dialog.agent_config,
        )

        assistant_msg = await message_repo.create(
            session,
            dialog_id=dialog.id,
            role="assistant",
            content=response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        await session.flush()

        await token_service.deduct_tokens(
            session,
            user_id=user_id,
            amount=prompt_tokens + completion_tokens,
            dialog_id=dialog.id,
            message_id=assistant_msg.id,
        )
        await session.commit()

        # Verify success
        messages = await message_repo.get_by_dialog(session, dialog.id)
        assert len(messages) == 2
