"""Message Service - handles message flow with LLM integration.

Implements the complete message handling flow:
1. Append user message to dialog
2. Emit Message Sent event
3. Call LLM provider
4. Stream/return assistant response
5. Save assistant message with token counts
6. Deduct tokens
7. Emit events
"""

import logging
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Dialog
from src.data.repositories import DialogRepository, MessageRepository
from src.domain.token_service import TokenService
from src.shared.exceptions import (
    ForbiddenError,
    InsufficientTokensError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
)
from src.shared.schemas import (
    LLMResponseEvent,
    MessageCreate,
    MessageResponse,
    MessageSentEvent,
    StreamChunk,
)

logger = logging.getLogger(__name__)

# Event handler type
MessageEventHandler = Callable[[MessageSentEvent | LLMResponseEvent], None]


class LLMProvider(Protocol):
    """Protocol for LLM provider interface.

    This will be implemented by the Integrations layer.
    """

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

        Returns:
            Tuple of (response_content, prompt_tokens, completion_tokens)

        Raises:
            LLMTimeoutError: If request times out
            LLMError: If LLM returns an error
        """
        ...

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        """Generate a streaming response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

        Yields:
            Tuples of (content_chunk, is_done, prompt_tokens, completion_tokens)
            Token counts are only provided in the final chunk (is_done=True)

        Raises:
            LLMTimeoutError: If request times out
            LLMError: If LLM returns an error
        """
        ...


class MessageService:
    """Service for handling message flow with LLM integration.

    Features:
    - Append user message to dialog
    - Call LLM provider (streaming or non-streaming)
    - Save assistant message with token counts
    - Deduct tokens after successful response
    - Emit events for message flow
    - Atomic transaction for save + deduct
    """

    def __init__(
        self,
        token_service: TokenService,
        llm_provider: LLMProvider | None = None,
    ):
        """Initialize message service.

        Args:
            token_service: Service for token operations
            llm_provider: LLM provider implementation (optional, for testing)
        """
        self.token_service = token_service
        self.llm_provider = llm_provider
        self.dialog_repo = DialogRepository()
        self.message_repo = MessageRepository()
        self._event_handlers: list[MessageEventHandler] = []

    def register_event_handler(self, handler: MessageEventHandler) -> None:
        """Register an event handler for message events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: MessageSentEvent | LLMResponseEvent) -> None:
        """Emit event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def _get_dialog(
        self,
        session: AsyncSession,
        dialog_id: UUID,
        user_id: int,
        is_admin: bool = False,
    ) -> Dialog:
        """Get dialog with ownership verification.

        Raises:
            NotFoundError: If dialog not found
            ForbiddenError: If user doesn't own dialog
        """
        dialog = await self.dialog_repo.get_by_id(session, dialog_id)

        if not dialog:
            raise NotFoundError(f"Dialog {dialog_id} not found")

        if not is_admin and dialog.user_id != user_id:
            raise ForbiddenError(f"Access denied to dialog {dialog_id}")

        return dialog

    async def _build_messages_for_llm(
        self,
        session: AsyncSession,
        dialog: Dialog,
    ) -> list[dict[str, str]]:
        """Build messages list for LLM API.

        Includes system prompt (if any) and conversation history.
        User message should already be saved to DB before calling this.
        """
        messages: list[dict[str, str]] = []

        # Add system prompt if present
        if dialog.system_prompt:
            messages.append({"role": "system", "content": dialog.system_prompt})

        # Get conversation history (includes the just-saved user message)
        history = await self.message_repo.get_by_dialog(session, dialog.id)
        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    async def send_message(
        self,
        session: AsyncSession,
        dialog_id: UUID,
        user_id: int,
        data: MessageCreate,
        config: dict[str, Any] | None = None,
        is_admin: bool = False,
    ) -> MessageResponse:
        """Send a message and get LLM response (non-streaming).

        Args:
            session: Database session
            dialog_id: Dialog ID
            user_id: User ID
            data: Message data
            config: Optional LLM config overrides
            is_admin: Whether user is admin

        Returns:
            Assistant message response

        Raises:
            NotFoundError: If dialog not found (404)
            ForbiddenError: If access denied (403)
            InsufficientTokensError: If not enough tokens (402)
            LLMTimeoutError: If LLM times out (504)
            LLMError: If LLM error occurs (500)
        """
        start_time = time.time()

        # Get dialog with ownership check
        dialog = await self._get_dialog(session, dialog_id, user_id, is_admin)

        # Check balance before proceeding (estimate based on input)
        estimated_tokens = len(data.content) // 4 + 100  # Rough estimate
        has_balance = await self.token_service.check_balance(session, user_id, estimated_tokens)
        if not has_balance:
            raise InsufficientTokensError(
                f"Insufficient tokens. Estimated cost: {estimated_tokens}"
            )

        # Save user message
        user_message = await self.message_repo.create_user_message(session, dialog.id, data.content)
        await session.flush()

        # Emit Message Sent event
        self._emit_event(
            MessageSentEvent(
                dialog_id=dialog.id,
                user_id=user_id,
                message_id=user_message.id,
                content_length=len(data.content),
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Build messages for LLM (user message already saved above)
        messages = await self._build_messages_for_llm(session, dialog)

        # Call LLM
        if self.llm_provider is None:
            raise LLMError("LLM provider not configured")

        try:
            response_content, prompt_tokens, completion_tokens = await self.llm_provider.generate(
                messages=messages,
                model=dialog.model_name,
                config=config,
            )
        except LLMTimeoutError:
            # Rollback user message on LLM failure
            await session.rollback()
            raise
        except LLMError:
            # Rollback user message on LLM failure
            await session.rollback()
            raise
        except Exception as e:
            await session.rollback()
            raise LLMError(f"LLM error: {e}")

        # Estimate tokens if provider returned 0 (e.g. LM Studio doesn't support usage)
        if prompt_tokens == 0 and completion_tokens == 0:
            # Rough estimate: ~4 characters per token
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(response_content) // 4)
            logger.debug(
                f"Estimated tokens: prompt={prompt_tokens}, completion={completion_tokens}"
            )

        # Save assistant message
        assistant_message = await self.message_repo.create_assistant_message(
            session,
            dialog.id,
            response_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Deduct tokens
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            try:
                await self.token_service.deduct_tokens(
                    session,
                    user_id,
                    total_tokens,
                    dialog.id,
                    assistant_message.id,
                )
            except InsufficientTokensError:
                # This shouldn't happen since we checked, but handle it
                await session.rollback()
                raise

        # Commit transaction (atomic: messages + token deduction)
        await session.commit()

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Emit LLM Response event
        self._emit_event(
            LLMResponseEvent(
                dialog_id=dialog.id,
                user_id=user_id,
                message_id=assistant_message.id,
                model=dialog.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc),
            )
        )

        logger.info(
            f"Message sent: dialog={dialog_id}, user={user_id}, "
            f"tokens={total_tokens}, latency={latency_ms}ms"
        )

        return MessageResponse.model_validate(assistant_message)

    async def send_message_stream(
        self,
        session: AsyncSession,
        dialog_id: UUID,
        user_id: int,
        data: MessageCreate,
        config: dict[str, Any] | None = None,
        is_admin: bool = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Send a message and stream LLM response.

        Args:
            session: Database session
            dialog_id: Dialog ID
            user_id: User ID
            data: Message data
            config: Optional LLM config overrides
            is_admin: Whether user is admin

        Yields:
            StreamChunk objects with response content

        Raises:
            NotFoundError: If dialog not found (404)
            ForbiddenError: If access denied (403)
            InsufficientTokensError: If not enough tokens (402)
            LLMTimeoutError: If LLM times out (504)
            LLMError: If LLM error occurs (500)
        """
        start_time = time.time()

        # Get dialog with ownership check
        dialog = await self._get_dialog(session, dialog_id, user_id, is_admin)

        # Check balance before proceeding
        estimated_tokens = len(data.content) // 4 + 100
        has_balance = await self.token_service.check_balance(session, user_id, estimated_tokens)
        if not has_balance:
            raise InsufficientTokensError(
                f"Insufficient tokens. Estimated cost: {estimated_tokens}"
            )

        # Save user message
        user_message = await self.message_repo.create_user_message(session, dialog.id, data.content)
        await session.flush()

        # Emit Message Sent event
        self._emit_event(
            MessageSentEvent(
                dialog_id=dialog.id,
                user_id=user_id,
                message_id=user_message.id,
                content_length=len(data.content),
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Build messages for LLM (user message already saved above)
        messages = await self._build_messages_for_llm(session, dialog)

        # Call LLM with streaming
        if self.llm_provider is None:
            raise LLMError("LLM provider not configured")

        full_response = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for chunk, done, p_tokens, c_tokens in self.llm_provider.generate_stream(
                messages=messages,
                model=dialog.model_name,
                config=config,
            ):
                full_response += chunk

                if done:
                    prompt_tokens = p_tokens or 0
                    completion_tokens = c_tokens or 0

                yield StreamChunk(
                    content=chunk,
                    done=done,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                )

        except LLMTimeoutError:
            await session.rollback()
            raise
        except LLMError:
            await session.rollback()
            raise
        except Exception as e:
            await session.rollback()
            raise LLMError(f"LLM streaming error: {e}")

        # Estimate tokens if provider returned 0 (e.g. LM Studio doesn't support usage)
        if prompt_tokens == 0 and completion_tokens == 0:
            # Rough estimate: ~4 characters per token
            prompt_text = " ".join(m["content"] for m in messages)
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(full_response) // 4)
            logger.debug(
                f"Estimated tokens: prompt={prompt_tokens}, completion={completion_tokens}"
            )

        # Save assistant message
        assistant_message = await self.message_repo.create_assistant_message(
            session,
            dialog.id,
            full_response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # Deduct tokens
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0:
            try:
                await self.token_service.deduct_tokens(
                    session,
                    user_id,
                    total_tokens,
                    dialog.id,
                    assistant_message.id,
                )
            except InsufficientTokensError:
                await session.rollback()
                raise

        # Commit transaction
        await session.commit()

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Emit LLM Response event
        self._emit_event(
            LLMResponseEvent(
                dialog_id=dialog.id,
                user_id=user_id,
                message_id=assistant_message.id,
                model=dialog.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                timestamp=datetime.now(timezone.utc),
            )
        )

        # Yield final chunk with message ID
        yield StreamChunk(
            content="",
            done=True,
            message_id=assistant_message.id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def get_messages(
        self,
        session: AsyncSession,
        dialog_id: UUID,
        user_id: int,
        is_admin: bool = False,
        skip: int = 0,
        limit: int = 100,
    ) -> list[MessageResponse]:
        """Get messages for a dialog.

        Args:
            session: Database session
            dialog_id: Dialog ID
            user_id: User ID
            is_admin: Whether user is admin
            skip: Number of messages to skip
            limit: Maximum messages to return

        Returns:
            List of messages ordered by created_at asc
        """
        # Verify dialog access
        await self._get_dialog(session, dialog_id, user_id, is_admin)

        # Get messages
        messages = await self.message_repo.get_by_dialog(session, dialog_id, skip, limit)

        return [MessageResponse.model_validate(m) for m in messages]
