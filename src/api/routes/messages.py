"""Message API routes with SSE streaming.

Endpoints:
- POST /dialogs/{dialog_id}/messages - Send message and stream LLM response
- GET /dialogs/{dialog_id}/messages - Get message history
"""
import logging
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    CurrentUserId,
    DbSession,
    IsAdmin,
    MessageServiceDep,
)
from src.domain.message_service import MessageService
from src.shared.schemas import MessageCreate, MessageResponse, StreamChunk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["messages"])


async def sse_stream(
    service: MessageService,
    session,
    dialog_id: UUID,
    user_id: int,
    data: MessageCreate,
    is_admin: bool,
) -> AsyncGenerator[str, None]:
    """Generate SSE events from message stream.

    Yields Server-Sent Events in the format:
    data: {"content": "...", "done": false}

    data: {"content": "", "done": true, "message_id": "...", "prompt_tokens": N, "completion_tokens": N}
    """
    import json

    try:
        async for chunk in service.send_message_stream(
            session=session,
            dialog_id=dialog_id,
            user_id=user_id,
            data=data,
            is_admin=is_admin,
        ):
            # Convert StreamChunk to SSE event
            event_data = {
                "content": chunk.content,
                "done": chunk.done,
            }

            if chunk.done:
                if chunk.message_id:
                    event_data["message_id"] = str(chunk.message_id)
                if chunk.prompt_tokens is not None:
                    event_data["prompt_tokens"] = chunk.prompt_tokens
                if chunk.completion_tokens is not None:
                    event_data["completion_tokens"] = chunk.completion_tokens

            yield f"data: {json.dumps(event_data)}\n\n"

    except Exception as e:
        # Send error event
        error_data = {"error": str(e), "done": True}
        yield f"data: {json.dumps(error_data)}\n\n"
        raise


@router.post(
    "/dialogs/{dialog_id}/messages",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    summary="Send message and stream LLM response",
    description="Send a message to the dialog and receive the LLM response via Server-Sent Events (SSE).",
    responses={
        200: {
            "description": "SSE stream of LLM response chunks",
            "content": {"text/event-stream": {}},
        },
        402: {"description": "Insufficient tokens"},
        403: {"description": "Access denied to dialog"},
        404: {"description": "Dialog not found"},
        504: {"description": "LLM timeout"},
    },
)
async def send_message_stream(
    dialog_id: UUID,
    data: MessageCreate,
    session: DbSession,
    user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: MessageServiceDep,
) -> StreamingResponse:
    """Send a message and stream the LLM response.

    Args:
        dialog_id: Dialog UUID
        data: Message content
        session: Database session
        user_id: Current user ID from JWT
        is_admin: Whether user is admin
        service: Message service dependency

    Returns:
        SSE stream of response chunks

    Raises:
        NotFoundError: If dialog not found (404)
        ForbiddenError: If user doesn't own dialog (403)
        InsufficientTokensError: If not enough tokens (402)
        LLMTimeoutError: If LLM times out (504)
        LLMError: If LLM error occurs (500)
    """
    return StreamingResponse(
        sse_stream(service, session, dialog_id, user_id, data, is_admin),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post(
    "/dialogs/{dialog_id}/messages/sync",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send message and get response (non-streaming)",
    description="Send a message to the dialog and receive the complete LLM response.",
)
async def send_message_sync(
    dialog_id: UUID,
    data: MessageCreate,
    session: DbSession,
    user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: MessageServiceDep,
) -> MessageResponse:
    """Send a message and get the complete LLM response.

    Args:
        dialog_id: Dialog UUID
        data: Message content
        session: Database session
        user_id: Current user ID from JWT
        is_admin: Whether user is admin
        service: Message service dependency

    Returns:
        Complete assistant message response

    Raises:
        NotFoundError: If dialog not found (404)
        ForbiddenError: If user doesn't own dialog (403)
        InsufficientTokensError: If not enough tokens (402)
        LLMTimeoutError: If LLM times out (504)
        LLMError: If LLM error occurs (500)
    """
    return await service.send_message(
        session=session,
        dialog_id=dialog_id,
        user_id=user_id,
        data=data,
        is_admin=is_admin,
    )


@router.get(
    "/dialogs/{dialog_id}/messages",
    response_model=list[MessageResponse],
    summary="Get message history",
    description="Get all messages for a dialog, ordered by created_at ascending.",
)
async def get_messages(
    dialog_id: UUID,
    session: DbSession,
    user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: MessageServiceDep,
    skip: int = 0,
    limit: int = 100,
) -> list[MessageResponse]:
    """Get messages for a dialog.

    Args:
        dialog_id: Dialog UUID
        session: Database session
        user_id: Current user ID from JWT
        is_admin: Whether user is admin
        service: Message service dependency
        skip: Number of messages to skip
        limit: Maximum messages to return

    Returns:
        List of messages ordered by created_at ascending

    Raises:
        NotFoundError: If dialog not found (404)
        ForbiddenError: If user doesn't own dialog (403)
    """
    return await service.get_messages(
        session=session,
        dialog_id=dialog_id,
        user_id=user_id,
        is_admin=is_admin,
        skip=skip,
        limit=limit,
    )
