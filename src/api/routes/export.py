"""Export/Import API routes.

Endpoints:
- GET /export - Export all user dialogs
- POST /import - Import dialogs from JSON
"""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.dependencies import (
    CurrentUserId,
    DbSession,
)
from src.data.models import Dialog, Message
from src.shared.schemas import (
    DialogExport,
    ExportResponse,
    ImportRequest,
    ImportResult,
    MessageExport,
)
from sqlalchemy import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["export"])


@router.get(
    "",
    response_model=ExportResponse,
    summary="Export all dialogs",
    description="Export all user dialogs with messages in JSON format.",
)
async def export_dialogs(
    session: DbSession,
    user_id: CurrentUserId,
) -> ExportResponse:
    """Export all user dialogs with messages.

    Args:
        session: Database session
        user_id: Current user ID from JWT

    Returns:
        ExportResponse with all dialogs and messages
    """
    # Get all dialogs with messages
    result = await session.execute(
        select(Dialog)
        .where(Dialog.user_id == user_id)
        .options(selectinload(Dialog.messages))
        .order_by(Dialog.created_at)
    )
    dialogs = result.scalars().all()

    # Build export data
    dialog_exports = []
    total_messages = 0

    for dialog in dialogs:
        messages = sorted(dialog.messages, key=lambda m: m.created_at)
        message_exports = [
            MessageExport(
                role=msg.role,
                content=msg.content,
                prompt_tokens=msg.prompt_tokens,
                completion_tokens=msg.completion_tokens,
                created_at=msg.created_at,
            )
            for msg in messages
        ]
        total_messages += len(message_exports)

        dialog_exports.append(
            DialogExport(
                id=dialog.id,
                title=dialog.title,
                system_prompt=dialog.system_prompt,
                model_name=dialog.model_name,
                agent_config=dialog.agent_config,
                created_at=dialog.created_at,
                updated_at=dialog.updated_at,
                messages=message_exports,
            )
        )

    logger.info(
        f"Exported {len(dialog_exports)} dialogs with {total_messages} messages",
        extra={"user_id": user_id},
    )

    return ExportResponse(
        exported_at=datetime.now(timezone.utc),
        user_id=user_id,
        dialog_count=len(dialog_exports),
        message_count=total_messages,
        dialogs=dialog_exports,
    )


@router.post(
    "/import",
    response_model=ImportResult,
    summary="Import dialogs",
    description="Import dialogs from JSON export. Creates new dialogs (does not update existing).",
)
async def import_dialogs(
    data: ImportRequest,
    session: DbSession,
    user_id: CurrentUserId,
) -> ImportResult:
    """Import dialogs from export data.

    Args:
        data: Import request with dialogs to import
        session: Database session
        user_id: Current user ID from JWT

    Returns:
        ImportResult with counts and any errors
    """
    from src.config.settings import settings

    dialogs_imported = 0
    messages_imported = 0
    errors: list[str] = []

    for i, dialog_data in enumerate(data.dialogs):
        try:
            # Create new dialog
            dialog = Dialog(
                user_id=user_id,
                title=dialog_data.title,
                system_prompt=dialog_data.system_prompt,
                model_name=dialog_data.model_name or settings.llm_default_model,
                agent_config=dialog_data.agent_config,
            )
            session.add(dialog)
            await session.flush()  # Get dialog ID

            # Create messages
            for msg_data in dialog_data.messages:
                message = Message(
                    dialog_id=dialog.id,
                    role=msg_data.role,
                    content=msg_data.content,
                    prompt_tokens=msg_data.prompt_tokens,
                    completion_tokens=msg_data.completion_tokens,
                )
                session.add(message)
                messages_imported += 1

            dialogs_imported += 1

        except Exception as e:
            errors.append(f"Dialog {i}: {str(e)}")
            logger.warning(f"Import error for dialog {i}: {e}")

    await session.commit()

    logger.info(
        f"Imported {dialogs_imported} dialogs with {messages_imported} messages",
        extra={"user_id": user_id, "errors": len(errors)},
    )

    return ImportResult(
        dialogs_imported=dialogs_imported,
        messages_imported=messages_imported,
        errors=errors,
    )
