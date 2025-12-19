"""Dialog management API routes.

Endpoints:
- POST /dialogs - Create a new dialog
- GET /dialogs - List user's dialogs
- GET /dialogs/{dialog_id} - Get dialog by ID
"""
import logging
from uuid import UUID

from fastapi import APIRouter, status

from src.api.dependencies import (
    CurrentUserId,
    DbSession,
    DialogServiceDep,
    IsAdmin,
)
from src.shared.schemas import DialogCreate, DialogList, DialogResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dialogs", tags=["dialogs"])


@router.post(
    "",
    response_model=DialogResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new dialog",
    description="Create a new dialog with optional settings (title, system_prompt, model_name, agent_config).",
)
async def create_dialog(
    data: DialogCreate,
    session: DbSession,
    user_id: CurrentUserId,
    service: DialogServiceDep,
) -> DialogResponse:
    """Create a new dialog.

    Args:
        data: Dialog creation data
        session: Database session
        user_id: Current user ID from JWT
        service: Dialog service

    Returns:
        Created dialog with 201 status

    Raises:
        ValidationError: If model_name or agent_config is invalid (400)
        UnauthorizedError: If JWT is invalid (401)
    """
    return await service.create_dialog(session, user_id, data)


@router.get(
    "",
    response_model=DialogList,
    summary="List dialogs",
    description="List dialogs for the current user with pagination.",
)
async def list_dialogs(
    session: DbSession,
    user_id: CurrentUserId,
    service: DialogServiceDep,
    page: int = 1,
    page_size: int = 20,
) -> DialogList:
    """List dialogs for the current user.

    Args:
        session: Database session
        user_id: Current user ID from JWT
        service: Dialog service
        page: Page number (1-indexed)
        page_size: Number of items per page (max 100)

    Returns:
        Paginated list of dialogs
    """
    return await service.list_dialogs(session, user_id, page, page_size)


@router.get(
    "/{dialog_id}",
    response_model=DialogResponse,
    summary="Get dialog by ID",
    description="Retrieve a specific dialog by its ID. Users can only access their own dialogs unless admin.",
)
async def get_dialog(
    dialog_id: UUID,
    session: DbSession,
    user_id: CurrentUserId,
    is_admin: IsAdmin,
    service: DialogServiceDep,
) -> DialogResponse:
    """Get dialog by ID.

    Args:
        dialog_id: Dialog UUID
        session: Database session
        user_id: Current user ID from JWT
        is_admin: Whether user is admin
        service: Dialog service

    Returns:
        Dialog data

    Raises:
        NotFoundError: If dialog not found (404)
        ForbiddenError: If user doesn't own dialog (403)
    """
    return await service.get_dialog(session, dialog_id, user_id, is_admin)
