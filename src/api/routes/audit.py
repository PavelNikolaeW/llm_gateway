"""Audit log API routes.

Endpoints:
- GET /admin/audit - Get audit logs (admin only)
"""
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Query

from src.api.dependencies import (
    DbSession,
    IsAdmin,
)
from src.domain.audit_service import audit_service
from src.shared.exceptions import ForbiddenError
from src.shared.schemas import AuditLogResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/audit", tags=["admin"])


@router.get(
    "",
    response_model=list[AuditLogResponse],
    summary="Get audit logs",
    description="Get audit logs with optional filtering. Admin only.",
    responses={
        403: {"description": "Access denied - admin required"},
    },
)
async def get_audit_logs(
    session: DbSession,
    is_admin: IsAdmin,
    user_id: int | None = Query(None, description="Filter by user ID"),
    action: str | None = Query(None, description="Filter by action"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    start_date: date | None = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: date | None = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
) -> list[AuditLogResponse]:
    """Get audit logs with filtering.

    Args:
        session: Database session
        is_admin: Whether caller is admin
        user_id: Filter by user ID
        action: Filter by action type
        resource_type: Filter by resource type
        start_date: Start of date range
        end_date: End of date range
        skip: Pagination offset
        limit: Pagination limit

    Returns:
        List of audit log entries

    Raises:
        ForbiddenError: If caller is not admin
    """
    if not is_admin:
        raise ForbiddenError("Admin access required")

    # Convert dates to datetime
    start_dt = None
    end_dt = None
    if start_date:
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    if end_date:
        end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    return await audit_service.get_logs(
        session=session,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        start_date=start_dt,
        end_date=end_dt,
        skip=skip,
        limit=limit,
    )
