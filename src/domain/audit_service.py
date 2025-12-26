"""Audit logging service.

Provides functionality for logging and querying audit events.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import AuditLog
from src.shared.schemas import AuditLogResponse

logger = logging.getLogger(__name__)


class AuditAction:
    """Constants for audit actions."""

    # Authentication
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    TOKEN_REFRESH = "token_refresh"

    # Dialog operations
    DIALOG_CREATE = "dialog_create"
    DIALOG_DELETE = "dialog_delete"

    # Message operations
    MESSAGE_SEND = "message_send"

    # Token operations
    TOKEN_DEDUCT = "token_deduct"
    TOKEN_TOPUP = "token_topup"
    LIMIT_SET = "limit_set"

    # Admin operations
    ADMIN_USER_VIEW = "admin_user_view"
    ADMIN_STATS_VIEW = "admin_stats_view"

    # Export/Import
    EXPORT_DIALOGS = "export_dialogs"
    IMPORT_DIALOGS = "import_dialogs"


class AuditResourceType:
    """Constants for audit resource types."""

    USER = "user"
    DIALOG = "dialog"
    MESSAGE = "message"
    TOKEN = "token"
    SYSTEM = "system"


class AuditService:
    """Service for audit logging operations."""

    async def log(
        self,
        session: AsyncSession,
        action: str,
        resource_type: str,
        user_id: int | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditLog:
        """Log an audit event.

        Args:
            session: Database session
            action: Action performed (e.g., 'dialog_create')
            resource_type: Type of resource (e.g., 'dialog')
            user_id: User who performed the action
            resource_id: ID of the affected resource
            details: Additional details as JSON
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created AuditLog entry
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        session.add(audit_log)
        await session.flush()

        logger.info(
            f"Audit: {action} on {resource_type}",
            extra={
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )

        return audit_log

    async def get_logs(
        self,
        session: AsyncSession,
        user_id: int | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[AuditLogResponse]:
        """Get audit logs with optional filtering.

        Args:
            session: Database session
            user_id: Filter by user ID
            action: Filter by action
            resource_type: Filter by resource type
            start_date: Filter by start date
            end_date: Filter by end date
            skip: Number of records to skip
            limit: Maximum number of records

        Returns:
            List of audit log entries
        """
        query = select(AuditLog)

        if user_id is not None:
            query = query.where(AuditLog.user_id == user_id)
        if action is not None:
            query = query.where(AuditLog.action == action)
        if resource_type is not None:
            query = query.where(AuditLog.resource_type == resource_type)
        if start_date is not None:
            query = query.where(AuditLog.created_at >= start_date)
        if end_date is not None:
            query = query.where(AuditLog.created_at < end_date)

        query = query.order_by(AuditLog.created_at.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        logs = result.scalars().all()

        return [
            AuditLogResponse(
                id=log.id,
                user_id=log.user_id,
                action=log.action,
                resource_type=log.resource_type,
                resource_id=log.resource_id,
                details=log.details,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                created_at=log.created_at,
            )
            for log in logs
        ]

    async def count_logs(
        self,
        session: AsyncSession,
        user_id: int | None = None,
        action: str | None = None,
        resource_type: str | None = None,
    ) -> int:
        """Count audit logs with optional filtering.

        Args:
            session: Database session
            user_id: Filter by user ID
            action: Filter by action
            resource_type: Filter by resource type

        Returns:
            Count of matching logs
        """
        query = select(func.count(AuditLog.id))

        if user_id is not None:
            query = query.where(AuditLog.user_id == user_id)
        if action is not None:
            query = query.where(AuditLog.action == action)
        if resource_type is not None:
            query = query.where(AuditLog.resource_type == resource_type)

        result = await session.execute(query)
        return result.scalar() or 0


# Global audit service instance
audit_service = AuditService()
