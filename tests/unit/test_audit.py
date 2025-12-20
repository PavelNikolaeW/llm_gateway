"""Tests for audit service and schemas."""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.audit_service import (
    AuditAction,
    AuditResourceType,
    AuditService,
)
from src.shared.schemas import AuditLogResponse


class TestAuditConstants:
    """Tests for audit constants."""

    def test_audit_actions(self):
        """Test AuditAction constants exist."""
        assert AuditAction.LOGIN_SUCCESS == "login_success"
        assert AuditAction.DIALOG_CREATE == "dialog_create"
        assert AuditAction.TOKEN_TOPUP == "token_topup"
        assert AuditAction.EXPORT_DIALOGS == "export_dialogs"

    def test_audit_resource_types(self):
        """Test AuditResourceType constants exist."""
        assert AuditResourceType.USER == "user"
        assert AuditResourceType.DIALOG == "dialog"
        assert AuditResourceType.MESSAGE == "message"
        assert AuditResourceType.TOKEN == "token"
        assert AuditResourceType.SYSTEM == "system"


class TestAuditLogResponse:
    """Tests for AuditLogResponse schema."""

    def test_audit_log_response(self):
        """Test AuditLogResponse schema."""
        response = AuditLogResponse(
            id=1,
            user_id=123,
            action="dialog_create",
            resource_type="dialog",
            resource_id="uuid-123",
            details={"title": "Test Dialog"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            created_at=datetime.now(timezone.utc),
        )
        assert response.id == 1
        assert response.user_id == 123
        assert response.action == "dialog_create"
        assert response.resource_type == "dialog"
        assert response.details["title"] == "Test Dialog"

    def test_audit_log_response_nullable_fields(self):
        """Test AuditLogResponse with null fields."""
        response = AuditLogResponse(
            id=1,
            user_id=None,
            action="system_startup",
            resource_type="system",
            resource_id=None,
            details=None,
            ip_address=None,
            user_agent=None,
            created_at=datetime.now(timezone.utc),
        )
        assert response.user_id is None
        assert response.resource_id is None
        assert response.details is None


class TestAuditService:
    """Tests for AuditService."""

    @pytest.mark.asyncio
    async def test_log_creates_audit_entry(self):
        """Test log creates an audit entry."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        service = AuditService()

        with patch("src.domain.audit_service.AuditLog") as mock_audit_log:
            mock_log_instance = MagicMock()
            mock_audit_log.return_value = mock_log_instance

            result = await service.log(
                session=mock_session,
                action="dialog_create",
                resource_type="dialog",
                user_id=123,
                resource_id="uuid-123",
                details={"title": "Test"},
                ip_address="192.168.1.1",
                user_agent="Test Agent",
            )

            mock_session.add.assert_called_once()
            mock_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_count_logs(self):
        """Test count_logs returns count."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = AuditService()
        count = await service.count_logs(session=mock_session)

        assert count == 42

    @pytest.mark.asyncio
    async def test_count_logs_with_filters(self):
        """Test count_logs with filters."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_session.execute = AsyncMock(return_value=mock_result)

        service = AuditService()
        count = await service.count_logs(
            session=mock_session,
            user_id=123,
            action="dialog_create",
            resource_type="dialog",
        )

        assert count == 5
        mock_session.execute.assert_called_once()
