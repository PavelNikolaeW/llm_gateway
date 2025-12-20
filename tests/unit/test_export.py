"""Tests for export/import schemas."""
import pytest
from datetime import datetime, timezone
from uuid import uuid4

from src.shared.schemas import (
    DialogExport,
    DialogImport,
    ExportResponse,
    ImportRequest,
    ImportResult,
    MessageExport,
)


class TestExportSchemas:
    """Tests for export schemas."""

    def test_message_export(self):
        """Test MessageExport schema."""
        msg = MessageExport(
            role="user",
            content="Hello",
            prompt_tokens=10,
            completion_tokens=None,
            created_at=datetime.now(timezone.utc),
        )
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.prompt_tokens == 10

    def test_dialog_export(self):
        """Test DialogExport schema."""
        dialog = DialogExport(
            id=uuid4(),
            title="Test Dialog",
            system_prompt="You are helpful",
            model_name="gpt-4",
            agent_config={"temperature": 0.7},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            messages=[
                MessageExport(
                    role="user",
                    content="Hello",
                    created_at=datetime.now(timezone.utc),
                )
            ],
        )
        assert dialog.title == "Test Dialog"
        assert len(dialog.messages) == 1

    def test_export_response(self):
        """Test ExportResponse schema."""
        response = ExportResponse(
            exported_at=datetime.now(timezone.utc),
            user_id=123,
            dialog_count=5,
            message_count=100,
            dialogs=[],
        )
        assert response.version == "1.0"
        assert response.dialog_count == 5
        assert response.message_count == 100


class TestImportSchemas:
    """Tests for import schemas."""

    def test_dialog_import(self):
        """Test DialogImport schema."""
        dialog = DialogImport(
            title="Imported Dialog",
            system_prompt="Test prompt",
            model_name="gpt-4",
            messages=[
                MessageExport(
                    role="user",
                    content="Hello",
                    created_at=datetime.now(timezone.utc),
                ),
                MessageExport(
                    role="assistant",
                    content="Hi there!",
                    created_at=datetime.now(timezone.utc),
                ),
            ],
        )
        assert dialog.title == "Imported Dialog"
        assert len(dialog.messages) == 2

    def test_dialog_import_defaults(self):
        """Test DialogImport with defaults."""
        dialog = DialogImport()
        assert dialog.title is None
        assert dialog.system_prompt is None
        assert dialog.model_name is None
        assert dialog.messages == []

    def test_import_request(self):
        """Test ImportRequest schema."""
        request = ImportRequest(
            dialogs=[
                DialogImport(title="Dialog 1"),
                DialogImport(title="Dialog 2"),
            ]
        )
        assert len(request.dialogs) == 2

    def test_import_result(self):
        """Test ImportResult schema."""
        result = ImportResult(
            dialogs_imported=5,
            messages_imported=50,
            errors=["Error 1", "Error 2"],
        )
        assert result.dialogs_imported == 5
        assert result.messages_imported == 50
        assert len(result.errors) == 2

    def test_import_result_defaults(self):
        """Test ImportResult with defaults."""
        result = ImportResult(
            dialogs_imported=0,
            messages_imported=0,
        )
        assert result.errors == []
