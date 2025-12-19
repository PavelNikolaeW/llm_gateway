"""Unit tests for DialogService with mocked Data Access."""
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.models import Dialog, Model
from src.domain.dialog_service import DialogService
from src.domain.model_registry import ModelRegistry
from src.shared.exceptions import ForbiddenError, NotFoundError, ValidationError
from src.shared.schemas import DialogCreate


@pytest.fixture
def mock_model_registry():
    """Create mock model registry."""
    registry = MagicMock(spec=ModelRegistry)

    # Mock models
    gpt35 = MagicMock(spec=Model)
    gpt35.name = "gpt-3.5-turbo"
    gpt35.provider = "openai"
    gpt35.enabled = True

    gpt4 = MagicMock(spec=Model)
    gpt4.name = "gpt-4-turbo"
    gpt4.provider = "openai"
    gpt4.enabled = True

    registry.get_all_models.return_value = [gpt35, gpt4]
    registry.model_exists.side_effect = lambda name: name in ["gpt-3.5-turbo", "gpt-4-turbo"]

    return registry


@pytest.fixture
def dialog_service(mock_model_registry):
    """Create DialogService with mocked registry."""
    return DialogService(mock_model_registry)


@pytest.mark.asyncio
async def test_create_dialog_with_model(dialog_service):
    """Test creating dialog with specified model."""
    # Mock session and repository
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    # Mock created dialog
    created_dialog = Dialog(
        id=uuid.uuid4(),
        user_id=1,
        title="Test Dialog",
        system_prompt="You are helpful",
        model_name="gpt-4-turbo",
        agent_config=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    dialog_service.dialog_repo.create.return_value = created_dialog

    # Create dialog
    data = DialogCreate(title="Test Dialog", model_name="gpt-4-turbo")
    result = await dialog_service.create_dialog(session, user_id=1, data=data)

    # Verify
    assert result.id == created_dialog.id
    assert result.model_name == "gpt-4-turbo"
    assert result.title == "Test Dialog"
    dialog_service.dialog_repo.create.assert_called_once()
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_create_dialog_with_default_model(dialog_service):
    """Test creating dialog without model uses default."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    created_dialog = Dialog(
        id=uuid.uuid4(),
        user_id=1,
        title=None,
        system_prompt=None,
        model_name="gpt-3.5-turbo",  # Default
        agent_config=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    dialog_service.dialog_repo.create.return_value = created_dialog

    # Create dialog without specifying model
    data = DialogCreate()
    result = await dialog_service.create_dialog(session, user_id=1, data=data)

    # Should use gpt-3.5-turbo as default
    assert result.model_name == "gpt-3.5-turbo"


@pytest.mark.asyncio
async def test_create_dialog_invalid_model(dialog_service):
    """Test creating dialog with invalid model raises ValidationError."""
    session = AsyncMock()

    data = DialogCreate(model_name="invalid-model")

    with pytest.raises(ValidationError) as exc_info:
        await dialog_service.create_dialog(session, user_id=1, data=data)

    assert "Invalid model_name" in str(exc_info.value.message)
    assert "invalid-model" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_dialog_by_owner(dialog_service):
    """Test retrieving dialog by owner."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    dialog_id = uuid.uuid4()
    dialog = Dialog(
        id=dialog_id,
        user_id=1,
        title="Test",
        system_prompt=None,
        model_name="gpt-3.5-turbo",
        agent_config=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    dialog_service.dialog_repo.get_by_id.return_value = dialog

    # User 1 retrieves their own dialog
    result = await dialog_service.get_dialog(session, dialog_id, user_id=1)

    assert result.id == dialog_id
    assert result.user_id == 1


@pytest.mark.asyncio
async def test_get_dialog_not_found(dialog_service):
    """Test retrieving non-existent dialog raises NotFoundError."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()
    dialog_service.dialog_repo.get_by_id.return_value = None

    dialog_id = uuid.uuid4()

    with pytest.raises(NotFoundError) as exc_info:
        await dialog_service.get_dialog(session, dialog_id, user_id=1)

    assert str(dialog_id) in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_dialog_forbidden_not_owner(dialog_service):
    """Test retrieving dialog by non-owner raises ForbiddenError."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    dialog_id = uuid.uuid4()
    dialog = Dialog(
        id=dialog_id,
        user_id=1,  # Owned by user 1
        title="Test",
        system_prompt=None,
        model_name="gpt-3.5-turbo",
        agent_config=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    dialog_service.dialog_repo.get_by_id.return_value = dialog

    # User 2 tries to access user 1's dialog
    with pytest.raises(ForbiddenError) as exc_info:
        await dialog_service.get_dialog(session, dialog_id, user_id=2, is_admin=False)

    assert "Access denied" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_get_dialog_admin_can_access_any(dialog_service):
    """Test admin can access any dialog."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    dialog_id = uuid.uuid4()
    dialog = Dialog(
        id=dialog_id,
        user_id=1,  # Owned by user 1
        title="Test",
        system_prompt=None,
        model_name="gpt-3.5-turbo",
        agent_config=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    dialog_service.dialog_repo.get_by_id.return_value = dialog

    # Admin (user 999) can access any dialog
    result = await dialog_service.get_dialog(session, dialog_id, user_id=999, is_admin=True)

    assert result.id == dialog_id
    assert result.user_id == 1  # Still owned by user 1, but admin can see it


@pytest.mark.asyncio
async def test_list_dialogs(dialog_service):
    """Test listing dialogs with pagination."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    # Mock 3 dialogs
    dialogs = [
        Dialog(
            id=uuid.uuid4(),
            user_id=1,
            title=f"Dialog {i}",
            system_prompt=None,
            model_name="gpt-3.5-turbo",
            agent_config=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        for i in range(3)
    ]
    dialog_service.dialog_repo.get_by_user.return_value = dialogs

    # List dialogs
    result = await dialog_service.list_dialogs(session, user_id=1, page=1, page_size=20)

    assert len(result.items) == 3
    assert result.page == 1
    assert result.page_size == 20
    assert result.has_next is False


@pytest.mark.asyncio
async def test_list_dialogs_pagination(dialog_service):
    """Test listing dialogs checks for next page."""
    session = AsyncMock()
    dialog_service.dialog_repo = AsyncMock()

    # Return page_size + 1 dialogs to indicate there's a next page
    dialogs = [
        Dialog(
            id=uuid.uuid4(),
            user_id=1,
            title=f"Dialog {i}",
            system_prompt=None,
            model_name="gpt-3.5-turbo",
            agent_config=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        for i in range(21)  # 20 + 1
    ]
    dialog_service.dialog_repo.get_by_user.return_value = dialogs

    result = await dialog_service.list_dialogs(session, user_id=1, page=1, page_size=20)

    assert len(result.items) == 20  # Should trim to page_size
    assert result.has_next is True  # Should indicate more pages
