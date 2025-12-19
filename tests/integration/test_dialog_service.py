"""Integration tests for DialogService with real database."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.dialog_service import DialogService
from src.domain.model_registry import ModelRegistry
from src.shared.exceptions import ForbiddenError, NotFoundError, ValidationError
from src.shared.schemas import DialogCreate


@pytest.fixture(scope="function")
async def dialog_service(session: AsyncSession):
    """Create dialog service with loaded registry."""
    registry = ModelRegistry()
    await registry.load_models(session)
    return DialogService(registry)


@pytest.mark.asyncio
async def test_create_dialog_integration(session: AsyncSession, dialog_service: DialogService):
    """Test creating dialog with database."""
    data = DialogCreate(
        title="Integration Test Dialog", system_prompt="Test prompt", model_name="gpt-3.5-turbo"
    )

    result = await dialog_service.create_dialog(session, user_id=100, data=data)

    assert result.id is not None
    assert result.user_id == 100
    assert result.title == "Integration Test Dialog"
    assert result.model_name == "gpt-3.5-turbo"
    assert result.system_prompt == "Test prompt"

    print(f"\n✓ Created dialog {result.id}")


@pytest.mark.asyncio
async def test_create_dialog_with_default_model(
    session: AsyncSession, dialog_service: DialogService
):
    """Test creating dialog without model uses default."""
    data = DialogCreate(title="Default Model Test")

    result = await dialog_service.create_dialog(session, user_id=101, data=data)

    assert result.model_name == "gpt-3.5-turbo"  # Default model
    print(f"\n✓ Created dialog with default model: {result.model_name}")


@pytest.mark.asyncio
async def test_create_dialog_with_agent_config(
    session: AsyncSession, dialog_service: DialogService
):
    """Test creating dialog with agent configuration."""
    data = DialogCreate(
        title="Agent Config Test",
        model_name="claude-3-sonnet",
        agent_config={"temperature": 0.7, "max_tokens": 1000},
    )

    result = await dialog_service.create_dialog(session, user_id=102, data=data)

    assert result.agent_config == {"temperature": 0.7, "max_tokens": 1000}
    assert result.model_name == "claude-3-sonnet"
    print(f"\n✓ Created dialog with agent config: {result.agent_config}")


@pytest.mark.asyncio
async def test_create_dialog_invalid_model(session: AsyncSession, dialog_service: DialogService):
    """Test creating dialog with invalid model raises ValidationError."""
    data = DialogCreate(model_name="non-existent-model")

    with pytest.raises(ValidationError) as exc_info:
        await dialog_service.create_dialog(session, user_id=103, data=data)

    assert "Invalid model_name" in exc_info.value.message
    assert "non-existent-model" in exc_info.value.message
    print(f"\n✓ Validation error for invalid model: {exc_info.value.message}")


@pytest.mark.asyncio
async def test_get_dialog_by_owner(session: AsyncSession, dialog_service: DialogService):
    """Test retrieving dialog by owner."""
    # Create dialog
    create_data = DialogCreate(title="Owner Test", model_name="gpt-4-turbo")
    created = await dialog_service.create_dialog(session, user_id=104, data=create_data)

    # Retrieve by owner
    retrieved = await dialog_service.get_dialog(session, created.id, user_id=104)

    assert retrieved.id == created.id
    assert retrieved.user_id == 104
    assert retrieved.title == "Owner Test"
    print(f"\n✓ Retrieved dialog {retrieved.id} by owner")


@pytest.mark.asyncio
async def test_get_dialog_not_found(session: AsyncSession, dialog_service: DialogService):
    """Test retrieving non-existent dialog raises NotFoundError."""
    import uuid

    fake_id = uuid.uuid4()

    with pytest.raises(NotFoundError) as exc_info:
        await dialog_service.get_dialog(session, fake_id, user_id=105)

    assert str(fake_id) in exc_info.value.message
    print(f"\n✓ NotFoundError for non-existent dialog")


@pytest.mark.asyncio
async def test_get_dialog_forbidden(session: AsyncSession, dialog_service: DialogService):
    """Test retrieving dialog by non-owner raises ForbiddenError."""
    # User 106 creates dialog
    create_data = DialogCreate(title="Forbidden Test")
    created = await dialog_service.create_dialog(session, user_id=106, data=create_data)

    # User 107 tries to access it
    with pytest.raises(ForbiddenError) as exc_info:
        await dialog_service.get_dialog(session, created.id, user_id=107, is_admin=False)

    assert "Access denied" in exc_info.value.message
    print(f"\n✓ ForbiddenError for unauthorized access")


@pytest.mark.asyncio
async def test_get_dialog_admin_access(session: AsyncSession, dialog_service: DialogService):
    """Test admin can access any dialog."""
    # User 108 creates dialog
    create_data = DialogCreate(title="Admin Access Test")
    created = await dialog_service.create_dialog(session, user_id=108, data=create_data)

    # Admin (user 999) can access it
    retrieved = await dialog_service.get_dialog(session, created.id, user_id=999, is_admin=True)

    assert retrieved.id == created.id
    assert retrieved.user_id == 108  # Still owned by 108
    print(f"\n✓ Admin accessed dialog owned by different user")


@pytest.mark.asyncio
async def test_list_dialogs(session: AsyncSession, dialog_service: DialogService):
    """Test listing dialogs with pagination."""
    # Create multiple dialogs for user 109
    for i in range(5):
        data = DialogCreate(title=f"List Test {i}")
        await dialog_service.create_dialog(session, user_id=109, data=data)

    # List dialogs
    result = await dialog_service.list_dialogs(session, user_id=109, page=1, page_size=3)

    assert len(result.items) == 3
    assert result.page == 1
    assert result.page_size == 3
    assert result.has_next is True  # Should have more
    assert all(d.user_id == 109 for d in result.items)

    # Most recent first (ordered by created_at desc)
    assert result.items[0].title == "List Test 4"  # Last created

    print(f"\n✓ Listed {len(result.items)} dialogs, has_next={result.has_next}")


@pytest.mark.asyncio
async def test_list_dialogs_empty(session: AsyncSession, dialog_service: DialogService):
    """Test listing dialogs for user with no dialogs."""
    result = await dialog_service.list_dialogs(session, user_id=999999, page=1, page_size=20)

    assert len(result.items) == 0
    assert result.has_next is False
    print(f"\n✓ Empty list for user with no dialogs")


@pytest.mark.asyncio
async def test_list_dialogs_pagination(session: AsyncSession, dialog_service: DialogService):
    """Test pagination works correctly."""
    import uuid

    # Use unique user_id to avoid conflicts with previous test runs
    user_id = 110000 + abs(hash(str(uuid.uuid4()))) % 10000

    # Create 25 dialogs
    for i in range(25):
        data = DialogCreate(title=f"Page Test {i}")
        await dialog_service.create_dialog(session, user_id=user_id, data=data)

    # Page 1
    page1 = await dialog_service.list_dialogs(session, user_id=user_id, page=1, page_size=20)
    assert len(page1.items) == 20
    assert page1.has_next is True

    # Page 2
    page2 = await dialog_service.list_dialogs(session, user_id=user_id, page=2, page_size=20)
    assert len(page2.items) == 5
    assert page2.has_next is False

    print(f"\n✓ Pagination: page1={len(page1.items)}, page2={len(page2.items)}")


print("\n✅ All DialogService integration tests completed")
