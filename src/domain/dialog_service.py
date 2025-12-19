"""Dialog Service - business logic for dialog management."""
import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Dialog
from src.data.repositories import DialogRepository
from src.domain.model_registry import ModelRegistry
from src.shared.exceptions import ForbiddenError, NotFoundError, ValidationError
from src.shared.schemas import DialogCreate, DialogList, DialogResponse

logger = logging.getLogger(__name__)


class DialogService:
    """Service for managing dialogs with business logic and validation."""

    def __init__(self, model_registry: ModelRegistry):
        """Initialize service with model registry."""
        self.model_registry = model_registry
        self.dialog_repo = DialogRepository()

    def _get_default_model(self) -> str:
        """Get default model name from registry.

        Returns the first enabled model, preferring gpt-3.5-turbo if available.
        """
        all_models = self.model_registry.get_all_models()
        if not all_models:
            raise ValidationError("No models available in registry")

        # Prefer gpt-3.5-turbo as default
        for model in all_models:
            if model.name == "gpt-3.5-turbo":
                return model.name

        # Otherwise return first available model
        return all_models[0].name

    def _validate_model(self, model_name: str) -> None:
        """Validate that model exists and is enabled.

        Raises ValidationError if model is invalid.
        """
        if not self.model_registry.model_exists(model_name):
            available = [m.name for m in self.model_registry.get_all_models()]
            raise ValidationError(
                f"Invalid model_name '{model_name}'. "
                f"Available models: {', '.join(available)}"
            )

    async def create_dialog(
        self, session: AsyncSession, user_id: int, data: DialogCreate
    ) -> DialogResponse:
        """Create a new dialog with validation.

        Args:
            session: Database session
            user_id: ID of the user creating the dialog
            data: Dialog creation data

        Returns:
            Created dialog

        Raises:
            ValidationError: If model_name is invalid
        """
        # Use provided model or default
        model_name = data.model_name or self._get_default_model()

        # Validate model exists
        self._validate_model(model_name)

        # Create dialog
        dialog = await self.dialog_repo.create(
            session,
            user_id=user_id,
            title=data.title,
            system_prompt=data.system_prompt,
            model_name=model_name,
            agent_config=data.agent_config,
        )

        await session.commit()

        logger.info(f"Created dialog {dialog.id} for user {user_id} with model {model_name}")

        return DialogResponse.model_validate(dialog)

    async def get_dialog(
        self, session: AsyncSession, dialog_id: UUID, user_id: int, is_admin: bool = False
    ) -> DialogResponse:
        """Retrieve dialog by ID with ownership verification.

        Args:
            session: Database session
            dialog_id: Dialog ID to retrieve
            user_id: ID of the requesting user
            is_admin: Whether user is admin (bypasses ownership check)

        Returns:
            Dialog data

        Raises:
            NotFoundError: If dialog not found
            ForbiddenError: If user doesn't own dialog and is not admin
        """
        dialog = await self.dialog_repo.get_by_id(session, dialog_id)

        if not dialog:
            raise NotFoundError(f"Dialog {dialog_id} not found")

        # Verify ownership (admins can access any dialog)
        if not is_admin and dialog.user_id != user_id:
            raise ForbiddenError(f"Access denied to dialog {dialog_id}")

        return DialogResponse.model_validate(dialog)

    async def list_dialogs(
        self, session: AsyncSession, user_id: int, page: int = 1, page_size: int = 20
    ) -> DialogList:
        """List dialogs for a user with pagination.

        Args:
            session: Database session
            user_id: ID of the user
            page: Page number (1-indexed)
            page_size: Number of items per page (default 20)

        Returns:
            Paginated list of dialogs
        """
        if page < 1:
            page = 1
        if page_size < 1 or page_size > 100:
            page_size = 20

        # Calculate offset
        skip = (page - 1) * page_size

        # Get dialogs
        dialogs = await self.dialog_repo.get_by_user(
            session, user_id=user_id, skip=skip, limit=page_size + 1  # +1 to check has_next
        )

        # Check if there are more pages
        has_next = len(dialogs) > page_size
        if has_next:
            dialogs = dialogs[:page_size]

        # Convert to response models
        items = [DialogResponse.model_validate(d) for d in dialogs]

        # For total, we'd need a count query, but for now we'll estimate
        # In production, add a count query to repository
        total = skip + len(items)
        if has_next:
            total += 1  # At least one more

        return DialogList(
            items=items, total=total, page=page, page_size=page_size, has_next=has_next
        )
