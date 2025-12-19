"""Model Registry - loads and caches available LLM models from database."""
import logging
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Model
from src.data.repositories import ModelRepository

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available LLM models.

    Loads models from database on startup and provides methods to query them.
    Models are cached in memory for fast access.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, Model] = {}
        self._loaded = False

    async def load_models(self, session: AsyncSession) -> None:
        """Load all enabled models from database.

        This should be called on application startup.
        """
        repo = ModelRepository()
        models = await repo.get_enabled_models(session)

        self._models = {model.name: model for model in models}
        self._loaded = True

        logger.info(f"Loaded {len(self._models)} models from database")
        for model_name in self._models.keys():
            logger.info(f"  - {model_name}")

    def get_model(self, name: str) -> Model | None:
        """Get model by name.

        Returns None if model not found or not enabled.
        """
        if not self._loaded:
            logger.warning("ModelRegistry not loaded yet, call load_models() first")
            return None

        return self._models.get(name)

    def get_all_models(self) -> list[Model]:
        """Get all enabled models."""
        return list(self._models.values())

    def get_models_by_provider(self, provider: str) -> list[Model]:
        """Get all models for a specific provider."""
        return [model for model in self._models.values() if model.provider == provider]

    def is_loaded(self) -> bool:
        """Check if models have been loaded."""
        return self._loaded

    def model_exists(self, name: str) -> bool:
        """Check if a model exists and is enabled."""
        return name in self._models


# Global model registry instance
model_registry = ModelRegistry()
