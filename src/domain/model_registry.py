"""Model Registry - loads and caches available LLM models from database.

Reload on SIGHUP (v1 documentation):
    In v1, the registry is loaded once on startup. To reload models after
    database changes, restart the application.

    Future versions may support SIGHUP signal handling:
        import signal
        signal.signal(signal.SIGHUP, lambda s, f: asyncio.create_task(registry.reload()))

    Or config file watching using watchdog or similar libraries.
"""

import logging
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import Model
from src.data.repositories import ModelRepository
from src.shared.exceptions import ValidationError
from src.shared.schemas import CostEstimate, ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for available LLM models.

    Loads models from database on startup and provides methods to query them.
    Models are cached in memory for fast access.

    Features:
    - Load models from DB on startup
    - Validate model names (returns 400 if unknown/disabled)
    - Calculate estimated cost from model pricing
    - Provide model metadata (provider, cost, context_window)
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

    async def reload(self, session: AsyncSession) -> None:
        """Reload models from database.

        Can be called to refresh the registry after model changes.
        """
        logger.info("Reloading model registry...")
        await self.load_models(session)

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

    def validate_model(self, name: str) -> None:
        """Validate that a model exists and is enabled.

        Args:
            name: Model name to validate

        Raises:
            ValidationError: If model is unknown or disabled (HTTP 400)
        """
        if not self._loaded:
            raise ValidationError("Model registry not loaded")

        if name not in self._models:
            available = list(self._models.keys())
            raise ValidationError(
                f"Unknown model '{name}'. Available models: {', '.join(available)}"
            )

    def get_model_metadata(self, name: str) -> ModelMetadata:
        """Get model metadata as a typed response.

        Args:
            name: Model name

        Returns:
            ModelMetadata with provider, cost, context_window

        Raises:
            ValidationError: If model is unknown
        """
        self.validate_model(name)
        model = self._models[name]

        return ModelMetadata(
            name=model.name,
            provider=model.provider,
            cost_per_1k_prompt_tokens=float(model.cost_per_1k_prompt_tokens),
            cost_per_1k_completion_tokens=float(model.cost_per_1k_completion_tokens),
            context_window=model.context_window,
            enabled=model.enabled,
        )

    def estimate_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
    ) -> CostEstimate:
        """Calculate estimated cost for a request.

        Uses model pricing from database:
        - cost = (prompt_tokens / 1000) * cost_per_1k_prompt_tokens
                + (completion_tokens / 1000) * cost_per_1k_completion_tokens

        Args:
            model_name: Name of the model
            prompt_tokens: Number of prompt/input tokens
            completion_tokens: Number of completion/output tokens (0 for estimation before request)

        Returns:
            CostEstimate with breakdown of costs

        Raises:
            ValidationError: If model is unknown
        """
        self.validate_model(model_name)
        model = self._models[model_name]

        prompt_cost = (prompt_tokens / 1000) * float(model.cost_per_1k_prompt_tokens)
        completion_cost = (completion_tokens / 1000) * float(model.cost_per_1k_completion_tokens)
        total_cost = prompt_cost + completion_cost

        return CostEstimate(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_cost=round(prompt_cost, 6),
            completion_cost=round(completion_cost, 6),
            total_cost=round(total_cost, 6),
        )

    def estimate_tokens(self, text: str, model_name: str | None = None) -> int:
        """Estimate token count for text.

        Uses a simple approximation: ~4 characters per token for English text.
        For more accurate counts, use tiktoken library with specific model encoding.

        Args:
            text: Text to estimate tokens for
            model_name: Optional model name (for future model-specific tokenizers)

        Returns:
            Estimated token count
        """
        # Simple approximation: ~4 characters per token
        # This is a rough estimate for English text
        # For production, consider using tiktoken for accurate counts
        return max(1, len(text) // 4)


# Global model registry instance
model_registry = ModelRegistry()
