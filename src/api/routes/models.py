"""Models API routes - list available LLM models."""

from fastapi import APIRouter

from src.domain.model_registry import model_registry
from src.shared.schemas import ModelMetadata

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelMetadata])
async def list_models() -> list[ModelMetadata]:
    """Get list of all available LLM models.

    Returns models with their metadata including:
    - name: Model identifier for API calls
    - provider: LLM provider (openai, anthropic, etc.)
    - context_window: Maximum context size in tokens
    - cost_per_1k_prompt_tokens: Cost per 1000 input tokens
    - cost_per_1k_completion_tokens: Cost per 1000 output tokens
    - enabled: Whether the model is currently available
    """
    models = model_registry.get_all_models()

    return [
        ModelMetadata(
            name=model.name,
            provider=model.provider,
            cost_per_1k_prompt_tokens=float(model.cost_per_1k_prompt_tokens),
            cost_per_1k_completion_tokens=float(model.cost_per_1k_completion_tokens),
            context_window=model.context_window,
            enabled=model.enabled,
        )
        for model in models
    ]


@router.get("/{model_name}", response_model=ModelMetadata)
async def get_model(model_name: str) -> ModelMetadata:
    """Get metadata for a specific model.

    Args:
        model_name: Name of the model to retrieve

    Returns:
        Model metadata

    Raises:
        400: If model is unknown or disabled
    """
    return model_registry.get_model_metadata(model_name)
