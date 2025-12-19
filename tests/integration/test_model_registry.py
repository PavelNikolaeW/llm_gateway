"""Integration tests for Model Registry."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.model_registry import ModelRegistry


@pytest.mark.asyncio
async def test_model_registry_loads_from_database(session: AsyncSession):
    """Test that ModelRegistry loads models from database on startup.

    This test demonstrates:
    - Model Registry initialization
    - Loading models from database
    - Querying models by name and provider
    """
    # Initialize registry
    registry = ModelRegistry()
    assert not registry.is_loaded()

    # Load models from database
    await registry.load_models(session)
    assert registry.is_loaded()

    # Verify all seeded models are loaded
    all_models = registry.get_all_models()
    model_names = {model.name for model in all_models}

    # Check that seed models exist (at least these 4)
    expected_models = {"gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"}
    assert expected_models.issubset(model_names), f"Missing models: {expected_models - model_names}"

    print(f"\n✓ Loaded {len(all_models)} models from database")
    for model in all_models:
        if model.name in expected_models:
            print(f"  - {model.name} ({model.provider})")

    # Test get_model by name
    gpt4 = registry.get_model("gpt-4-turbo")
    assert gpt4 is not None
    assert gpt4.provider == "openai"
    assert gpt4.enabled is True
    assert gpt4.context_window == 128000
    print(f"\n✓ Retrieved gpt-4-turbo: ${gpt4.cost_per_1k_prompt_tokens}/1k prompt tokens")

    claude = registry.get_model("claude-3-opus")
    assert claude is not None
    assert claude.provider == "anthropic"
    assert claude.enabled is True
    assert claude.context_window == 200000
    print(f"✓ Retrieved claude-3-opus: ${claude.cost_per_1k_prompt_tokens}/1k prompt tokens")

    # Test get_models_by_provider
    openai_models = registry.get_models_by_provider("openai")
    openai_model_names = {model.name for model in openai_models}
    assert "gpt-4-turbo" in openai_model_names
    assert "gpt-3.5-turbo" in openai_model_names
    print(f"\n✓ Found {len(openai_models)} OpenAI models")

    anthropic_models = registry.get_models_by_provider("anthropic")
    anthropic_model_names = {model.name for model in anthropic_models}
    assert "claude-3-opus" in anthropic_model_names
    assert "claude-3-sonnet" in anthropic_model_names
    print(f"✓ Found {len(anthropic_models)} Anthropic models")

    # Test model_exists
    assert registry.model_exists("gpt-4-turbo")
    assert registry.model_exists("claude-3-opus")
    assert not registry.model_exists("non-existent-model")
    print("\n✓ model_exists() working correctly")

    print("\n✅ Model Registry integration test completed")
    print("   ✓ Models loaded from database")
    print("   ✓ Query by name working")
    print("   ✓ Query by provider working")
    print("   ✓ All seed models present")
