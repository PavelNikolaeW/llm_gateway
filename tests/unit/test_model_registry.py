"""Unit tests for ModelRegistry with mocked data."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.data.models import Model
from src.domain.model_registry import ModelRegistry
from src.shared.exceptions import ValidationError


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    gpt35 = MagicMock(spec=Model)
    gpt35.name = "gpt-3.5-turbo"
    gpt35.provider = "openai"
    gpt35.cost_per_1k_prompt_tokens = Decimal("0.0015")
    gpt35.cost_per_1k_completion_tokens = Decimal("0.002")
    gpt35.context_window = 16385
    gpt35.enabled = True

    gpt4 = MagicMock(spec=Model)
    gpt4.name = "gpt-4-turbo"
    gpt4.provider = "openai"
    gpt4.cost_per_1k_prompt_tokens = Decimal("0.01")
    gpt4.cost_per_1k_completion_tokens = Decimal("0.03")
    gpt4.context_window = 128000
    gpt4.enabled = True

    claude = MagicMock(spec=Model)
    claude.name = "claude-3-sonnet"
    claude.provider = "anthropic"
    claude.cost_per_1k_prompt_tokens = Decimal("0.003")
    claude.cost_per_1k_completion_tokens = Decimal("0.015")
    claude.context_window = 200000
    claude.enabled = True

    return [gpt35, gpt4, claude]


@pytest.fixture
def registry(mock_models):
    """Create a loaded registry with mock models."""
    reg = ModelRegistry()
    reg._models = {model.name: model for model in mock_models}
    reg._loaded = True
    return reg


# Validate Model Tests


def test_validate_model_success(registry):
    """Test validate_model passes for known model."""
    # Should not raise
    registry.validate_model("gpt-3.5-turbo")
    registry.validate_model("gpt-4-turbo")
    registry.validate_model("claude-3-sonnet")


def test_validate_model_unknown_raises_error(registry):
    """Test validate_model raises ValidationError for unknown model."""
    with pytest.raises(ValidationError) as exc_info:
        registry.validate_model("unknown-model")

    assert "Unknown model 'unknown-model'" in exc_info.value.message
    assert "Available models:" in exc_info.value.message


def test_validate_model_not_loaded_raises_error():
    """Test validate_model raises error when registry not loaded."""
    reg = ModelRegistry()  # Not loaded

    with pytest.raises(ValidationError) as exc_info:
        reg.validate_model("gpt-3.5-turbo")

    assert "not loaded" in exc_info.value.message


# Get Model Metadata Tests


def test_get_model_metadata_success(registry):
    """Test get_model_metadata returns proper metadata."""
    metadata = registry.get_model_metadata("gpt-4-turbo")

    assert metadata.name == "gpt-4-turbo"
    assert metadata.provider == "openai"
    assert metadata.cost_per_1k_prompt_tokens == 0.01
    assert metadata.cost_per_1k_completion_tokens == 0.03
    assert metadata.context_window == 128000
    assert metadata.enabled is True


def test_get_model_metadata_unknown_raises_error(registry):
    """Test get_model_metadata raises error for unknown model."""
    with pytest.raises(ValidationError):
        registry.get_model_metadata("unknown-model")


def test_get_model_metadata_anthropic(registry):
    """Test get_model_metadata for Anthropic model."""
    metadata = registry.get_model_metadata("claude-3-sonnet")

    assert metadata.name == "claude-3-sonnet"
    assert metadata.provider == "anthropic"
    assert metadata.context_window == 200000


# Estimate Cost Tests


def test_estimate_cost_prompt_only(registry):
    """Test cost estimation with only prompt tokens."""
    estimate = registry.estimate_cost("gpt-3.5-turbo", prompt_tokens=1000)

    assert estimate.model_name == "gpt-3.5-turbo"
    assert estimate.prompt_tokens == 1000
    assert estimate.completion_tokens == 0
    assert estimate.prompt_cost == 0.0015  # 1000/1000 * 0.0015
    assert estimate.completion_cost == 0.0
    assert estimate.total_cost == 0.0015


def test_estimate_cost_with_completion(registry):
    """Test cost estimation with prompt and completion tokens."""
    estimate = registry.estimate_cost("gpt-4-turbo", prompt_tokens=2000, completion_tokens=500)

    assert estimate.prompt_tokens == 2000
    assert estimate.completion_tokens == 500
    # prompt: 2000/1000 * 0.01 = 0.02
    # completion: 500/1000 * 0.03 = 0.015
    # total: 0.035
    assert estimate.prompt_cost == 0.02
    assert estimate.completion_cost == 0.015
    assert estimate.total_cost == 0.035


def test_estimate_cost_large_request(registry):
    """Test cost estimation for large request."""
    estimate = registry.estimate_cost("claude-3-sonnet", prompt_tokens=50000, completion_tokens=10000)

    # prompt: 50000/1000 * 0.003 = 0.15
    # completion: 10000/1000 * 0.015 = 0.15
    # total: 0.30
    assert estimate.prompt_cost == 0.15
    assert estimate.completion_cost == 0.15
    assert estimate.total_cost == 0.3


def test_estimate_cost_unknown_model_raises_error(registry):
    """Test estimate_cost raises error for unknown model."""
    with pytest.raises(ValidationError):
        registry.estimate_cost("unknown-model", prompt_tokens=1000)


def test_estimate_cost_zero_tokens(registry):
    """Test cost estimation with zero tokens."""
    estimate = registry.estimate_cost("gpt-3.5-turbo", prompt_tokens=0, completion_tokens=0)

    assert estimate.prompt_cost == 0.0
    assert estimate.completion_cost == 0.0
    assert estimate.total_cost == 0.0


# Estimate Tokens Tests


def test_estimate_tokens_short_text(registry):
    """Test token estimation for short text."""
    text = "Hello, world!"
    tokens = registry.estimate_tokens(text)

    # 13 chars / 4 = 3 tokens
    assert tokens == 3


def test_estimate_tokens_long_text(registry):
    """Test token estimation for longer text."""
    text = "This is a longer text that should have more tokens." * 10
    tokens = registry.estimate_tokens(text)

    # 51 chars * 10 = 510 chars / 4 = 127 tokens
    assert tokens == len(text) // 4


def test_estimate_tokens_empty_text(registry):
    """Test token estimation for empty text returns at least 1."""
    tokens = registry.estimate_tokens("")

    assert tokens == 1  # Minimum 1 token


def test_estimate_tokens_very_short(registry):
    """Test token estimation for very short text."""
    tokens = registry.estimate_tokens("Hi")

    assert tokens == 1  # Minimum 1 token


# Model Exists and Get Model Tests


def test_model_exists_true(registry):
    """Test model_exists returns True for known models."""
    assert registry.model_exists("gpt-3.5-turbo") is True
    assert registry.model_exists("gpt-4-turbo") is True
    assert registry.model_exists("claude-3-sonnet") is True


def test_model_exists_false(registry):
    """Test model_exists returns False for unknown models."""
    assert registry.model_exists("unknown-model") is False
    assert registry.model_exists("") is False


def test_get_model_success(registry):
    """Test get_model returns model for known name."""
    model = registry.get_model("gpt-3.5-turbo")

    assert model is not None
    assert model.name == "gpt-3.5-turbo"


def test_get_model_unknown_returns_none(registry):
    """Test get_model returns None for unknown name."""
    model = registry.get_model("unknown-model")

    assert model is None


def test_get_model_not_loaded_returns_none():
    """Test get_model returns None when not loaded."""
    reg = ModelRegistry()  # Not loaded
    model = reg.get_model("gpt-3.5-turbo")

    assert model is None


# Get All Models Tests


def test_get_all_models(registry):
    """Test get_all_models returns all models."""
    models = registry.get_all_models()

    assert len(models) == 3
    names = [m.name for m in models]
    assert "gpt-3.5-turbo" in names
    assert "gpt-4-turbo" in names
    assert "claude-3-sonnet" in names


def test_get_models_by_provider(registry):
    """Test get_models_by_provider filters correctly."""
    openai_models = registry.get_models_by_provider("openai")
    anthropic_models = registry.get_models_by_provider("anthropic")

    assert len(openai_models) == 2
    assert len(anthropic_models) == 1
    assert anthropic_models[0].name == "claude-3-sonnet"


def test_get_models_by_provider_unknown(registry):
    """Test get_models_by_provider returns empty for unknown provider."""
    models = registry.get_models_by_provider("unknown")

    assert len(models) == 0


# Is Loaded Tests


def test_is_loaded_true(registry):
    """Test is_loaded returns True for loaded registry."""
    assert registry.is_loaded() is True


def test_is_loaded_false():
    """Test is_loaded returns False for new registry."""
    reg = ModelRegistry()
    assert reg.is_loaded() is False
