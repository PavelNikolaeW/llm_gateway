"""Unit tests for LLM Provider Factory."""
import pytest
from unittest.mock import MagicMock, patch

from src.integrations.llm_factory import (
    LLMProviderContract,
    LLMProviderFactory,
    SUPPORTED_PROVIDERS,
    get_llm_provider,
)
from src.integrations.openai_client import OpenAIProvider
from src.integrations.anthropic_client import AnthropicProvider
from src.shared.exceptions import LLMError
from src.shared.schemas import ModelMetadata


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory class."""

    def test_get_provider_openai(self):
        """Test factory returns OpenAIProvider for 'openai'."""
        provider = LLMProviderFactory.get_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_anthropic(self):
        """Test factory returns AnthropicProvider for 'anthropic'."""
        provider = LLMProviderFactory.get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_get_provider_case_insensitive(self):
        """Test provider name is case insensitive."""
        provider_upper = LLMProviderFactory.get_provider("OPENAI")
        provider_mixed = LLMProviderFactory.get_provider("OpenAI")
        provider_lower = LLMProviderFactory.get_provider("openai")

        assert isinstance(provider_upper, OpenAIProvider)
        assert isinstance(provider_mixed, OpenAIProvider)
        assert isinstance(provider_lower, OpenAIProvider)

    def test_get_provider_strips_whitespace(self):
        """Test provider name strips whitespace."""
        provider = LLMProviderFactory.get_provider("  openai  ")
        assert isinstance(provider, OpenAIProvider)

    def test_get_provider_unknown_raises_llm_error(self):
        """Test unknown provider raises LLMError with 500 status."""
        with pytest.raises(LLMError) as exc_info:
            LLMProviderFactory.get_provider("unknown_provider")

        assert exc_info.value.status_code == 500
        assert "Unknown LLM provider" in exc_info.value.message
        assert "unknown_provider" in exc_info.value.message

    def test_get_provider_unknown_lists_available(self):
        """Test unknown provider error lists available providers."""
        with pytest.raises(LLMError) as exc_info:
            LLMProviderFactory.get_provider("invalid")

        assert "anthropic" in exc_info.value.message
        assert "openai" in exc_info.value.message

    def test_get_provider_empty_string_raises(self):
        """Test empty string raises LLMError."""
        with pytest.raises(LLMError) as exc_info:
            LLMProviderFactory.get_provider("")

        assert exc_info.value.status_code == 500

    def test_get_provider_handles_initialization_error(self):
        """Test factory handles provider initialization errors."""
        with patch(
            "src.integrations.llm_factory.OpenAIProvider",
            side_effect=Exception("Init failed"),
        ):
            with pytest.raises(LLMError) as exc_info:
                LLMProviderFactory.get_provider("openai")

            assert exc_info.value.status_code == 500
            assert "Failed to initialize" in exc_info.value.message

    def test_get_provider_for_model_openai(self):
        """Test get_provider_for_model returns OpenAI for OpenAI model."""
        mock_registry = MagicMock()
        mock_registry.get_model_metadata.return_value = ModelMetadata(
            name="gpt-4",
            provider="openai",
            cost_per_1k_prompt_tokens=0.03,
            cost_per_1k_completion_tokens=0.06,
            context_window=8192,
            enabled=True,
        )

        provider = LLMProviderFactory.get_provider_for_model("gpt-4", mock_registry)
        assert isinstance(provider, OpenAIProvider)
        mock_registry.get_model_metadata.assert_called_once_with("gpt-4")

    def test_get_provider_for_model_anthropic(self):
        """Test get_provider_for_model returns Anthropic for Anthropic model."""
        mock_registry = MagicMock()
        mock_registry.get_model_metadata.return_value = ModelMetadata(
            name="claude-3-sonnet-20240229",
            provider="anthropic",
            cost_per_1k_prompt_tokens=0.003,
            cost_per_1k_completion_tokens=0.015,
            context_window=200000,
            enabled=True,
        )

        provider = LLMProviderFactory.get_provider_for_model(
            "claude-3-sonnet-20240229", mock_registry
        )
        assert isinstance(provider, AnthropicProvider)

    def test_is_supported_returns_true_for_valid(self):
        """Test is_supported returns True for valid providers."""
        assert LLMProviderFactory.is_supported("openai") is True
        assert LLMProviderFactory.is_supported("anthropic") is True
        assert LLMProviderFactory.is_supported("OPENAI") is True
        assert LLMProviderFactory.is_supported("  Anthropic  ") is True

    def test_is_supported_returns_false_for_invalid(self):
        """Test is_supported returns False for invalid providers."""
        assert LLMProviderFactory.is_supported("unknown") is False
        assert LLMProviderFactory.is_supported("") is False
        assert LLMProviderFactory.is_supported("gemini") is False

    def test_get_supported_providers(self):
        """Test get_supported_providers returns sorted list."""
        providers = LLMProviderFactory.get_supported_providers()
        assert providers == ["anthropic", "openai"]
        assert len(providers) == 2


class TestGetLLMProvider:
    """Tests for get_llm_provider convenience function."""

    def test_get_llm_provider_openai(self):
        """Test convenience function returns OpenAIProvider."""
        provider = get_llm_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_llm_provider_anthropic(self):
        """Test convenience function returns AnthropicProvider."""
        provider = get_llm_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_get_llm_provider_unknown_raises(self):
        """Test convenience function raises for unknown provider."""
        with pytest.raises(LLMError) as exc_info:
            get_llm_provider("invalid")

        assert exc_info.value.status_code == 500


class TestSupportedProviders:
    """Tests for SUPPORTED_PROVIDERS constant."""

    def test_supported_providers_contains_openai(self):
        """Test SUPPORTED_PROVIDERS contains 'openai'."""
        assert "openai" in SUPPORTED_PROVIDERS

    def test_supported_providers_contains_anthropic(self):
        """Test SUPPORTED_PROVIDERS contains 'anthropic'."""
        assert "anthropic" in SUPPORTED_PROVIDERS

    def test_supported_providers_count(self):
        """Test SUPPORTED_PROVIDERS has expected count."""
        assert len(SUPPORTED_PROVIDERS) == 2


class TestLLMProviderContract:
    """Tests for LLMProviderContract protocol."""

    def test_openai_provider_implements_contract(self):
        """Test OpenAIProvider implements LLMProviderContract."""
        provider = OpenAIProvider()

        # Check methods exist
        assert hasattr(provider, "generate")
        assert hasattr(provider, "generate_stream")
        assert callable(provider.generate)
        assert callable(provider.generate_stream)

    def test_anthropic_provider_implements_contract(self):
        """Test AnthropicProvider implements LLMProviderContract."""
        provider = AnthropicProvider()

        # Check methods exist
        assert hasattr(provider, "generate")
        assert hasattr(provider, "generate_stream")
        assert callable(provider.generate)
        assert callable(provider.generate_stream)

    def test_providers_are_type_compatible(self):
        """Test both providers can be assigned to LLMProviderContract type."""
        # This is a static type check - runtime assertion
        openai: LLMProviderContract = OpenAIProvider()
        anthropic: LLMProviderContract = AnthropicProvider()

        # Both should have the same interface
        assert hasattr(openai, "generate")
        assert hasattr(openai, "generate_stream")
        assert hasattr(anthropic, "generate")
        assert hasattr(anthropic, "generate_stream")
