"""LLM Provider Factory - Dynamic Selection of LLM Adapters.

Factory pattern implementation for selecting the appropriate LLM provider
adapter (OpenAI or Anthropic) based on provider name or model metadata.
"""
import logging
from typing import Protocol, Any
from collections.abc import AsyncGenerator

from src.integrations.anthropic_client import AnthropicProvider
from src.integrations.openai_client import OpenAIProvider
from src.shared.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMProviderContract(Protocol):
    """Common interface for all LLM provider adapters.

    Both OpenAI and Anthropic adapters must implement this protocol
    to ensure they are interchangeable.
    """

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

        Returns:
            Tuple of (response_content, prompt_tokens, completion_tokens)

        Raises:
            LLMTimeoutError: If request times out
            LLMError: If LLM returns an error
        """
        ...

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        """Generate a streaming response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

        Yields:
            Tuples of (content_chunk, is_done, prompt_tokens, completion_tokens)
            Token counts are only provided in the final chunk (is_done=True)

        Raises:
            LLMTimeoutError: If request times out
            LLMError: If LLM returns an error
        """
        ...


# Supported providers
SUPPORTED_PROVIDERS = {"openai", "anthropic"}


class LLMProviderFactory:
    """Factory for creating LLM provider adapters.

    Selects the appropriate adapter based on provider name and ensures
    all adapters implement the common LLMProviderContract interface.
    """

    @staticmethod
    def get_provider(provider_name: str) -> LLMProviderContract:
        """Get LLM provider adapter by provider name.

        Args:
            provider_name: Provider name ('openai' or 'anthropic')

        Returns:
            LLM provider adapter implementing LLMProviderContract

        Raises:
            LLMError: If provider is unknown or initialization fails (500)
        """
        provider_name = provider_name.lower().strip()

        if provider_name not in SUPPORTED_PROVIDERS:
            available = ", ".join(sorted(SUPPORTED_PROVIDERS))
            logger.error(f"Unknown LLM provider: {provider_name}")
            raise LLMError(
                f"Unknown LLM provider '{provider_name}'. "
                f"Supported providers: {available}"
            )

        try:
            if provider_name == "openai":
                return OpenAIProvider()
            elif provider_name == "anthropic":
                return AnthropicProvider()
            else:
                # This shouldn't happen due to the check above
                raise LLMError(f"Unsupported provider: {provider_name}")

        except LLMError:
            # Re-raise LLM errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {provider_name} provider: {e}")
            raise LLMError(f"Failed to initialize LLM provider '{provider_name}': {e}")

    @staticmethod
    def get_provider_for_model(
        model_name: str,
        model_registry: Any,
    ) -> LLMProviderContract:
        """Get LLM provider adapter based on model metadata.

        Args:
            model_name: Model name to look up
            model_registry: ModelRegistry instance to get model metadata

        Returns:
            LLM provider adapter implementing LLMProviderContract

        Raises:
            LLMError: If provider is unknown or initialization fails (500)
            ValidationError: If model is not found (from model_registry)
        """
        metadata = model_registry.get_model_metadata(model_name)
        return LLMProviderFactory.get_provider(metadata.provider)

    @staticmethod
    def is_supported(provider_name: str) -> bool:
        """Check if a provider is supported.

        Args:
            provider_name: Provider name to check

        Returns:
            True if provider is supported, False otherwise
        """
        return provider_name.lower().strip() in SUPPORTED_PROVIDERS

    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported provider names.

        Returns:
            List of supported provider names
        """
        return sorted(SUPPORTED_PROVIDERS)


# Convenience function for simple provider lookup
def get_llm_provider(provider_name: str) -> LLMProviderContract:
    """Get LLM provider adapter by name.

    This is a convenience function that wraps LLMProviderFactory.get_provider().

    Args:
        provider_name: Provider name ('openai' or 'anthropic')

    Returns:
        LLM provider adapter

    Raises:
        LLMError: If provider is unknown or initialization fails (500)
    """
    return LLMProviderFactory.get_provider(provider_name)
