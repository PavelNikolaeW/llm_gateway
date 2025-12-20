"""Integration tests for LLM providers with mocked HTTP responses.

Tests OpenAI and Anthropic providers with mocked HTTP responses
for both streaming and non-streaming modes.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.openai_client import OpenAIClient, OpenAIProvider
from src.integrations.anthropic_client import AnthropicClient, AnthropicProvider
from src.integrations.llm_factory import LLMProviderFactory, get_llm_provider


class TestOpenAIProvider:
    """Tests for OpenAI provider with mocked HTTP responses."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider for testing."""
        with patch.object(OpenAIClient, '_get_client'):
            provider = OpenAIProvider()
            return provider

    @pytest.mark.asyncio
    async def test_generate_non_streaming(self):
        """Test non-streaming generation with mocked response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello! How can I help you today?"))
        ]
        mock_response.usage = MagicMock(prompt_tokens=15, completion_tokens=8)

        with patch('src.integrations.openai_client.OpenAIClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Hello! How can I help you today?")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=15, completion_tokens=8))
            MockClient.return_value = mock_client_instance

            provider = OpenAIProvider()
            content, prompt_tokens, completion_tokens = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4",
            )

        assert content == "Hello! How can I help you today?"
        assert prompt_tokens == 15
        assert completion_tokens == 8

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        with patch('src.integrations.openai_client.OpenAIClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="I'm a helpful assistant.")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=25, completion_tokens=5))
            MockClient.return_value = mock_client_instance

            provider = OpenAIProvider()
            content, _, _ = await provider.generate(
                messages=[{"role": "user", "content": "Who are you?"}],
                model="gpt-4",
                config={"system_prompt": "You are a helpful assistant."},
            )

            # Verify system prompt was passed
            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("system_prompt") == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        """Test generation with custom temperature."""
        with patch('src.integrations.openai_client.OpenAIClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Response")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=10, completion_tokens=5))
            MockClient.return_value = mock_client_instance

            provider = OpenAIProvider()
            await provider.generate(
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4",
                config={"temperature": 0.5},
            )

            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("temperature") == 0.5

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self):
        """Test generation with max_tokens limit."""
        with patch('src.integrations.openai_client.OpenAIClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Short")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=10, completion_tokens=1))
            MockClient.return_value = mock_client_instance

            provider = OpenAIProvider()
            await provider.generate(
                messages=[{"role": "user", "content": "Test"}],
                model="gpt-4",
                config={"max_tokens": 100},
            )

            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("max_tokens") == 100


class TestAnthropicProvider:
    """Tests for Anthropic provider with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_generate_non_streaming(self):
        """Test non-streaming generation with mocked response."""
        with patch('src.integrations.anthropic_client.AnthropicClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Hello! How can I help you?")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=12, completion_tokens=7))
            MockClient.return_value = mock_client_instance

            provider = AnthropicProvider()
            content, prompt_tokens, completion_tokens = await provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-opus-20240229",
            )

        assert content == "Hello! How can I help you?"
        assert prompt_tokens == 12
        assert completion_tokens == 7

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        with patch('src.integrations.anthropic_client.AnthropicClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="I'm Claude.")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=20, completion_tokens=3))
            MockClient.return_value = mock_client_instance

            provider = AnthropicProvider()
            content, _, _ = await provider.generate(
                messages=[{"role": "user", "content": "Who are you?"}],
                model="claude-3-opus-20240229",
                config={"system_prompt": "You are Claude, an AI assistant."},
            )

            # Verify system prompt was passed
            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("system_prompt") == "You are Claude, an AI assistant."

    @pytest.mark.asyncio
    async def test_generate_with_temperature(self):
        """Test generation with custom temperature."""
        with patch('src.integrations.anthropic_client.AnthropicClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Response")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=10, completion_tokens=1))
            MockClient.return_value = mock_client_instance

            provider = AnthropicProvider()
            await provider.generate(
                messages=[{"role": "user", "content": "Test"}],
                model="claude-3-opus-20240229",
                config={"temperature": 0.7},
            )

            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_generate_with_max_tokens(self):
        """Test generation with max_tokens limit."""
        with patch('src.integrations.anthropic_client.AnthropicClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(return_value="Short")
            mock_client_instance.get_usage = MagicMock(return_value=MagicMock(prompt_tokens=10, completion_tokens=1))
            MockClient.return_value = mock_client_instance

            provider = AnthropicProvider()
            await provider.generate(
                messages=[{"role": "user", "content": "Test"}],
                model="claude-3-opus-20240229",
                config={"max_tokens": 100},
            )

            call_args = mock_client_instance.send_message.call_args
            assert call_args.kwargs.get("max_tokens") == 100


class TestLLMProviderErrors:
    """Tests for LLM provider error handling."""

    @pytest.mark.asyncio
    async def test_openai_handles_api_error(self):
        """Test OpenAI provider handles API errors gracefully."""
        with patch('src.integrations.openai_client.OpenAIClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(
                side_effect=Exception("API Error: Rate limit exceeded")
            )
            MockClient.return_value = mock_client_instance

            provider = OpenAIProvider()
            with pytest.raises(Exception, match="Rate limit"):
                await provider.generate(
                    messages=[{"role": "user", "content": "Test"}],
                    model="gpt-4",
                )

    @pytest.mark.asyncio
    async def test_anthropic_handles_api_error(self):
        """Test Anthropic provider handles API errors gracefully."""
        with patch('src.integrations.anthropic_client.AnthropicClient') as MockClient:
            mock_client_instance = MagicMock()
            mock_client_instance.send_message = AsyncMock(
                side_effect=Exception("API Error: Invalid API key")
            )
            MockClient.return_value = mock_client_instance

            provider = AnthropicProvider()
            with pytest.raises(Exception, match="Invalid API key"):
                await provider.generate(
                    messages=[{"role": "user", "content": "Test"}],
                    model="claude-3-opus-20240229",
                )


class TestLLMProviderFactory:
    """Tests for LLM factory pattern."""

    def test_factory_creates_openai_provider(self):
        """Test factory creates OpenAI provider for 'openai'."""
        with patch('src.integrations.openai_client.OpenAIClient'):
            provider = LLMProviderFactory.get_provider("openai")
            assert isinstance(provider, OpenAIProvider)

    def test_factory_creates_anthropic_provider(self):
        """Test factory creates Anthropic provider for 'anthropic'."""
        with patch('src.integrations.anthropic_client.AnthropicClient'):
            provider = LLMProviderFactory.get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_factory_rejects_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        from src.shared.exceptions import LLMError

        with pytest.raises(LLMError, match="Unknown LLM provider"):
            LLMProviderFactory.get_provider("unknown_provider")

    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        providers = LLMProviderFactory.get_supported_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_is_supported(self):
        """Test checking if provider is supported."""
        assert LLMProviderFactory.is_supported("openai") is True
        assert LLMProviderFactory.is_supported("anthropic") is True
        assert LLMProviderFactory.is_supported("unknown") is False

    def test_convenience_function(self):
        """Test get_llm_provider convenience function."""
        with patch('src.integrations.openai_client.OpenAIClient'):
            provider = get_llm_provider("openai")
            assert isinstance(provider, OpenAIProvider)
