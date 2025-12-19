"""Unit tests for OpenAI client adapter with mocked API calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIConnectionError, APIStatusError, APITimeoutError

from src.integrations.openai_client import OpenAIClient, OpenAIProvider, TokenUsage
from src.shared.exceptions import LLMError, LLMTimeoutError


@pytest.fixture
def mock_openai_response():
    """Create mock OpenAI response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Hello! How can I help you?"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def mock_stream_chunks():
    """Create mock streaming chunks."""
    chunks = []

    # Content chunks
    for text in ["Hello", "!", " How", " can", " I", " help", "?"]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunk.usage = None
        chunks.append(chunk)

    # Final chunk with usage
    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage = MagicMock()
    final_chunk.usage.prompt_tokens = 10
    final_chunk.usage.completion_tokens = 7
    final_chunk.usage.total_tokens = 17
    chunks.append(final_chunk)

    return chunks


class TestOpenAIClient:
    """Tests for OpenAIClient class."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenAIClient(api_key="test-key")
        assert client._api_key == "test-key"
        assert client._timeout == 30.0
        assert client._client is None

    def test_init_with_custom_timeout(self):
        """Test client initialization with custom timeout."""
        client = OpenAIClient(api_key="test-key", timeout=60.0)
        assert client._timeout == 60.0

    def test_init_no_api_key_logs_warning(self):
        """Test client logs warning when no API key."""
        with patch("src.integrations.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = None
            client = OpenAIClient()
            assert client._api_key is None

    def test_get_client_raises_without_api_key(self):
        """Test _get_client raises LLMError without API key."""
        with patch("src.integrations.openai_client.settings") as mock_settings:
            mock_settings.openai_api_key = None
            client = OpenAIClient()

            with pytest.raises(LLMError) as exc_info:
                client._get_client()

            assert "not configured" in exc_info.value.message

    def test_get_usage_returns_none_initially(self):
        """Test get_usage returns None before any request."""
        client = OpenAIClient(api_key="test-key")
        assert client.get_usage() is None

    @pytest.mark.asyncio
    async def test_send_message_non_streaming(self, mock_openai_response):
        """Test non-streaming message send."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_get_client.return_value = mock_async_client

            result = await client.send_message(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            assert result == "Hello! How can I help you?"
            assert client.get_usage() is not None
            assert client.get_usage().prompt_tokens == 10
            assert client.get_usage().completion_tokens == 20
            assert client.get_usage().total_tokens == 30

    @pytest.mark.asyncio
    async def test_send_message_with_system_prompt(self, mock_openai_response):
        """Test message with system prompt prepended."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            mock_get_client.return_value = mock_async_client

            await client.send_message(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are helpful.",
                stream=False,
            )

            # Verify system prompt was added
            call_args = mock_async_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_send_message_streaming(self, mock_stream_chunks):
        """Test streaming message send."""
        client = OpenAIClient(api_key="test-key")

        async def mock_stream():
            for chunk in mock_stream_chunks:
                yield chunk

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_stream()
            )
            mock_get_client.return_value = mock_async_client

            result = await client.send_message(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert "".join(chunks) == "Hello! How can I help?"
            assert client.get_usage() is not None
            assert client.get_usage().prompt_tokens == 10
            assert client.get_usage().completion_tokens == 7

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test timeout error handling."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                side_effect=APITimeoutError(request=MagicMock())
            )
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMTimeoutError) as exc_info:
                await client.send_message(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "timed out" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_auth_error_401(self):
        """Test 401 authentication error handling."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            error = APIStatusError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
            mock_async_client.chat.completions.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "authentication failed" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_rate_limit_error_429(self):
        """Test 429 rate limit error handling."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            error = APIStatusError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
            mock_async_client.chat.completions.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "rate limit" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_server_error_500(self):
        """Test 500 server error handling."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            error = APIStatusError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            )
            mock_async_client.chat.completions.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "server error" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error handling."""
        client = OpenAIClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "Failed to connect" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_streaming_timeout_error(self, mock_stream_chunks):
        """Test timeout error during streaming."""
        client = OpenAIClient(api_key="test-key")

        async def mock_stream_with_timeout():
            yield mock_stream_chunks[0]
            raise APITimeoutError(request=MagicMock())

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_stream_with_timeout()
            )
            mock_get_client.return_value = mock_async_client

            result = await client.send_message(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )

            with pytest.raises(LLMTimeoutError):
                async for _ in result:
                    pass

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing client connection."""
        client = OpenAIClient(api_key="test-key")

        # First create the client
        mock_async_client = AsyncMock()
        client._client = mock_async_client

        await client.close()

        mock_async_client.close.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        """Test closing client when not initialized."""
        client = OpenAIClient(api_key="test-key")

        # Should not raise
        await client.close()
        assert client._client is None


class TestOpenAIProvider:
    """Tests for OpenAIProvider adapter."""

    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self, mock_openai_response):
        """Test generate returns (content, prompt_tokens, completion_tokens)."""
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="Hello!")
        mock_client.get_usage.return_value = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        provider = OpenAIProvider(client=mock_client)

        content, prompt_tokens, completion_tokens = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-3.5-turbo",
        )

        assert content == "Hello!"
        assert prompt_tokens == 10
        assert completion_tokens == 20

    @pytest.mark.asyncio
    async def test_generate_with_config(self, mock_openai_response):
        """Test generate passes config to client."""
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="Response")
        mock_client.get_usage.return_value = TokenUsage(10, 20, 30)

        provider = OpenAIProvider(client=mock_client)

        await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4",
            config={"temperature": 0.5, "max_tokens": 100},
        )

        mock_client.send_message.assert_called_once()
        call_kwargs = mock_client.send_message.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_handles_no_usage(self):
        """Test generate handles missing usage gracefully."""
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="Response")
        mock_client.get_usage.return_value = None

        provider = OpenAIProvider(client=mock_client)

        content, prompt_tokens, completion_tokens = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-3.5-turbo",
        )

        assert content == "Response"
        assert prompt_tokens == 0
        assert completion_tokens == 0

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self):
        """Test generate_stream yields correct tuples."""
        mock_client = MagicMock()

        async def mock_stream():
            yield "Hello"
            yield " World"

        mock_client.send_message = AsyncMock(return_value=mock_stream())
        mock_client.get_usage.return_value = TokenUsage(15, 5, 20)

        provider = OpenAIProvider(client=mock_client)

        chunks = []
        async for chunk in provider.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-3.5-turbo",
        ):
            chunks.append(chunk)

        # Content chunks
        assert chunks[0] == ("Hello", False, None, None)
        assert chunks[1] == (" World", False, None, None)

        # Final chunk with usage
        assert chunks[2][1] is True  # done=True
        assert chunks[2][2] == 15  # prompt_tokens
        assert chunks[2][3] == 5  # completion_tokens

    @pytest.mark.asyncio
    async def test_generate_stream_handles_no_usage(self):
        """Test generate_stream handles missing usage."""
        mock_client = MagicMock()

        async def mock_stream():
            yield "Response"

        mock_client.send_message = AsyncMock(return_value=mock_stream())
        mock_client.get_usage.return_value = None

        provider = OpenAIProvider(client=mock_client)

        chunks = []
        async for chunk in provider.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-3.5-turbo",
        ):
            chunks.append(chunk)

        # Final chunk should have 0 for tokens
        final = chunks[-1]
        assert final[1] is True
        assert final[2] == 0
        assert final[3] == 0


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_creation(self):
        """Test TokenUsage dataclass creation."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_equality(self):
        """Test TokenUsage equality comparison."""
        usage1 = TokenUsage(10, 20, 30)
        usage2 = TokenUsage(10, 20, 30)
        usage3 = TokenUsage(10, 25, 35)

        assert usage1 == usage2
        assert usage1 != usage3
