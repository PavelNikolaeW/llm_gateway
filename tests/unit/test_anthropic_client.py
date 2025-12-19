"""Unit tests for Anthropic client adapter with mocked API calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic import APIConnectionError, APIStatusError, APITimeoutError

from src.integrations.anthropic_client import AnthropicClient, AnthropicProvider, TokenUsage
from src.shared.exceptions import LLMError, LLMTimeoutError


@pytest.fixture
def mock_anthropic_response():
    """Create mock Anthropic response."""
    response = MagicMock()
    response.content = [MagicMock()]
    response.content[0].text = "Hello! How can I help you?"
    response.usage = MagicMock()
    response.usage.input_tokens = 10
    response.usage.output_tokens = 20
    return response


@pytest.fixture
def mock_stream_events():
    """Create mock streaming events."""
    events = []

    # Message start event with input tokens
    start_event = MagicMock()
    start_event.type = "message_start"
    start_event.message = MagicMock()
    start_event.message.usage = MagicMock()
    start_event.message.usage.input_tokens = 10
    events.append(start_event)

    # Content block delta events
    for text in ["Hello", "!", " How", " can", " I", " help", "?"]:
        delta_event = MagicMock()
        delta_event.type = "content_block_delta"
        delta_event.delta = MagicMock()
        delta_event.delta.text = text
        events.append(delta_event)

    # Message delta event with output tokens
    final_event = MagicMock()
    final_event.type = "message_delta"
    final_event.usage = MagicMock()
    final_event.usage.output_tokens = 7
    events.append(final_event)

    return events


class TestAnthropicClient:
    """Tests for AnthropicClient class."""

    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = AnthropicClient(api_key="test-key")
        assert client._api_key == "test-key"
        assert client._timeout == 30.0
        assert client._client is None

    def test_init_with_custom_timeout(self):
        """Test client initialization with custom timeout."""
        client = AnthropicClient(api_key="test-key", timeout=60.0)
        assert client._timeout == 60.0

    def test_init_no_api_key_logs_warning(self):
        """Test client logs warning when no API key."""
        with patch("src.integrations.anthropic_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            client = AnthropicClient()
            assert client._api_key is None

    def test_get_client_raises_without_api_key(self):
        """Test _get_client raises LLMError without API key."""
        with patch("src.integrations.anthropic_client.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            client = AnthropicClient()

            with pytest.raises(LLMError) as exc_info:
                client._get_client()

            assert "not configured" in exc_info.value.message

    def test_get_usage_returns_none_initially(self):
        """Test get_usage returns None before any request."""
        client = AnthropicClient(api_key="test-key")
        assert client.get_usage() is None

    @pytest.mark.asyncio
    async def test_send_message_non_streaming(self, mock_anthropic_response):
        """Test non-streaming message send."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_get_client.return_value = mock_async_client

            result = await client.send_message(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            assert result == "Hello! How can I help you?"
            assert client.get_usage() is not None
            assert client.get_usage().prompt_tokens == 10
            assert client.get_usage().completion_tokens == 20
            assert client.get_usage().total_tokens == 30

    @pytest.mark.asyncio
    async def test_send_message_with_system_prompt(self, mock_anthropic_response):
        """Test message with system prompt."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_get_client.return_value = mock_async_client

            await client.send_message(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are helpful.",
                stream=False,
            )

            # Verify system prompt was passed correctly
            call_kwargs = mock_async_client.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_send_message_extracts_system_from_messages(self, mock_anthropic_response):
        """Test system prompt extraction from messages."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_get_client.return_value = mock_async_client

            await client.send_message(
                model="claude-3-sonnet-20240229",
                messages=[
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ],
                stream=False,
            )

            # Verify system was extracted and messages filtered
            call_kwargs = mock_async_client.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "Be helpful."
            assert len(call_kwargs["messages"]) == 1
            assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_send_message_sets_default_max_tokens(self, mock_anthropic_response):
        """Test default max_tokens is set."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                return_value=mock_anthropic_response
            )
            mock_get_client.return_value = mock_async_client

            await client.send_message(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

            call_kwargs = mock_async_client.messages.create.call_args.kwargs
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_send_message_streaming(self, mock_stream_events):
        """Test streaming message send."""
        client = AnthropicClient(api_key="test-key")

        class MockStreamContext:
            def __init__(self, events):
                self.events = events

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.events:
                    return self.events.pop(0)
                raise StopAsyncIteration

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = MagicMock()
            mock_async_client.messages.stream = MagicMock(
                return_value=MockStreamContext(mock_stream_events.copy())
            )
            mock_get_client.return_value = mock_async_client

            result = await client.send_message(
                model="claude-3-sonnet-20240229",
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
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                side_effect=APITimeoutError(request=MagicMock())
            )
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMTimeoutError) as exc_info:
                await client.send_message(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "timed out" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_auth_error_401(self):
        """Test 401 authentication error handling."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            error = APIStatusError(
                message="Invalid API key",
                response=mock_response,
                body=None,
            )
            mock_async_client.messages.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "authentication failed" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_rate_limit_error_429(self):
        """Test 429 rate limit error handling."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            error = APIStatusError(
                message="Rate limit exceeded",
                response=mock_response,
                body=None,
            )
            mock_async_client.messages.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "rate limit" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_server_error_500(self):
        """Test 500 server error handling."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            error = APIStatusError(
                message="Server error",
                response=mock_response,
                body=None,
            )
            mock_async_client.messages.create = AsyncMock(side_effect=error)
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "server error" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_connection_error(self):
        """Test connection error handling."""
        client = AnthropicClient(api_key="test-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_async_client = AsyncMock()
            mock_async_client.messages.create = AsyncMock(
                side_effect=APIConnectionError(request=MagicMock())
            )
            mock_get_client.return_value = mock_async_client

            with pytest.raises(LLMError) as exc_info:
                await client.send_message(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "Failed to connect" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing client connection."""
        client = AnthropicClient(api_key="test-key")

        # First create the client
        mock_async_client = AsyncMock()
        client._client = mock_async_client

        await client.close()

        mock_async_client.close.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_client_when_none(self):
        """Test closing client when not initialized."""
        client = AnthropicClient(api_key="test-key")

        # Should not raise
        await client.close()
        assert client._client is None


class TestAnthropicProvider:
    """Tests for AnthropicProvider adapter."""

    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self, mock_anthropic_response):
        """Test generate returns (content, prompt_tokens, completion_tokens)."""
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="Hello!")
        mock_client.get_usage.return_value = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        provider = AnthropicProvider(client=mock_client)

        content, prompt_tokens, completion_tokens = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-sonnet-20240229",
        )

        assert content == "Hello!"
        assert prompt_tokens == 10
        assert completion_tokens == 20

    @pytest.mark.asyncio
    async def test_generate_with_config(self, mock_anthropic_response):
        """Test generate passes config to client."""
        mock_client = MagicMock()
        mock_client.send_message = AsyncMock(return_value="Response")
        mock_client.get_usage.return_value = TokenUsage(10, 20, 30)

        provider = AnthropicProvider(client=mock_client)

        await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-opus-20240229",
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

        provider = AnthropicProvider(client=mock_client)

        content, prompt_tokens, completion_tokens = await provider.generate(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-sonnet-20240229",
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

        provider = AnthropicProvider(client=mock_client)

        chunks = []
        async for chunk in provider.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-sonnet-20240229",
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

        provider = AnthropicProvider(client=mock_client)

        chunks = []
        async for chunk in provider.generate_stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-sonnet-20240229",
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
