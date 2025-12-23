"""OpenAI API client adapter for chat completions.

Implements async client for OpenAI API with streaming support,
token usage tracking, and proper error handling.

Security:
    - API key from environment variable OPENAI_API_KEY
    - API key is never logged
"""
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from src.config.settings import settings
from src.shared.exceptions import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)

# Timeout configuration
DEFAULT_TIMEOUT = 30.0  # 30 seconds


@dataclass
class TokenUsage:
    """Token usage from LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIClient:
    """Async client adapter for OpenAI API.

    Features:
    - Async chat completions (streaming and non-streaming)
    - Token usage tracking
    - 30s timeout with connection pooling
    - Proper error handling (401 -> 500, 429 -> 429, timeout -> 504)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
            base_url: Custom base URL (defaults to settings.openai_base_url)
                      Useful for LM Studio, Ollama, or other OpenAI-compatible APIs
            timeout: Request timeout in seconds (default 30s)
        """
        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._timeout = timeout
        self._last_usage: TokenUsage | None = None

        if not self._api_key:
            logger.warning("OpenAI API key not configured")

        if self._base_url:
            logger.info(f"Using custom OpenAI base URL: {self._base_url}")

        # Initialize async client with connection pooling
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise LLMError("OpenAI API key not configured")

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,  # Support custom endpoints (LM Studio, Ollama)
                timeout=self._timeout,
                max_retries=0,  # We handle retries ourselves
            )
        return self._client

    def get_usage(self) -> TokenUsage | None:
        """Get token usage from last request.

        Returns:
            TokenUsage from last request, or None if no request made
        """
        return self._last_usage

    async def send_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncGenerator[str, None]:
        """Send chat completion request to OpenAI.

        Args:
            model: Model name (e.g., 'gpt-3.5-turbo', 'gpt-4-turbo')
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response content (str) or async generator of chunks

        Raises:
            LLMError: For API errors (401, 5xx)
            LLMTimeoutError: For timeout errors (504)
        """
        # Prepend system prompt if provided and not already in messages
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

        if stream:
            return self._stream_message(model, messages, **kwargs)
        else:
            return await self._send_message_sync(model, messages, **kwargs)

    async def _send_message_sync(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Send non-streaming chat completion request."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                stream=False,
                **kwargs,
            )

            # Track usage
            if response.usage:
                self._last_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            content = response.choices[0].message.content or ""
            logger.debug(f"OpenAI response: model={model}, tokens={self._last_usage}")

            return content

        except APITimeoutError as e:
            logger.error(f"OpenAI timeout: {e}")
            raise LLMTimeoutError(f"OpenAI request timed out after {self._timeout}s")

        except APIStatusError as e:
            self._handle_api_error(e)

        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise LLMError(f"Failed to connect to OpenAI API: {e}")

        except Exception as e:
            logger.error(f"OpenAI unexpected error: {e}")
            raise LLMError(f"OpenAI error: {e}")

    async def _stream_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response."""
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )

            async for chunk in stream:
                # Handle usage in final chunk
                if chunk.usage:
                    self._last_usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

                # Yield content chunks
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except APITimeoutError as e:
            logger.error(f"OpenAI streaming timeout: {e}")
            raise LLMTimeoutError(f"OpenAI streaming timed out after {self._timeout}s")

        except APIStatusError as e:
            self._handle_api_error(e)

        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise LLMError(f"Failed to connect to OpenAI API: {e}")

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMError(f"OpenAI streaming error: {e}")

    def _handle_api_error(self, error: APIStatusError) -> None:
        """Handle OpenAI API errors with proper status codes.

        Error mapping:
        - 401 (invalid key) -> LLMError (500)
        - 429 (rate limit) -> LLMError with rate_limit info
        - 5xx (server error) -> LLMError (500)
        """
        status_code = error.status_code
        error_message = str(error)

        # Don't log the API key in error messages
        if "api_key" in error_message.lower():
            error_message = "Invalid API key"

        logger.error(f"OpenAI API error: status={status_code}")

        if status_code == 401:
            raise LLMError("OpenAI authentication failed - check API key")
        elif status_code == 429:
            # Rate limit - could implement retry-after handling
            raise LLMError("OpenAI rate limit exceeded. Please retry later.")
        elif status_code >= 500:
            raise LLMError(f"OpenAI server error: {status_code}")
        else:
            raise LLMError(f"OpenAI API error: {error_message}")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Adapter to implement LLMProvider protocol from message_service
class OpenAIProvider:
    """OpenAI provider implementing LLMProvider protocol.

    This adapter bridges OpenAIClient to the MessageService's LLMProvider protocol.
    """

    def __init__(self, client: OpenAIClient | None = None):
        """Initialize with optional client (creates new one if not provided)."""
        self._client = client or OpenAIClient()

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config (temperature, max_tokens, etc.)

        Returns:
            Tuple of (response_content, prompt_tokens, completion_tokens)
        """
        kwargs = config or {}

        content = await self._client.send_message(
            model=model,
            messages=messages,
            stream=False,
            **kwargs,
        )

        usage = self._client.get_usage()
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return content, prompt_tokens, completion_tokens  # type: ignore

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str, bool, int | None, int | None], None]:
        """Generate a streaming response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

        Yields:
            Tuples of (content_chunk, is_done, prompt_tokens, completion_tokens)
        """
        kwargs = config or {}

        stream = await self._client.send_message(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:  # type: ignore
            yield (chunk, False, None, None)

        # Final chunk with usage
        usage = self._client.get_usage()
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        yield ("", True, prompt_tokens, completion_tokens)
