"""Anthropic API client adapter for Messages API.

Implements async client for Anthropic API with streaming support,
token usage tracking, and proper error handling.

Security:
    - API key from environment variable ANTHROPIC_API_KEY
    - API key is never logged
"""
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
)

from src.config.settings import settings
from src.shared.exceptions import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)

# Timeout configuration
DEFAULT_TIMEOUT = 30.0  # 30 seconds

# Default max_tokens for Anthropic (required parameter)
DEFAULT_MAX_TOKENS = 4096


@dataclass
class TokenUsage:
    """Token usage from LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AnthropicClient:
    """Async client adapter for Anthropic API.

    Features:
    - Async chat completions (streaming and non-streaming)
    - Token usage tracking
    - 30s timeout with connection pooling
    - Proper error handling (401 -> 500, 429 -> 429, timeout -> 504)
    """

    def __init__(self, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to settings.anthropic_api_key)
            timeout: Request timeout in seconds (default 30s)
        """
        self._api_key = api_key or settings.anthropic_api_key
        self._timeout = timeout
        self._last_usage: TokenUsage | None = None

        if not self._api_key:
            logger.warning("Anthropic API key not configured")

        # Initialize async client with connection pooling
        self._client: AsyncAnthropic | None = None

    def _get_client(self) -> AsyncAnthropic:
        """Get or create async Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise LLMError("Anthropic API key not configured")

            self._client = AsyncAnthropic(
                api_key=self._api_key,
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
        """Send message request to Anthropic.

        Args:
            model: Model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229')
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response content (str) or async generator of chunks

        Raises:
            LLMError: For API errors (401, 5xx)
            LLMTimeoutError: For timeout errors (504)
        """
        # Extract system prompt from messages if present
        filtered_messages = []
        extracted_system = system_prompt

        for msg in messages:
            if msg.get("role") == "system":
                # Anthropic requires system prompt as separate parameter
                extracted_system = msg.get("content", extracted_system)
            else:
                filtered_messages.append(msg)

        # Use provided system_prompt or extracted one
        final_system = system_prompt if system_prompt else extracted_system

        # Ensure max_tokens is set (required by Anthropic)
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        if stream:
            return self._stream_message(model, filtered_messages, final_system, **kwargs)
        else:
            return await self._send_message_sync(model, filtered_messages, final_system, **kwargs)

    async def _send_message_sync(
        self,
        model: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send non-streaming message request."""
        client = self._get_client()

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                **kwargs,
            }

            # Add system prompt if provided
            if system_prompt:
                request_kwargs["system"] = system_prompt

            response = await client.messages.create(**request_kwargs)

            # Track usage
            self._last_usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            # Extract text content
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            logger.debug(f"Anthropic response: model={model}, tokens={self._last_usage}")

            return content

        except APITimeoutError as e:
            logger.error(f"Anthropic timeout: {e}")
            raise LLMTimeoutError(f"Anthropic request timed out after {self._timeout}s")

        except APIStatusError as e:
            self._handle_api_error(e)
            raise  # unreachable, but satisfies type checker

        except APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")
            raise LLMError(f"Failed to connect to Anthropic API: {e}")

        except Exception as e:
            logger.error(f"Anthropic unexpected error: {e}")
            raise LLMError(f"Anthropic error: {e}")

    async def _stream_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream message response."""
        client = self._get_client()

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                **kwargs,
            }

            # Add system prompt if provided
            if system_prompt:
                request_kwargs["system"] = system_prompt

            # Track tokens during streaming
            input_tokens = 0
            output_tokens = 0

            async with client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    # Handle message start for input tokens
                    if hasattr(event, "type"):
                        if event.type == "message_start" and hasattr(event, "message"):
                            if hasattr(event.message, "usage"):
                                input_tokens = event.message.usage.input_tokens

                        # Handle content block delta for text chunks
                        elif event.type == "content_block_delta":
                            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                                yield event.delta.text

                        # Handle message delta for output tokens
                        elif event.type == "message_delta":
                            if hasattr(event, "usage") and hasattr(event.usage, "output_tokens"):
                                output_tokens = event.usage.output_tokens

                # Update usage after stream completes
                self._last_usage = TokenUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )

        except APITimeoutError as e:
            logger.error(f"Anthropic streaming timeout: {e}")
            raise LLMTimeoutError(f"Anthropic streaming timed out after {self._timeout}s")

        except APIStatusError as e:
            self._handle_api_error(e)

        except APIConnectionError as e:
            logger.error(f"Anthropic connection error: {e}")
            raise LLMError(f"Failed to connect to Anthropic API: {e}")

        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise LLMError(f"Anthropic streaming error: {e}")

    def _handle_api_error(self, error: APIStatusError) -> None:
        """Handle Anthropic API errors with proper status codes.

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

        logger.error(f"Anthropic API error: status={status_code}")

        if status_code == 401:
            raise LLMError("Anthropic authentication failed - check API key")
        elif status_code == 429:
            # Rate limit - could implement retry-after handling
            raise LLMError("Anthropic rate limit exceeded. Please retry later.")
        elif status_code >= 500:
            raise LLMError(f"Anthropic server error: {status_code}")
        else:
            raise LLMError(f"Anthropic API error: {error_message}")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Adapter to implement LLMProvider protocol from message_service
class AnthropicProvider:
    """Anthropic provider implementing LLMProvider protocol.

    This adapter bridges AnthropicClient to the MessageService's LLMProvider protocol.
    """

    def __init__(self, client: AnthropicClient | None = None):
        """Initialize with optional client (creates new one if not provided)."""
        self._client = client or AnthropicClient()

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response from Anthropic.

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
        """Generate a streaming response from Anthropic.

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
