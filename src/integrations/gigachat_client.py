"""GigaChat API client adapter (Sber).

Implements async client for GigaChat API with streaming support,
token usage tracking, and proper error handling.

GigaChat uses OAuth2 for authentication with client credentials flow.

Security:
    - Authorization key from environment variable GIGACHAT_AUTH_KEY
    - Scope from GIGACHAT_SCOPE (GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP)
    - Credentials are never logged
"""

import logging
import ssl
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from src.config.settings import settings
from src.shared.exceptions import LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)

# Timeout configuration
DEFAULT_TIMEOUT = 120.0  # 2 minutes (GigaChat can be slow)

# GigaChat API endpoints
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1"

# Default max_tokens for GigaChat
DEFAULT_MAX_TOKENS = 4096


@dataclass
class TokenUsage:
    """Token usage from LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class GigaChatToken:
    """GigaChat OAuth2 access token."""

    access_token: str
    expires_at: float  # Unix timestamp


class GigaChatClient:
    """Async client adapter for GigaChat API (Sber).

    Features:
    - OAuth2 authentication with automatic token refresh
    - Async chat completions (streaming and non-streaming)
    - Token usage tracking
    - Proper error handling
    """

    def __init__(
        self,
        auth_key: str | None = None,
        scope: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        verify_ssl: bool = False,  # GigaChat uses self-signed certs
    ):
        """Initialize GigaChat client.

        Args:
            auth_key: GigaChat authorization key (base64 encoded client_id:client_secret)
            scope: API scope (GIGACHAT_API_PERS, GIGACHAT_API_B2B, GIGACHAT_API_CORP)
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates (default False for GigaChat)
        """
        self._auth_key = auth_key or getattr(settings, "gigachat_auth_key", None)
        self._scope = scope or getattr(settings, "gigachat_scope", "GIGACHAT_API_PERS")
        self._timeout = timeout
        self._verify_ssl = verify_ssl
        self._last_usage: TokenUsage | None = None
        self._token: GigaChatToken | None = None
        self._client: httpx.AsyncClient | None = None

        if not self._auth_key:
            logger.warning("GigaChat authorization key not configured")

    def _get_ssl_context(self) -> ssl.SSLContext | bool:
        """Get SSL context for GigaChat (self-signed certs)."""
        if self._verify_ssl:
            return True
        # GigaChat uses self-signed certificates
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                verify=self._get_ssl_context(),
            )
        return self._client

    async def _get_access_token(self) -> str:
        """Get valid access token, refreshing if needed.

        Returns:
            Valid access token string

        Raises:
            LLMError: If authentication fails
        """
        # Check if we have a valid token
        if self._token and self._token.expires_at > time.time() + 60:
            return self._token.access_token

        if not self._auth_key:
            raise LLMError("GigaChat authorization key not configured")

        client = await self._get_client()

        try:
            response = await client.post(
                GIGACHAT_AUTH_URL,
                headers={
                    "Authorization": f"Basic {self._auth_key}",
                    "RqUID": str(uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"scope": self._scope},
            )

            if response.status_code != 200:
                logger.error(f"GigaChat auth failed: {response.status_code}")
                raise LLMError(f"GigaChat authentication failed: {response.status_code}")

            data = response.json()
            self._token = GigaChatToken(
                access_token=data["access_token"],
                expires_at=data["expires_at"] / 1000,  # Convert ms to seconds
            )

            logger.debug("GigaChat token obtained successfully")
            return self._token.access_token

        except httpx.TimeoutException:
            raise LLMTimeoutError("GigaChat authentication timed out")
        except Exception as e:
            logger.error(f"GigaChat auth error: {e}")
            raise LLMError(f"GigaChat authentication error: {e}")

    def get_usage(self) -> TokenUsage | None:
        """Get token usage from last request."""
        return self._last_usage

    async def send_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncGenerator[str, None]:
        """Send message request to GigaChat.

        Args:
            model: Model name (GigaChat, GigaChat-Pro, GigaChat-Max)
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Response content (str) or async generator of chunks

        Raises:
            LLMError: For API errors
            LLMTimeoutError: For timeout errors
        """
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
        """Send non-streaming message request."""
        access_token = await self._get_access_token()
        client = await self._get_client()

        # Set default max_tokens if not provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        payload = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        try:
            response = await client.post(
                f"{GIGACHAT_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.status_code != 200:
                self._handle_api_error(response.status_code, response.text)

            data = response.json()

            # Track usage
            usage = data.get("usage", {})
            self._last_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

            # Extract content
            content = data["choices"][0]["message"]["content"]

            logger.debug(f"GigaChat response: model={model}, tokens={self._last_usage}")

            return content

        except httpx.TimeoutException:
            logger.error(f"GigaChat timeout after {self._timeout}s")
            raise LLMTimeoutError(f"GigaChat request timed out after {self._timeout}s")
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"GigaChat unexpected error: {e}")
            raise LLMError(f"GigaChat error: {e}")

    async def _stream_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream message response."""
        access_token = await self._get_access_token()
        client = await self._get_client()

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with client.stream(
                "POST",
                f"{GIGACHAT_API_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    self._handle_api_error(response.status_code, error_text.decode())

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        import json

                        data = json.loads(data_str)

                        # Extract content delta
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content

                        # Track usage if provided
                        usage = data.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)

                    except json.JSONDecodeError:
                        continue

            # Update usage after stream completes
            self._last_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        except httpx.TimeoutException:
            logger.error(f"GigaChat streaming timeout after {self._timeout}s")
            raise LLMTimeoutError(f"GigaChat streaming timed out after {self._timeout}s")
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"GigaChat streaming error: {e}")
            raise LLMError(f"GigaChat streaming error: {e}")

    def _handle_api_error(self, status_code: int, error_text: str) -> None:
        """Handle GigaChat API errors."""
        logger.error(f"GigaChat API error: status={status_code}")

        if status_code == 401:
            self._token = None  # Force token refresh
            raise LLMError("GigaChat authentication failed - check credentials")
        elif status_code == 429:
            raise LLMError("GigaChat rate limit exceeded. Please retry later.")
        elif status_code >= 500:
            raise LLMError(f"GigaChat server error: {status_code}")
        else:
            raise LLMError(f"GigaChat API error ({status_code}): {error_text[:200]}")

    async def close(self) -> None:
        """Close the client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class GigaChatProvider:
    """GigaChat provider implementing LLMProvider protocol.

    This adapter bridges GigaChatClient to the MessageService's LLMProvider protocol.
    """

    def __init__(self, client: GigaChatClient | None = None):
        """Initialize with optional client (creates new one if not provided)."""
        self._client = client or GigaChatClient()

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, int, int]:
        """Generate a response from GigaChat.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            config: Optional generation config

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
        """Generate a streaming response from GigaChat.

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
