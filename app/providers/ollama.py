from typing import AsyncIterator, Dict, Any
from .base import LLMProvider
from ..config import settings
from openai import AsyncOpenAI


class LlamaProvider(LLMProvider):
    name = "llama"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    async def chat(
        self,
        *,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        mdl = model or self.default_model
        if not mdl:
            raise ValueError("model is required")

        resp = await self.client.chat.completions.create(
            model=mdl,
            messages=messages,
            **kwargs,
        )

        choice = resp.choices[0]
        msg = choice.message
        usage = resp.usage

        return {
            "message": {
                "role": msg.role,
                "content": msg.content,
            },
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "cost": None,
            },
            "raw": resp.model_dump(),
        }

    async def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        mdl = model or self.default_model
        if not mdl:
            raise ValueError("model is required")

        stream = await self.client.chat.completions.create(
            model=mdl,
            messages=messages,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def count_tokens(
        self,
        *,
        messages: list[dict],
        model: str,
    ) -> Dict[str, int]:
        prompt_tokens = approx_token_count_from_messages(messages)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }
