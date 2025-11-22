from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Protocol
import fnmatch
import math


class Usage(Protocol):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        *,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Возвращает минимум:
        {
          "message": { "role": "assistant", "content": "..." },
          "usage": {
             "prompt_tokens": ...,
             "completion_tokens": ...,
             "total_tokens": ...,
             "cost": ... (опционально)
          },
          "raw": ...
        }
        """
        ...

    @abstractmethod
    async def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        ...

    def supports(self, capability: str) -> bool:
        return False

    @abstractmethod
    def count_tokens(
        self,
        *,
        messages: list[dict],
        model: str,
    ) -> Dict[str, int]:
        ...


def approx_token_count_from_messages(messages: list[dict]) -> int:
    text_chunks: list[str] = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            text_chunks.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str):
                        text_chunks.append(t)
    text = "\n".join(text_chunks)
    words = text.split()
    return math.ceil(len(words) * 1.3)


class ProviderRegistry:
    def __init__(self):
        self._by_name: dict[str, LLMProvider] = {}
        self._model_routes: list[tuple[str, str]] = []  # (pattern, provider_name)

    def register(self, name: str, provider: LLMProvider):
        self._by_name[name] = provider

    def route(self, model_pattern: str, provider_name: str):
        """Map glob-like pattern to provider (e.g., 'gpt-*' -> 'openai')."""
        self._model_routes.append((model_pattern, provider_name))

    def provider_for_model(self, model: str) -> LLMProvider:
        for pat, pname in self._model_routes:
            if fnmatch.fnmatch(model, pat):
                return self._by_name[pname]
        # fallback to first provider
        if not self._by_name:
            raise RuntimeError("No LLM providers registered")
        return next(iter(self._by_name.values()))


registry = ProviderRegistry()
