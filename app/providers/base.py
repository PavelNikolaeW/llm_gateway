from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any
import fnmatch


class LLMProvider(ABC):
    """Generic LLM adapter interface.

    Implement providers for OpenAI-compatible, Ollama, Anthropic, vLLM, etc.
    """

    name: str = "base"

    @abstractmethod
    async def chat(self, *, messages: list[dict], model: str, **kwargs) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def stream(self, *, messages: list[dict], model: str, **kwargs) -> AsyncIterator[str]:
        ...

    def supports(self, capability: str) -> bool:
        return False


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
