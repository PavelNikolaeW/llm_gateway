from typing import AsyncIterator, Dict, Any
from .base import LLMProvider


class DummyProvider(LLMProvider):
    name = "dummy"

    async def chat(self, *, messages: list[dict], model: str, **kwargs) -> Dict[str, Any]:
        return {
            "text": "[dummy] " + messages[-1]["content"],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    async def stream(self, *, messages: list[dict], model: str, **kwargs) -> AsyncIterator[str]:
        yield "[dummy stream] " + messages[-1]["content"]
