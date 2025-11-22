import httpx
from typing import AsyncIterator, Dict, Any
from .base import LLMProvider
from ..config import settings


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or (settings.OLLAMA_BASE_URL or "http://localhost:11434")

    async def chat(self, *, messages: list[dict], model: str, **kwargs) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json={"model": model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            return r.json()

    async def stream(self, *, messages: list[dict], model: str, **kwargs) -> AsyncIterator[str]:
        prompt = "\n".join(
            [m["content"] for m in messages if m["role"] in ("system", "user")]
        )
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = r.json_loader(line)
                        part = obj.get("response")
                        if part:
                            yield part
                        if obj.get("done"):
                            break
                    except Exception:
                        continue
