import httpx
from typing import AsyncIterator, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from .base import LLMProvider
from ..config import settings


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or (settings.LLM_BASE_URL or "https://api.openai.com/v1")
        self.api_key = api_key or settings.LLM_API_KEY

    def _headers(self):
        hdrs = {"Content-Type": "application/json"}
        if self.api_key:
            hdrs["Authorization"] = f"Bearer {self.api_key}"
        return hdrs

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(0.2, 1.5))
    async def chat(self, *, messages: list[dict], model: str, **kwargs) -> Dict[str, Any]:
        payload = {"model": model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage")
            return {"text": text, "usage": usage}

    async def stream(self, *, messages: list[dict], model: str, **kwargs) -> AsyncIterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = r.json_loader(chunk)
                        delta = obj["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except Exception:
                        continue
