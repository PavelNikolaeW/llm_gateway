import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from httpx import AsyncClient
from app.main import app

# В реальных тестах нужно мокнуть verify_bearer_token или поднять тестовый auth-сервис.
# Здесь просто кладём заголовок, но авторизацию придётся настроить отдельно.


async def _auth_headers():
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_health():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_chat_dummy_smoke():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "options": {"include_profile": False, "model": "dummy"},
            "stream": False,
        }
        r = await ac.post("/chat", json=payload, headers=await _auth_headers())
        assert r.status_code == 200
        assert r.json()["message"]["content"].startswith("[dummy] Hello")
