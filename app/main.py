import uuid

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncIterator
import json

from .config import settings
from .auth import verify_bearer_token
from .db import Base, engine, fetch_conversation, get_session
from .orm_models import Message, Conversation
from .schemas import ChatRequest, ChatResponse, ChatMessage, ChatChunk, CheckTokenResponse
from .providers.base import registry
from .providers.openai_compatible import OpenAICompatProvider
from .providers.ollama import OllamaProvider
from .providers.dummy import DummyProvider
from .utils import measure_and_log, ensure_conversation, save_message
from .ratelimit import enforce
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI(title="LLM Gateway")

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register providers and routing rules (patterns are glob-like)
registry.register("dummy", DummyProvider())
registry.register("openai", OpenAICompatProvider())
registry.register("ollama", OllamaProvider())
registry.route("dummy", "dummy")
registry.route("gpt-*", "openai")
registry.route("*lama*", "ollama")  # e.g., "llama3", "mistral" via Ollama (подправь под свои модели)
registry.route("bartowski/Codestral-22B-v0.1-GGUF",
               "ollama")  # e.g., "llama3", "mistral" via Ollama (подправь под свои модели)
registry.route("*", "openai")  # default fallback


@app.on_event("startup")
async def on_startup():
    if engine:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get('/check_token')
async def check_token(
        authorization: str | None = Header(default=None),
):
    '''Тест верификации'''
    res = await verify_bearer_token(authorization)
    return CheckTokenResponse(is_valid=res)


@app.get("/conversations/{cid}/messages")
async def list_messages(
        cid: str,
        user_id: int = Depends(verify_bearer_token),
        session: AsyncSession = Depends(get_session),
):
    conversation = await fetch_conversation(session, cid, user_id, with_messages=True)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return [m.get_message_object() for m in conversation.messages]


@app.post("/chat", response_model=ChatResponse)
async def chat(
        req: ChatRequest,
        user_id: int = Depends(verify_bearer_token),
        session: AsyncSession = Depends(get_session),
):
    model = req.options.model or settings.DEFAULT_MODEL
    prov = registry.provider_for_model(model)
    async with measure_and_log(session, user_id=user_id, model=model):
        # 1) Подгружаем диалог
        cid = await ensure_conversation(session, req, user_id)
        conversation = await fetch_conversation(session, cid, user_id, with_messages=True)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        messages = conversation.get_messages(req.messages)
        # Отправляем в ЛЛМ
        res = await prov.chat(messages=messages, model=model)
        # Сохраняем ответ
        if isinstance(res, dict) and "choices" in res:
            for choice in res.get("choices", []):
                msg_data = choice.get("message", {})
                req.messages.append(ChatMessage(**msg_data))
        else:
            text = res.get("text") if isinstance(res, dict) else str(res)
            extra = res.get("usage") if isinstance(res, dict) else None
            req.messages.append(ChatMessage(role="assistant", content=text, extra={"usage": extra} if extra else {}))

        session.add_all(
            [
                Message(
                    id=str(uuid.uuid4()),
                    conversation_id=cid,
                    role=message.role,
                    content=message.content,
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    tool_calls=message.tool_calls,
                    meta=message.extra or None,
                )
                for message in req.messages
            ]
        )
        await session.commit()
        return ChatResponse(
            conversation_id=cid,
            message=req.messages[-1],
        )


@app.post("/chat/stream")
async def chat_stream(
        req: ChatRequest,
        user_id: str = Depends(verify_bearer_token),
        session: AsyncSession = Depends(get_session),
        authorization: str | None = Header(default=None),
):
    model = req.options.model or settings.DEFAULT_MODEL
    enforce(
        user_id,
        key=f"chat-stream:{model}",
        rps=settings.RATE_LIMIT_RPS,
        burst=settings.RATE_LIMIT_BURST,
    )

    prov = registry.provider_for_model(model)
    token = authorization.split(" ", 1)[1] if authorization else ""

    async def gen() -> AsyncIterator[bytes]:
        async with measure_and_log(session, user_id=user_id, model=model):
            cid = await ensure_conversation(session, req.conversation_id, user_id)
            llm_messages = build_messages(req.messages, ctx)
            await save_message(session, cid, role="user", content=req.messages[-1].content)

            async for delta in prov.stream(messages=llm_messages, model=model):
                yield (json.dumps(ChatChunk(type="delta", delta=delta).dict()) + "\n").encode()
            yield (json.dumps(ChatChunk(type="final", final="[stream complete]").dict()) + "\n").encode()

    return StreamingResponse(gen(), media_type="text/event-stream")
