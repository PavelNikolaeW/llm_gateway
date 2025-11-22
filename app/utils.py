import uuid
from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from .orm_models import Conversation, Message, RequestLog
from .schemas import ChatRequest


@asynccontextmanager
async def measure_and_log(session: AsyncSession, user_id: int, model: str):
    start = perf_counter()
    try:
        yield
        status, error = "OK", "None"
    except HTTPException as e:
        status, error = f"HTTP_{e.status_code}", e.detail
        raise
    except Exception as e:
        status, error = "ERROR", str(e)
        raise
    finally:
        dur_ms = int((perf_counter() - start) * 1000)
        session.add(
            RequestLog(
                user_id=str(user_id),
                model=model,
                latency_ms=dur_ms,
                tokens_prompt=0,
                tokens_completion=0,
                status=status,
                error=error,
            )
        )
        await session.commit()


async def ensure_conversation(
        session: AsyncSession,
        request: ChatRequest,
        user_id: int
) -> str:
    if request.conversation_id:
        try:
            uuid.UUID(request.conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Wrong conversation ID")
        return request.conversation_id
    cid = str(uuid.uuid4())
    session.add(Conversation(id=cid, user_id=user_id, system_prompt=request.system_prompt))
    await session.commit()
    return cid


async def save_message(session: AsyncSession, conversation_id: str, role: str, content: str):
    mid = str(uuid.uuid4())
    print('save_message', mid, content)
    session.add(Message(id=mid, conversation_id=conversation_id, role=role, content=content))
    await session.commit()
