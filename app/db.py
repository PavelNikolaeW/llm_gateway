from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, selectinload

from .config import settings

# In-memory stub storage
_conversations: dict[str, object] = {}
_messages: list[object] = []
_logs: list[object] = []


class StubSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        from .orm_models import Conversation, Message, RequestLog

        if isinstance(obj, Conversation):
            _conversations[obj.id] = obj
        elif isinstance(obj, Message):
            _messages.append(obj)
            conv = _conversations.get(obj.conversation_id)
            if conv:
                conv.messages.append(obj)
        elif isinstance(obj, RequestLog):
            _logs.append(obj)

    def add_all(self, objs):
        for obj in objs:
            self.add(obj)

    async def scalars(self, result):
        return result

    async def commit(self):
        return None


if settings.STUB_DB_ENABLED:
    engine = None
    SessionLocal = None

    class Base(DeclarativeBase):
        pass
else:
    if settings.PG_DSN.startswith("sqlite://"):
        engine = create_async_engine(
            settings.PG_DSN.replace("sqlite://", "sqlite+aiosqlite://"), pool_pre_ping=True
        )
    else:
        engine = create_async_engine(settings.PG_DSN, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    class Base(DeclarativeBase):
        pass


async def get_session() -> AsyncSession:
    if settings.STUB_DB_ENABLED:
        async with StubSession() as session:
            yield session
    else:
        async with SessionLocal() as session:  # type: ignore[arg-type]
            yield session


def get_stub_conversation(cid: str, user_id: int):
    conv = _conversations.get(cid)
    if conv and getattr(conv, "user_id", None) == user_id:
        return conv
    return None


def create_stub_conversation(conversation):
    _conversations[conversation.id] = conversation
    return conversation.id


async def fetch_conversation(session: AsyncSession, cid: str, user_id: int, with_messages: bool = False):
    if settings.STUB_DB_ENABLED:
        return get_stub_conversation(cid, user_id)

    from .orm_models import Conversation

    query = select(Conversation).where(
        Conversation.id == cid,
        Conversation.user_id == user_id,
    )
    if with_messages:
        query = query.options(selectinload(Conversation.messages))

    result = await session.scalars(query)
    return result.first()
