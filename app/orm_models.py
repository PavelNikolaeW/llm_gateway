import uuid
from typing import List, Dict, Any

from sqlalchemy import String, DateTime, JSON, ForeignKey, Integer, Text, UUID, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column, Session
from datetime import datetime, timezone

from app.config import settings
from app.db import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[str] = mapped_column(String, index=True)

    title: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    total_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_cost: Mapped[float] = mapped_column(Float, default=0.0)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    role: Mapped[str] = mapped_column(String, index=True)  # system/user/assistant/tool
    content: Mapped[Any] = mapped_column(JSONB)  # str или сложная структура

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )

    provider_name: Mapped[str | None] = mapped_column(String, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)

    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    meta: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    def get_message_object(self) -> Dict:
        msg: Dict[str, Any] = {'role': self.role}
        if self.content is not None:
            msg['content'] = self.content
        if self.name:
            msg['name'] = self.name
        if self.tool_call_id:
            msg['tool_call_id'] = self.tool_call_id
        if self.tool_calls:
            msg['tool_calls'] = self.tool_calls
        if self.meta:
            msg.update(self.meta)
        return msg


class RequestLog(Base):
    __tablename__ = "request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    model: Mapped[str] = mapped_column(String)
    latency_ms: Mapped[int] = mapped_column(Integer)
    tokens_prompt: Mapped[int] = mapped_column(Integer)
    tokens_completion: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String)  # OK/ERROR
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
