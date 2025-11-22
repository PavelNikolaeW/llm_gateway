from typing import List, Dict

from sqlalchemy import String, DateTime, Text, JSON, ForeignKey, Integer
from sqlalchemy.orm import relationship, Mapped, mapped_column, Session
from datetime import datetime, timezone

from app.config import settings
from app.db import Base


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    messages: Mapped[list["Message"]] = relationship(back_populates="conversation")
    system_prompt: Mapped[str] = mapped_column(String, default=settings.SYSTEM_PROMPT)

    def get_messages(self, new_messages):
        msgs: List[dict] = [{
            'role': 'system',
            'content': self.system_prompt
        }]
        for message in self.messages:
            msgs.append(message.get_message_object())
        for message in new_messages:
            msgs.append(message.get_message_object())
        return msgs


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant" | "system" | "tool"
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    def get_message_object(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content
        }


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
