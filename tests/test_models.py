"""Test-specific SQLAlchemy models with test_ prefix.

These models mirror the production models but use test_ prefixed table names
to ensure complete isolation from production data.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class TestBase(DeclarativeBase):
    """Base class for all test ORM models."""

    pass


class TestDialog(TestBase):
    """Test Dialog entity - conversation with LLM."""

    __tablename__ = "test_dialogs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    messages: Mapped[list["TestMessage"]] = relationship(
        "TestMessage", back_populates="dialog", cascade="all, delete-orphan"
    )


class TestMessage(TestBase):
    """Test Message entity - user or assistant message in a dialog."""

    __tablename__ = "test_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dialog_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("test_dialogs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )

    # Relationships
    dialog: Mapped["TestDialog"] = relationship("TestDialog", back_populates="messages")


class TestTokenBalance(TestBase):
    """Test Token balance entity - user's current token balance and limit."""

    __tablename__ = "test_token_balances"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    balance: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    limit: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class TestTokenTransaction(TestBase):
    """Test Token transaction entity - audit log for all token changes."""

    __tablename__ = "test_token_transactions"
    __table_args__ = (
        UniqueConstraint("message_id", "reason", name="test_uq_message_reason"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    amount: Mapped[int] = mapped_column(BigInteger, nullable=False)
    reason: Mapped[str] = mapped_column(String(50), nullable=False)
    dialog_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("test_dialogs.id", ondelete="SET NULL"), nullable=True
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("test_messages.id", ondelete="SET NULL"), nullable=True
    )
    admin_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )


class TestModel(TestBase):
    """Test Model entity - available LLM models with pricing."""

    __tablename__ = "test_models"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    cost_per_1k_prompt_tokens: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    cost_per_1k_completion_tokens: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
