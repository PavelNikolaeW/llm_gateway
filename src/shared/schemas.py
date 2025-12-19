"""Pydantic schemas for DTOs."""
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DialogCreate(BaseModel):
    """Schema for creating a dialog."""

    title: str | None = Field(None, max_length=255)
    system_prompt: str | None = None
    model_name: str | None = None
    agent_config: dict[str, Any] | None = None


class DialogResponse(BaseModel):
    """Schema for dialog response."""

    id: UUID
    user_id: int
    title: str | None
    system_prompt: str | None
    model_name: str
    agent_config: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DialogList(BaseModel):
    """Schema for paginated dialog list."""

    items: list[DialogResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
