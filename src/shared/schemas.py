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


# Token Schemas


class TokenBalanceResponse(BaseModel):
    """Schema for token balance response."""

    user_id: int
    balance: int
    limit: int | None = None
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenDeductRequest(BaseModel):
    """Schema for token deduction request."""

    amount: int = Field(..., gt=0, description="Amount of tokens to deduct")
    dialog_id: UUID
    message_id: UUID


class TokenTopUpRequest(BaseModel):
    """Schema for admin token top-up/deduct request."""

    amount: int = Field(..., description="Amount to add (positive) or remove (negative)")


class TokenTransactionResponse(BaseModel):
    """Schema for token transaction response."""

    id: int
    user_id: int
    amount: int
    reason: str
    dialog_id: UUID | None = None
    message_id: UUID | None = None
    admin_user_id: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class TokenEvent(BaseModel):
    """Schema for token events emitted by the service."""

    event_type: str  # 'tokens_deducted', 'balance_exhausted'
    user_id: int
    amount: int | None = None
    new_balance: int
    reason: str | None = None
    dialog_id: UUID | None = None
    message_id: UUID | None = None
    timestamp: datetime


# Model Schemas


class ModelMetadata(BaseModel):
    """Schema for model metadata response."""

    name: str
    provider: str
    cost_per_1k_prompt_tokens: float
    cost_per_1k_completion_tokens: float
    context_window: int
    enabled: bool

    class Config:
        from_attributes = True


class CostEstimate(BaseModel):
    """Schema for cost estimation response."""

    model_name: str
    prompt_tokens: int
    completion_tokens: int
    prompt_cost: float
    completion_cost: float
    total_cost: float


# Agent Config Schemas


class AgentConfig(BaseModel):
    """Schema for agent configuration.

    Configurable parameters for LLM behavior.
    """

    temperature: float | None = Field(None, ge=0.0, le=1.0, description="Sampling temperature (0-1)")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens to generate")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Top-p sampling (0-1)")
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Presence penalty (-2 to 2)")
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0, description="Frequency penalty (-2 to 2)")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")


class AgentTypeInfo(BaseModel):
    """Schema for agent type information."""

    name: str
    description: str
    config: AgentConfig
