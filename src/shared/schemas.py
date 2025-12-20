"""Pydantic schemas for DTOs."""
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


class TokenStatsResponse(BaseModel):
    """Schema for token stats response (GET /users/me/tokens)."""

    balance: int
    total_used: int
    limit: int | None = None


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

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


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


# Message Schemas


class MessageCreate(BaseModel):
    """Schema for creating a message."""

    content: str = Field(..., min_length=1, description="Message content")

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str) -> str:
        """Validate content doesn't exceed max length from settings."""
        from src.config.settings import settings

        if len(v) > settings.max_content_length:
            raise ValueError(
                f"Content exceeds maximum length of {settings.max_content_length} characters"
            )
        return v


class MessageResponse(BaseModel):
    """Schema for message response."""

    id: UUID
    dialog_id: UUID
    role: str  # 'user' or 'assistant'
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StreamChunk(BaseModel):
    """Schema for streaming response chunk."""

    content: str
    done: bool = False
    message_id: UUID | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# Message Events


class MessageSentEvent(BaseModel):
    """Event emitted when a user message is sent."""

    event_type: str = "message_sent"
    dialog_id: UUID
    user_id: int
    message_id: UUID
    content_length: int
    timestamp: datetime


class LLMResponseEvent(BaseModel):
    """Event emitted when LLM response is received."""

    event_type: str = "llm_response_received"
    dialog_id: UUID
    user_id: int
    message_id: UUID
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    timestamp: datetime


# Admin Schemas


class UserStatsResponse(BaseModel):
    """Schema for user stats in admin list."""

    user_id: int
    dialog_count: int
    total_tokens_used: int
    balance: int
    limit: int | None = None


class UserDetailsResponse(BaseModel):
    """Schema for detailed user info in admin view."""

    user_id: int
    dialog_count: int
    total_tokens_used: int
    balance: int
    limit: int | None = None
    last_activity: datetime | None = None


class SetLimitRequest(BaseModel):
    """Schema for setting user token limit."""

    limit: int | None = Field(None, ge=0, description="Token limit (null = unlimited)")


class TopUpTokensRequest(BaseModel):
    """Schema for admin token top-up/deduct."""

    amount: int = Field(..., description="Amount of tokens (positive = add, negative = deduct)")


class AdminActionEvent(BaseModel):
    """Event emitted for admin actions."""

    event_type: str
    admin_user_id: int
    target_user_id: int
    action: str
    details: dict[str, Any]
    timestamp: datetime


class ModelUsageStats(BaseModel):
    """Schema for model usage in stats response."""

    model: str
    usage: int


class GlobalStatsResponse(BaseModel):
    """Schema for global usage statistics (GET /admin/stats)."""

    total_tokens: int
    active_users: int
    top_models: list[ModelUsageStats]
    avg_latency_ms: float


# Export/Import Schemas


class MessageExport(BaseModel):
    """Schema for exported message."""

    role: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime


class DialogExport(BaseModel):
    """Schema for exported dialog."""

    id: UUID
    title: str | None
    system_prompt: str | None
    model_name: str
    agent_config: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageExport]


class ExportResponse(BaseModel):
    """Schema for export response."""

    version: str = "1.0"
    exported_at: datetime
    user_id: int
    dialog_count: int
    message_count: int
    dialogs: list[DialogExport]


class DialogImport(BaseModel):
    """Schema for importing a dialog."""

    title: str | None = None
    system_prompt: str | None = None
    model_name: str | None = None
    agent_config: dict[str, Any] | None = None
    messages: list[MessageExport] = Field(default_factory=list)


class ImportRequest(BaseModel):
    """Schema for import request."""

    dialogs: list[DialogImport]


class ImportResult(BaseModel):
    """Schema for import result."""

    dialogs_imported: int
    messages_imported: int
    errors: list[str] = Field(default_factory=list)


# Audit Log Schemas


class AuditLogResponse(BaseModel):
    """Schema for audit log entry response."""

    id: int
    user_id: int | None
    action: str
    resource_type: str
    resource_id: str | None
    details: dict[str, Any] | None
    ip_address: str | None
    user_agent: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
