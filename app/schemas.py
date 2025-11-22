from uuid import UUID

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any


class ChatMessage(BaseModel):
    """Универсальная структура сообщения для разных LLM."""

    role: str
    content: Any = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Произвольные поля провайдера (например, function_call, audio и т.д.)",
    )

    def get_message_object(self) -> Dict:
        msg: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.name:
            msg["name"] = self.name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        msg.update(self.extra)
        return msg


class ChatOptions(BaseModel):
    model: str | None = None
    system_prompt: str | None = None
    # optional data fetch knobs
    include_profile: bool = False
    include_knowledge_items: bool = False
    data_query: Optional[str] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    messages: List[ChatMessage]
    system_prompt: str | None = None
    options: ChatOptions = Field(default_factory=ChatOptions)
    stream: bool = False
    user_id: int


class ChatChunk(BaseModel):
    type: Literal["delta", "final", "error"]
    delta: Optional[str] = None
    final: Optional[str] = None
    error: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    message: ChatMessage


class CheckTokenRequest(BaseModel):
    token: str


class CheckTokenResponse(BaseModel):
    is_valid: bool
