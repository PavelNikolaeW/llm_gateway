from uuid import UUID

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, List

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    role: Role
    content: str

    def get_message_object(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content
        }


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
