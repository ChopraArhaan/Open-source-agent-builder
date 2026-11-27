"""Request and response schemas for the FastAPI layer."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.agent_builder.memory.models import StoredMessage
from src.agent_builder.models import LLMMessage, RunEvent


class RunRequest(BaseModel):
    """Payload for invoking an agent."""

    prompt: str = Field(min_length=1, description="User instruction or prompt")
    session_messages: list[LLMMessage] = Field(
        default_factory=list,
        description="Optional prior conversation context",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional persisted session ID for multi-turn memory",
    )


class RunResponse(BaseModel):
    """Synchronous outcome of an agent execution."""

    agent_name: str
    final_answer: str
    events: list[RunEvent]
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    """Payload for creating a new persisted conversation session."""

    agent: str = Field(min_length=1, description="Agent name bound to this session")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional JSON-safe session metadata",
    )


class SessionResponse(BaseModel):
    """Full representation of a persisted session including its messages."""

    session_id: str
    agent: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    messages: list[StoredMessage]


class SessionSummaryResponse(BaseModel):
    """Compact summary of a persisted session for list endpoints."""

    session_id: str
    agent: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    message_count: int


class AgentSummary(BaseModel):
    """High-level summary of an available agent."""

    name: str
    version: str
    description: str
    tools: list[str]
    provider: str
    model: str


class ToolSummary(BaseModel):
    """Specification of an available tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    timeout_seconds: float
