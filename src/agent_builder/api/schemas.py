"""Request and response schemas for the FastAPI layer."""

from typing import Any

from pydantic import BaseModel, Field

from src.agent_builder.models import LLMMessage, RunEvent


class RunRequest(BaseModel):
    """Payload for invoking an agent."""

    prompt: str = Field(min_length=1, description="User instruction or prompt")
    session_messages: list[LLMMessage] = Field(
        default_factory=list,
        description="Optional prior conversation context",
    )


class RunResponse(BaseModel):
    """Synchronous outcome of an agent execution."""

    agent_name: str
    final_answer: str
    events: list[RunEvent]


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
