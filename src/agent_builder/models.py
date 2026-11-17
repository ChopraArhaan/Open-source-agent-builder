import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


class ToolSpec(BaseModel):
    """Specification of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    timeout_seconds: float = Field(default=30.0, gt=0)

    def to_openai_schema(self) -> dict[str, Any]:
        """Provider payload. Deliberately excludes runtime-only fields."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str = Field(default_factory=_new_call_id)
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Outcome of a tool execution. Never raises; captures exceptions."""

    call_id: str
    tool_name: str
    output: Any = None
    is_error: bool = False
    error_message: str | None = None
    duration_ms: float | None = None

    @model_validator(mode="after")
    def _error_needs_message(self) -> "ToolResult":
        if self.is_error and not self.error_message:
            raise ValueError("error_message is required when is_error=True")
        return self

    def as_text(self) -> str:
        """String form sent back to the LLM as the observation."""
        if self.is_error:
            return f"Error: {self.error_message}"
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, default=str)


class LLMMessage(BaseModel):
    """Provider-independent chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def _tool_role_needs_call_id(self) -> "LLMMessage":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        return self


class LLMResponse(BaseModel):
    """Standardized completion from any provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    raw: dict[str, Any] | None = None  # provider payload, for debugging


RunEventType = Literal["thought", "tool_call", "observation", "final", "error"]


class RunEvent(BaseModel):
    """Single event yielded during agent execution."""

    type: RunEventType
    step: int | None = None  # loop iteration, for grouping on the client
    content: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
