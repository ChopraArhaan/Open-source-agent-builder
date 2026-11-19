"""Declarative agent configuration schemas using Pydantic v2."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for an agent's LLM provider and generation parameters."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class ExecutionConfig(BaseModel):
    """Execution runtime parameters and loop guardrails."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=10, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_repeats: int = Field(default=3, ge=2)


class AgentConfig(BaseModel):
    """Declarative specification for an autonomous agent."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Unique agent identifier / URL-safe slug",
    )
    version: str = "1.0.0"
    description: str = ""
    llm: LLMConfig
    system_prompt: str
    tools: list[str] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("tools")
    @classmethod
    def _validate_unique_tools(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for tool in v:
            if tool in seen and tool not in duplicates:
                duplicates.append(tool)
            seen.add(tool)
        if duplicates:
            raise ValueError(f"Duplicate tools found in configuration: {duplicates}")
        return v
