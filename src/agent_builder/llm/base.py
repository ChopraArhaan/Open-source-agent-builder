"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from src.agent_builder.models import LLMMessage, LLMResponse, ToolSpec


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM provider integrations."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.extra_config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'openai', 'fake')."""
        ...

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request to the LLM provider.

        Args:
            messages: List of standardized conversation messages.
            tools: Optional list of available tool specifications.
            **kwargs: Provider-specific execution options (e.g. tool_choice).

        Returns:
            Standardized LLMResponse with content and/or tool_calls.
        """
        ...
