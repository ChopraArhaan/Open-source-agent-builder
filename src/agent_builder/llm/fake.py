"""Fake LLM provider for zero-network offline and engine testing."""

from collections import deque
from typing import Any

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec


class FakeLLMProvider(BaseLLMProvider):
    """Predictable, scripted LLM provider for unit and integration testing.

    Allows tests to enqueue canned responses, simulated tool calls, or exceptions
    to test the engine's behavior under various LLM responses without making
    any real network requests.
    """

    def __init__(
        self,
        model: str = "fake-model",
        responses: list[LLMResponse | str | list[ToolCall] | Exception] | None = None,
        default_response: str = "Default fake response",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self._responses: deque[LLMResponse | str | list[ToolCall] | Exception] = deque(
            responses or []
        )
        self.default_response = default_response
        self.call_history: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    def add_response(
        self, response: LLMResponse | str | list[ToolCall] | Exception
    ) -> None:
        """Enqueue a canned response, tool call list, or exception."""
        self._responses.append(response)

    def set_responses(
        self, responses: list[LLMResponse | str | list[ToolCall] | Exception]
    ) -> None:
        """Replace all queued responses."""
        self._responses = deque(responses)

    @property
    def call_count(self) -> int:
        return len(self.call_history)

    def last_call(self) -> dict[str, Any] | None:
        return self.call_history[-1] if self.call_history else None

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Record the call and return the next scripted response."""
        self.call_history.append({
            "messages": messages,
            "tools": tools or [],
            "kwargs": kwargs,
        })

        if not self._responses:
            return LLMResponse(
                content=self.default_response,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            )

        next_item = self._responses.popleft()

        if isinstance(next_item, Exception):
            raise next_item

        if isinstance(next_item, LLMResponse):
            return next_item

        if isinstance(next_item, str):
            return LLMResponse(
                content=next_item,
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": len(next_item.split()), "total_tokens": 10 + len(next_item.split())},
            )

        if isinstance(next_item, list):
            # List of ToolCall objects
            return LLMResponse(
                content=None,
                tool_calls=next_item,
                finish_reason="tool_calls",
                usage={"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35},
            )

        raise TypeError(f"Unsupported canned response type: {type(next_item)}")
