"""Native Anthropic LLM provider using the Anthropic Messages API."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Callable
from typing import Any

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic's Claude models using the native Messages API."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str | None = None,
        client: Any | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 4096,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep_func: Callable[[float], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "Anthropic SDK not installed. Install with: pip install agent-builder[anthropic]"
            ) from e

        # Ensure max_tokens is present since Anthropic Messages API requires it
        effective_max_tokens = max_tokens if max_tokens is not None else 4096

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=effective_max_tokens,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

        self._api_key = api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        if client is None and not self._api_key:
            raise ValueError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key."
            )

        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._sleep_func = sleep_func or asyncio.sleep

        if client is not None:
            self._client = client
        else:
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=timeout_seconds,
            )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r}, max_tokens={self.max_tokens})"

    def _format_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert standardized LLMMessages to Anthropic (system_prompt, messages) format.

        Handles:
        - System messages extracted to top-level system parameter.
        - Tool results formatted as tool_result content blocks in user messages.
        - Assistant tool calls formatted as tool_use content blocks.
        - Merging consecutive turns with the same role to strictly enforce alternating roles.
        - Ensuring conversation begins with a user turn.
        """
        system_parts: list[str] = []
        raw_turns: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == "tool":
                # In Anthropic Messages API, tool results are sent inside a user turn
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
                raw_turns.append({
                    "role": "user",
                    "content": [tool_result_block],
                })
            elif msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                raw_turns.append({
                    "role": "assistant",
                    "content": blocks,
                })
            else:
                # user
                raw_turns.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.content or ""}],
                })

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # Merge consecutive turns of identical roles
        merged_turns: list[dict[str, Any]] = []
        for turn in raw_turns:
            if merged_turns and merged_turns[-1]["role"] == turn["role"]:
                prev_content = merged_turns[-1]["content"]
                curr_content = turn["content"]
                merged_turns[-1]["content"] = prev_content + curr_content
            else:
                merged_turns.append(turn)

        # Anthropic requires the first message to be role="user"
        if not merged_turns:
            merged_turns.append({
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            })
        elif merged_turns[0]["role"] != "user":
            merged_turns.insert(0, {
                "role": "user",
                "content": [{"type": "text", "text": "Begin."}],
            })

        return system_instruction, merged_turns

    def _format_tools(self, tools: list[ToolSpec] | None) -> list[dict[str, Any]] | None:
        """Convert standardized ToolSpec schemas to Anthropic tool definitions."""
        if not tools:
            return None

        formatted: list[dict[str, Any]] = []
        for tool in tools:
            formatted.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            })
        return formatted

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse native Anthropic Message response into standardized LLMResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in getattr(response, "content", []):
            b_type = getattr(block, "type", None)
            if b_type == "text":
                text = getattr(block, "text", "")
                if text:
                    content_parts.append(text)
            elif b_type == "tool_use":
                tc_id = getattr(block, "id", "")
                tc_name = getattr(block, "name", "")
                tc_input = getattr(block, "input", {})
                if not isinstance(tc_input, dict):
                    tc_input = {"_raw": tc_input}
                tool_calls.append(
                    ToolCall(
                        id=tc_id,
                        name=tc_name,
                        arguments=tc_input,
                    )
                )

        combined_content = "\n\n".join(content_parts) if content_parts else None

        # Map stop_reason to standardized finish_reason
        raw_stop = getattr(response, "stop_reason", None)
        finish_reason: str | None
        if raw_stop == "end_turn":
            finish_reason = "stop"
        elif raw_stop == "tool_use":
            finish_reason = "tool_calls"
        elif raw_stop == "max_tokens":
            finish_reason = "length"
        elif raw_stop == "stop_sequence":
            finish_reason = "stop"
        else:
            finish_reason = raw_stop

        # Normalize usage
        usage_obj = getattr(response, "usage", None)
        normalized_usage: dict[str, int] | None = None
        if usage_obj is not None:
            input_tokens = getattr(usage_obj, "input_tokens", None)
            output_tokens = getattr(usage_obj, "output_tokens", None)
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                normalized_usage = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

        # Serialized raw payload for debugging
        raw_payload: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_payload = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw_payload = response.to_dict()
        else:
            raw_payload = {
                "id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
                "stop_reason": raw_stop,
            }

        return LLMResponse(
            content=combined_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=normalized_usage,
            raw=raw_payload,
        )

    def _is_retryable(self, error: Exception) -> bool:
        """Determine whether an error is transient and retryable."""
        import anthropic

        if isinstance(
            error,
            (
                anthropic.RateLimitError,
                anthropic.InternalServerError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ),
        ):
            return True

        if isinstance(error, anthropic.APIStatusError):
            return error.status_code in (429, 500, 502, 503, 504, 529)

        return False

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request to Anthropic with backoff and retry."""
        system_instruction, formatted_messages = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens or 4096,
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system_instruction:
            create_kwargs["system"] = system_instruction

        if formatted_tools:
            create_kwargs["tools"] = formatted_tools

        for k, v in kwargs.items():
            if k not in create_kwargs and k != "temperature":
                create_kwargs[k] = v

        attempt = 0
        while True:
            try:
                response = await self._client.messages.create(**create_kwargs)
                return self._parse_response(response)
            except Exception as e:
                attempt += 1
                if self._is_retryable(e) and attempt <= self.max_retries:
                    # Exponential backoff with jitter
                    base_delay = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
                    jitter = random.uniform(0, 0.5 * self.initial_delay)
                    delay = base_delay + jitter
                    await self._sleep_func(delay)
                else:
                    raise
