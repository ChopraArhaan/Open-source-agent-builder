"""Native Google Gemini LLM provider using the google-genai SDK."""

from __future__ import annotations

import asyncio
import os
import random
import uuid
from collections.abc import Callable
from typing import Any

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean JSON Schema to be strictly compatible with Gemini's types.Schema."""
    cleaned: dict[str, Any] = {}
    forbidden_keys = {"$schema", "$id", "$comment"}

    for k, v in schema.items():
        if k in forbidden_keys:
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned[k] = {prop_k: _clean_schema(prop_v) if isinstance(prop_v, dict) else prop_v for prop_k, prop_v in v.items()}
        elif k == "items" and isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        else:
            cleaned[k] = v

    return cleaned


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models using the native google-genai SDK."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        client: Any | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep_func: Callable[[float], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            from google import genai
            from google.genai import types  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "Google GenAI SDK not installed. Install with: pip install agent-builder[gemini]"
            ) from e

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )

        self._api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        if client is None and not self._api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key."
            )

        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._sleep_func = sleep_func or asyncio.sleep

        if client is not None:
            self._client = client
        else:
            self._client = genai.Client(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r})"

    def _format_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[Any]]:
        """Convert standardized LLMMessages to Gemini (system_instruction, contents)."""
        from google.genai import types

        system_parts: list[str] = []
        raw_contents: list[types.Content] = []

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == "tool":
                part = types.Part.from_function_response(
                    name=msg.name or "tool",
                    response={"output": msg.content or ""},
                )
                raw_contents.append(types.Content(role="user", parts=[part]))
            elif msg.role == "assistant":
                parts: list[Any] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        )
                if not parts:
                    parts.append(types.Part.from_text(text=""))
                raw_contents.append(types.Content(role="model", parts=parts))
            else:
                # user
                part = types.Part.from_text(text=msg.content or "")
                raw_contents.append(types.Content(role="user", parts=[part]))

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # Merge consecutive contents of identical roles
        merged_contents: list[types.Content] = []
        for content in raw_contents:
            if merged_contents and merged_contents[-1].role == content.role:
                prev_parts = merged_contents[-1].parts or []
                curr_parts = content.parts or []
                merged_contents[-1].parts = prev_parts + curr_parts
            else:
                merged_contents.append(content)

        if not merged_contents:
            merged_contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text="")])
            )

        return system_instruction, merged_contents

    def _format_tools(self, tools: list[ToolSpec] | None) -> list[Any] | None:
        """Convert standardized ToolSpec schemas to Gemini Tool definitions."""
        if not tools:
            return None

        from google.genai import types

        function_declarations: list[types.FunctionDeclaration] = []
        for tool in tools:
            clean_params = _clean_schema(tool.parameters)
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=clean_params,  # type: ignore[arg-type]  # google-genai parses dict into Schema
                )
            )

        return [types.Tool(function_declarations=function_declarations)]

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse native Gemini GenerateContentResponse into standardized LLMResponse."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = getattr(response, "candidates", []) or []
        first_candidate = candidates[0] if candidates else None

        raw_finish = None
        if first_candidate is not None:
            raw_finish = getattr(first_candidate, "finish_reason", None)
            candidate_content = getattr(first_candidate, "content", None)
            parts = getattr(candidate_content, "parts", []) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    content_parts.append(text)
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    name = getattr(fn_call, "name", "")
                    args = getattr(fn_call, "args", {})
                    if not isinstance(args, dict):
                        if hasattr(args, "items"):
                            args = dict(args.items())
                        else:
                            args = {"_raw": str(args)}
                    tc_id = getattr(fn_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=name,
                            arguments=args,
                        )
                    )

        combined_content = "\n\n".join(content_parts) if content_parts else None

        finish_reason: str | None
        if tool_calls:
            finish_reason = "tool_calls"
        elif raw_finish is not None:
            finish_str = str(raw_finish).upper()
            if "STOP" in finish_str:
                finish_reason = "stop"
            elif "MAX_TOKENS" in finish_str or "LENGTH" in finish_str:
                finish_reason = "length"
            else:
                finish_reason = str(raw_finish).lower()
        else:
            finish_reason = None

        # Normalize usage
        usage_meta = getattr(response, "usage_metadata", None)
        normalized_usage: dict[str, int] | None = None
        if usage_meta is not None:
            prompt_cnt = getattr(usage_meta, "prompt_token_count", None)
            cand_cnt = getattr(usage_meta, "candidates_token_count", None)
            total_cnt = getattr(usage_meta, "total_token_count", None)
            usage_counts: dict[str, int] = {}
            if isinstance(prompt_cnt, int):
                usage_counts["prompt_tokens"] = prompt_cnt
            if isinstance(cand_cnt, int):
                usage_counts["completion_tokens"] = cand_cnt
            if isinstance(total_cnt, int):
                usage_counts["total_tokens"] = total_cnt
            elif "prompt_tokens" in usage_counts and "completion_tokens" in usage_counts:
                usage_counts["total_tokens"] = (
                    usage_counts["prompt_tokens"] + usage_counts["completion_tokens"]
                )
            if usage_counts:
                normalized_usage = usage_counts

        # Raw payload
        raw_payload: dict[str, Any]
        if hasattr(response, "model_dump"):
            raw_payload = response.model_dump()
        elif hasattr(response, "to_dict"):
            raw_payload = response.to_dict()
        else:
            raw_payload = {
                "finish_reason": str(raw_finish) if raw_finish else None,
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
        from google.genai import errors

        code = getattr(error, "code", None)
        if isinstance(code, int):
            if code in (429, 500, 502, 503, 504):
                return True
            if 400 <= code < 500:
                return False

        if isinstance(error, errors.ServerError):
            return True

        # Check for transient network/timeout error names
        err_type = type(error).__name__
        return bool("Timeout" in err_type or "Connect" in err_type or "Network" in err_type)

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request to Gemini with backoff and retry."""
        from google.genai import types

        system_instruction, contents = self._format_messages(messages)
        formatted_tools = self._format_tools(tools)

        config_args: dict[str, Any] = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens is not None:
            config_args["max_output_tokens"] = self.max_tokens
        if system_instruction:
            config_args["system_instruction"] = system_instruction
        if formatted_tools:
            config_args["tools"] = formatted_tools

        for k, v in kwargs.items():
            if k not in config_args and k != "temperature":
                config_args[k] = v

        config = types.GenerateContentConfig(**config_args)

        attempt = 0
        while True:
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_response(response)
            except Exception as e:
                attempt += 1
                if self._is_retryable(e) and attempt <= self.max_retries:
                    base_delay = min(self.max_delay, self.initial_delay * (2 ** (attempt - 1)))
                    jitter = random.uniform(0, 0.5 * self.initial_delay)
                    delay = base_delay + jitter
                    await self._sleep_func(delay)
                else:
                    raise
