"""OpenAI-compatible LLM provider using pure httpx."""

import json
import os
from typing import Any

import httpx

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs (Ollama, vLLM, Groq, Together)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout_seconds: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        effective_base = base_url if base_url != "https://api.openai.com/v1" else (os.getenv("OPENAI_BASE_URL") or base_url)
        self.base_url = effective_base.rstrip("/")
        self._http_client = http_client

    @property
    def provider_name(self) -> str:
        return "openai"

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Serialize standardized LLMMessage models to OpenAI wire format."""
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            entry: dict[str, Any] = {"role": msg.role}

            if msg.role == "tool":
                entry["content"] = msg.content or ""
                entry["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    entry["name"] = msg.name
            elif msg.role == "assistant":
                entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
            else:
                # user or system
                entry["content"] = msg.content or ""

            formatted.append(entry)

        return formatted

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse raw OpenAI completion payload into standardized LLMResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI response contains no choices")

        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage")

        tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls") or []

        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=data,
        )

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute chat completion with optional native tool schemas."""
        endpoint = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
        }

        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        if tools:
            payload["tools"] = [tool.to_openai_schema() for tool in tools]
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Merge any caller overrides
        for k, v in kwargs.items():
            if k not in payload and k != "tool_choice":
                payload[k] = v

        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Use injected client if provided, otherwise create temporary client
        if self._http_client:
            response = await self._http_client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )

        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI API error [{response.status_code}]: {response.text}"
            )

        data = response.json()
        return self._parse_response(data)
