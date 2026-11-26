"""Unit and integration tests for the native Anthropic LLM provider."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.anthropic_provider import AnthropicProvider
from src.agent_builder.models import LLMMessage, ToolCall, ToolSpec
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


class MockTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, id: str, name: str, input: Any) -> None:
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class MockUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockMessageResponse:
    def __init__(
        self,
        content: list[Any],
        stop_reason: str = "end_turn",
        usage: Any = None,
        model: str = "claude-3-5-haiku-20241022",
        id: str = "msg_123",
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        self.model = model
        self.id = id

    def model_dump(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "content": [
                {"type": getattr(b, "type", "unknown")}
                for b in self.content
            ],
        }


def test_anthropic_missing_sdk_raises_runtime_error() -> None:
    """When the SDK is not installed, AnthropicProvider must raise a clear RuntimeError."""
    with (
        patch.dict(sys.modules, {"anthropic": None}),
        pytest.raises(RuntimeError, match=r"pip install agent-builder\[anthropic\]"),
    ):
        AnthropicProvider(api_key="test-key")


def test_anthropic_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing API key without injected client must raise ValueError without leaking keys."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable"):
        AnthropicProvider(api_key=None, client=None)


def test_anthropic_repr_safe_from_key_leak() -> None:
    """__repr__ must not leak the API key."""
    provider = AnthropicProvider(
        model="claude-3-7-sonnet-20250219",
        api_key="sk-ant-secret-key-12345",
        client=MagicMock(),
    )
    rep = repr(provider)
    assert "sk-ant" not in rep
    assert "claude-3-7-sonnet-20250219" in rep
    assert provider.provider_name == "anthropic"


def test_anthropic_format_messages() -> None:
    """Verify system extraction, message formatting, and consecutive turn merging."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())

    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="What is the weather in Paris and London?"),
        LLMMessage(
            role="assistant",
            content="Checking weather now.",
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"}),
                ToolCall(id="call_2", name="get_weather", arguments={"city": "London"}),
            ],
        ),
        LLMMessage(role="tool", content="20C", tool_call_id="call_1", name="get_weather"),
        LLMMessage(role="tool", content="15C", tool_call_id="call_2", name="get_weather"),
    ]

    system_instruction, formatted = provider._format_messages(messages)
    assert system_instruction == "You are a helpful assistant."

    # formatted must alternate: user -> assistant -> user (merged tool results)
    assert len(formatted) == 3
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == [{"type": "text", "text": "What is the weather in Paris and London?"}]

    assert formatted[1]["role"] == "assistant"
    assert len(formatted[1]["content"]) == 3  # 1 text block + 2 tool_use blocks
    assert formatted[1]["content"][0]["type"] == "text"
    assert formatted[1]["content"][1]["type"] == "tool_use"
    assert formatted[1]["content"][1]["name"] == "get_weather"

    assert formatted[2]["role"] == "user"
    assert len(formatted[2]["content"]) == 2  # 2 merged tool_result blocks
    assert formatted[2]["content"][0]["type"] == "tool_result"
    assert formatted[2]["content"][0]["tool_use_id"] == "call_1"
    assert formatted[2]["content"][0]["content"] == "20C"
    assert formatted[2]["content"][1]["tool_use_id"] == "call_2"
    assert formatted[2]["content"][1]["content"] == "15C"


def test_anthropic_format_messages_empty_or_assistant_first() -> None:
    """Anthropic requires the first message to be role='user'."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())

    # Empty messages
    sys_inst, formatted = provider._format_messages([])
    assert sys_inst is None
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"

    # Assistant-first message
    sys_inst, formatted = provider._format_messages([
        LLMMessage(role="assistant", content="Hello!")
    ])
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"


def test_anthropic_format_tools() -> None:
    """Tool specs should map cleanly to Anthropic's input_schema structure."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())

    assert provider._format_tools(None) is None
    assert provider._format_tools([]) is None

    spec = ToolSpec(
        name="calc",
        description="Run calculation",
        parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
    )
    formatted = provider._format_tools([spec])
    assert formatted is not None
    assert len(formatted) == 1
    assert formatted[0]["name"] == "calc"
    assert formatted[0]["description"] == "Run calculation"
    assert formatted[0]["input_schema"] == spec.parameters


def test_anthropic_complete_success_text() -> None:
    """Simple text response completion."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockMessageResponse(
            content=[MockTextBlock("The capital of France is Paris.")],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=10, output_tokens=8),
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", client=mock_client)
        res = await provider.complete([LLMMessage(role="user", content="Capital of France?")])

        assert res.content == "The capital of France is Paris."
        assert res.tool_calls == []
        assert res.finish_reason == "stop"
        assert res.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        }
        assert res.raw is not None

    asyncio.run(_run())


def test_anthropic_complete_tool_calls() -> None:
    """Tool calling response completion."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockMessageResponse(
            content=[
                MockTextBlock("I will check the weather."),
                MockToolUseBlock("toolu_abc", "weather", {"city": "Berlin"}),
            ],
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=12, output_tokens=15),
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", client=mock_client)
        res = await provider.complete([LLMMessage(role="user", content="Weather Berlin?")])

        assert res.content == "I will check the weather."
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].id == "toolu_abc"
        assert res.tool_calls[0].name == "weather"
        assert res.tool_calls[0].arguments == {"city": "Berlin"}
        assert res.finish_reason == "tool_calls"

    asyncio.run(_run())


def test_anthropic_complete_non_dict_tool_input() -> None:
    """Tolerate non-dict tool inputs gracefully by wrapping in _raw."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockMessageResponse(
            content=[
                MockToolUseBlock("toolu_xyz", "parse", "not-a-dict"),
            ],
            stop_reason="tool_use",
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", client=mock_client)
        res = await provider.complete([LLMMessage(role="user", content="Parse this")])
        assert res.tool_calls[0].arguments == {"_raw": "not-a-dict"}

    asyncio.run(_run())


def test_anthropic_retry_transient_error_success() -> None:
    """Transient errors (RateLimitError, etc.) should be retried with exponential backoff."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockMessageResponse(
            content=[MockTextBlock("Success after retry")],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=5, output_tokens=5),
        )

        rate_limit_err = anthropic.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )

        mock_client.messages.create = AsyncMock(
            side_effect=[rate_limit_err, rate_limit_err, mock_response]
        )

        sleep_mock = AsyncMock()
        provider = AnthropicProvider(
            api_key="test-key",
            client=mock_client,
            max_retries=3,
            initial_delay=0.1,
            sleep_func=sleep_mock,
        )

        res = await provider.complete([LLMMessage(role="user", content="Ping")])
        assert res.content == "Success after retry"
        assert mock_client.messages.create.call_count == 3
        assert sleep_mock.call_count == 2

    asyncio.run(_run())


def test_anthropic_retry_exhausted_raises() -> None:
    """Exhausting max_retries on transient errors should re-raise the exception."""
    async def _run() -> None:
        mock_client = MagicMock()
        server_err = anthropic.InternalServerError(
            message="Internal server error",
            response=MagicMock(status_code=500, headers={}),
            body=None,
        )
        mock_client.messages.create = AsyncMock(side_effect=server_err)

        sleep_mock = AsyncMock()
        provider = AnthropicProvider(
            api_key="test-key",
            client=mock_client,
            max_retries=2,
            initial_delay=0.1,
            sleep_func=sleep_mock,
        )

        with pytest.raises(anthropic.InternalServerError):
            await provider.complete([LLMMessage(role="user", content="Ping")])

        assert mock_client.messages.create.call_count == 3  # initial + 2 retries
        assert sleep_mock.call_count == 2

    asyncio.run(_run())


def test_anthropic_non_retryable_error_fails_immediately() -> None:
    """Authentication or client errors (401, 400) must fail immediately without retry."""
    async def _run() -> None:
        mock_client = MagicMock()
        auth_err = anthropic.AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        mock_client.messages.create = AsyncMock(side_effect=auth_err)

        sleep_mock = AsyncMock()
        provider = AnthropicProvider(
            api_key="test-key",
            client=mock_client,
            max_retries=3,
            sleep_func=sleep_mock,
        )

        with pytest.raises(anthropic.AuthenticationError):
            await provider.complete([LLMMessage(role="user", content="Ping")])

        assert mock_client.messages.create.call_count == 1
        assert sleep_mock.call_count == 0

    asyncio.run(_run())


def test_anthropic_react_engine_integration() -> None:
    """Verify AnthropicProvider works end-to-end within ReActEngine."""
    async def _run() -> None:
        mock_client = MagicMock()

        turn1 = MockMessageResponse(
            content=[
                MockTextBlock("Thinking: calculating 25 * 4"),
                MockToolUseBlock("toolu_1", "calculator", {"expression": "25 * 4"}),
            ],
            stop_reason="tool_use",
            usage=MockUsage(input_tokens=20, output_tokens=15),
        )
        turn2 = MockMessageResponse(
            content=[
                MockTextBlock("The result of 25 * 4 is 100."),
            ],
            stop_reason="end_turn",
            usage=MockUsage(input_tokens=35, output_tokens=10),
        )

        mock_client.messages.create = AsyncMock(side_effect=[turn1, turn2])

        provider = AnthropicProvider(api_key="test-key", client=mock_client)
        registry = ToolRegistry()
        register_builtin_tools(registry)

        engine = ReActEngine(
            llm=provider,
            tools=registry,
            max_iterations=5,
        )

        final_answer, events = await engine.run("Calculate 25 * 4")
        assert final_answer == "The result of 25 * 4 is 100."
        assert any(e.type == "final" for e in events)
        assert mock_client.messages.create.call_count == 2

    asyncio.run(_run())


def test_anthropic_default_client_init() -> None:
    """Verify AnthropicProvider initializes real AsyncAnthropic when client is None."""
    with patch("anthropic.AsyncAnthropic") as mock_cls:
        provider = AnthropicProvider(api_key="test-key", client=None)
        assert mock_cls.called
        assert provider.model == "claude-3-5-haiku-20241022"


def test_anthropic_format_empty_assistant_message() -> None:
    """Assistant message with no content and no tool calls produces empty text block."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())
    _, formatted = provider._format_messages([
        LLMMessage(role="assistant", content=None, tool_calls=None),
    ])
    assert len(formatted) == 2  # prepended user + assistant
    assert formatted[1]["content"] == [{"type": "text", "text": ""}]


def test_anthropic_stop_reasons_and_serialization() -> None:
    """Test max_tokens, stop_sequence, unknown stop reasons, and raw serialization fallbacks."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())

    # max_tokens -> length
    r1 = provider._parse_response(
        SimpleNamespace(
            content=[],
            stop_reason="max_tokens",
            usage=None,
            model_dump=lambda: {"k": "v"},
        )
    )
    assert r1.finish_reason == "length"
    assert r1.raw == {"k": "v"}

    # stop_sequence -> stop, with to_dict
    r2 = provider._parse_response(
        SimpleNamespace(
            content=[],
            stop_reason="stop_sequence",
            usage=None,
            to_dict=lambda: {"to_dict": True},
        )
    )
    assert r2.finish_reason == "stop"
    assert r2.raw == {"to_dict": True}

    # unknown stop reason and fallback dict
    r3 = provider._parse_response(
        SimpleNamespace(
            content=[],
            stop_reason="custom_stop",
            usage=None,
            id="msg_999",
            model="claude-3",
        )
    )
    assert r3.finish_reason == "custom_stop"
    assert r3.raw == {"id": "msg_999", "model": "claude-3", "stop_reason": "custom_stop"}


def test_anthropic_complete_system_tools_and_kwargs() -> None:
    """Verify complete() passes system prompt, tools, and extra kwargs correctly."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockMessageResponse(
            content=[MockTextBlock("OK")],
            stop_reason="end_turn",
        )
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key", client=mock_client)
        spec = ToolSpec(
            name="test_tool",
            description="desc",
            parameters={"type": "object"},
        )

        res = await provider.complete(
            messages=[
                LLMMessage(role="system", content="System instruction"),
                LLMMessage(role="user", content="Hello"),
            ],
            tools=[spec],
            temperature=0.5,
            top_p=0.9,
        )

        assert res.content == "OK"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "System instruction"
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    asyncio.run(_run())


def test_anthropic_status_error_not_retryable() -> None:
    """APIStatusError with 404 should not be retryable."""
    provider = AnthropicProvider(api_key="test-key", client=MagicMock())
    not_found_err = anthropic.APIStatusError(
        message="Not found",
        response=MagicMock(status_code=404, headers={}),
        body=None,
    )
    assert not provider._is_retryable(not_found_err)
    generic_err = RuntimeError("Something bad")
    assert not provider._is_retryable(generic_err)

