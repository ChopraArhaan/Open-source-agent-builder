"""Unit and integration tests for the native Google Gemini LLM provider."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import errors

from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.gemini_provider import GeminiProvider, _clean_schema
from src.agent_builder.models import LLMMessage, ToolCall, ToolSpec
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


class MockCandidate:
    def __init__(self, parts: list[Any], finish_reason: str = "STOP") -> None:
        self.content = SimpleNamespace(parts=parts)
        self.finish_reason = finish_reason


class MockGeminiResponse:
    def __init__(
        self,
        candidates: list[Any] | None = None,
        usage_metadata: Any = None,
    ) -> None:
        self.candidates = candidates or []
        self.usage_metadata = usage_metadata

    def model_dump(self) -> dict[str, Any]:
        return {"candidates": len(self.candidates)}


def test_gemini_missing_sdk_raises_runtime_error() -> None:
    """When the SDK is not installed, GeminiProvider must raise a clear RuntimeError."""
    with (
        patch.dict(sys.modules, {"google.genai": None}),
        pytest.raises(RuntimeError, match=r"pip install agent-builder\[gemini\]"),
    ):
        GeminiProvider(api_key="test-key")


def test_gemini_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing API key without injected client must raise ValueError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable"):
        GeminiProvider(api_key=None, client=None)


def test_gemini_repr_safe_from_key_leak() -> None:
    """__repr__ must not leak the API key."""
    provider = GeminiProvider(
        model="gemini-2.0-flash",
        api_key="AIzaSySecretKey98765",
        client=MagicMock(),
    )
    rep = repr(provider)
    assert "AIzaSy" not in rep
    assert "gemini-2.0-flash" in rep
    assert provider.provider_name == "gemini"


def test_clean_schema_removes_unsupported_keys() -> None:
    """_clean_schema must recursively strip $schema, $id, $comment."""
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://example.com/schema.json",
        "$comment": "metadata",
        "type": "object",
        "properties": {
            "city": {
                "$comment": "the city name",
                "type": "string",
            },
            "tags": {
                "type": "array",
                "items": {
                    "$comment": "tag item",
                    "type": "string",
                },
            },
        },
        "required": ["city"],
    }
    cleaned = _clean_schema(schema)
    assert "$schema" not in cleaned
    assert "$id" not in cleaned
    assert "$comment" not in cleaned
    assert "$comment" not in cleaned["properties"]["city"]
    assert "$comment" not in cleaned["properties"]["tags"]["items"]
    assert cleaned["properties"]["city"]["type"] == "string"


def test_gemini_format_messages() -> None:
    """Verify system extraction, message formatting, and role alternation."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())

    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="Calculate 2 + 2"),
        LLMMessage(
            role="assistant",
            content="Thinking",
            tool_calls=[ToolCall(id="call_1", name="calc", arguments={"expr": "2+2"})],
        ),
        LLMMessage(role="tool", content="4", name="calc", tool_call_id="call_1"),
        LLMMessage(role="tool", content="extra", name="calc", tool_call_id="call_2"),
    ]

    sys_inst, contents = provider._format_messages(messages)
    assert sys_inst == "System instruction"
    assert len(contents) == 3

    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Calculate 2 + 2"

    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Thinking"
    assert contents[1].parts[1].function_call.name == "calc"
    assert contents[1].parts[1].function_call.args == {"expr": "2+2"}

    assert contents[2].role == "user"
    assert len(contents[2].parts) == 2  # merged 2 tool responses
    assert contents[2].parts[0].function_response.name == "calc"
    assert contents[2].parts[0].function_response.response == {"output": "4"}
    assert contents[2].parts[1].function_response.response == {"output": "extra"}


def test_gemini_format_messages_empty_and_assistant_no_content() -> None:
    """Test empty messages list and assistant message with no content or tool calls."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())

    _, contents = provider._format_messages([])
    assert len(contents) == 1
    assert contents[0].role == "user"

    _, contents2 = provider._format_messages([
        LLMMessage(role="assistant", content=None, tool_calls=None)
    ])
    assert contents2[0].role == "model"
    assert contents2[0].parts[0].text == ""


def test_gemini_format_tools() -> None:
    """Verify ToolSpec converts to types.Tool containing FunctionDeclaration."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())

    assert provider._format_tools(None) is None
    assert provider._format_tools([]) is None

    spec = ToolSpec(
        name="calc",
        description="Run calculation",
        parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
    )
    tools = provider._format_tools([spec])
    assert tools is not None
    assert len(tools) == 1
    assert len(tools[0].function_declarations) == 1
    fn = tools[0].function_declarations[0]
    assert fn.name == "calc"
    assert fn.description == "Run calculation"


def test_gemini_complete_text_success() -> None:
    """Verify successful text completion and usage normalization."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockGeminiResponse(
            candidates=[
                MockCandidate(
                    parts=[SimpleNamespace(text="Hello world!", function_call=None)],
                    finish_reason="STOP",
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=3,
                total_token_count=11,
            ),
        )
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiProvider(
            api_key="test-key",
            client=mock_client,
            max_tokens=500,
        )
        spec = ToolSpec(
            name="dummy",
            description="desc",
            parameters={"type": "object"},
        )
        res = await provider.complete(
            [
                LLMMessage(role="system", content="System prompt"),
                LLMMessage(role="user", content="Hi"),
            ],
            tools=[spec],
            temperature=0.8,
            top_p=0.95,
        )

        assert res.content == "Hello world!"
        assert res.tool_calls == []
        assert res.finish_reason == "stop"
        assert res.usage == {
            "prompt_tokens": 8,
            "completion_tokens": 3,
            "total_tokens": 11,
        }
        assert res.raw is not None

        call_config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert call_config.temperature == 0.8
        assert call_config.max_output_tokens == 500
        assert call_config.system_instruction == "System prompt"

    asyncio.run(_run())


def test_gemini_complete_tool_calls() -> None:
    """Verify tool call parsing and finish_reason."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockGeminiResponse(
            candidates=[
                MockCandidate(
                    parts=[
                        SimpleNamespace(
                            text="Calling tool",
                            function_call=SimpleNamespace(
                                name="calculator",
                                args={"expression": "10+5"},
                                id="call_custom_1",
                            ),
                        )
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GeminiProvider(api_key="test-key", client=mock_client)
        res = await provider.complete([LLMMessage(role="user", content="10+5")])

        assert res.content == "Calling tool"
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0].name == "calculator"
        assert res.tool_calls[0].arguments == {"expression": "10+5"}
        assert res.tool_calls[0].id == "call_custom_1"
        assert res.finish_reason == "tool_calls"

    asyncio.run(_run())


def test_gemini_parse_response_edge_cases() -> None:
    """Test non-dict args, missing total_tokens, length finish reason, and raw fallbacks."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())

    # Mapping args with items()
    class MappingArgs:
        def items(self) -> list[tuple[str, Any]]:
            return [("a", 1)]

    # Non-mapping string args
    resp1 = MockGeminiResponse(
        candidates=[
            MockCandidate(
                parts=[
                    SimpleNamespace(
                        text=None,
                        function_call=SimpleNamespace(
                            name="f1",
                            args=MappingArgs(),
                            id=None,
                        ),
                    ),
                    SimpleNamespace(
                        text=None,
                        function_call=SimpleNamespace(
                            name="f2",
                            args="not_dict",
                            id=None,
                        ),
                    ),
                ],
                finish_reason="MAX_TOKENS",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=None,
        ),
    )
    r1 = provider._parse_response(resp1)
    assert len(r1.tool_calls) == 2
    assert r1.tool_calls[0].arguments == {"a": 1}
    assert r1.tool_calls[1].arguments == {"_raw": "not_dict"}
    assert r1.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    # Empty candidates and to_dict raw fallback
    resp2 = SimpleNamespace(
        candidates=[],
        usage_metadata=None,
        to_dict=lambda: {"raw_key": "val"},
    )
    r2 = provider._parse_response(resp2)
    assert r2.content is None
    assert r2.finish_reason is None
    assert r2.raw == {"raw_key": "val"}

    # Fallback raw payload when no model_dump/to_dict
    resp3 = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="SAFETY", content=None)],
        usage_metadata=None,
    )
    r3 = provider._parse_response(resp3)
    assert r3.finish_reason == "safety"
    assert r3.raw == {"finish_reason": "SAFETY"}


def test_gemini_retry_transient_error_success() -> None:
    """Transient errors (429, 503) should be retried with exponential backoff."""
    async def _run() -> None:
        mock_client = MagicMock()
        mock_response = MockGeminiResponse(
            candidates=[
                MockCandidate(
                    parts=[SimpleNamespace(text="Done after retry", function_call=None)],
                    finish_reason="STOP",
                )
            ]
        )

        rate_limit_err = errors.APIError(429, "Resource exhausted")
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[rate_limit_err, rate_limit_err, mock_response]
        )

        sleep_mock = AsyncMock()
        provider = GeminiProvider(
            api_key="test-key",
            client=mock_client,
            max_retries=3,
            initial_delay=0.1,
            sleep_func=sleep_mock,
        )

        res = await provider.complete([LLMMessage(role="user", content="Ping")])
        assert res.content == "Done after retry"
        assert mock_client.aio.models.generate_content.call_count == 3
        assert sleep_mock.call_count == 2

    asyncio.run(_run())


def test_gemini_non_retryable_error_fails_immediately() -> None:
    """Non-retryable 401 error must fail immediately without retries."""
    async def _run() -> None:
        mock_client = MagicMock()
        auth_err = errors.APIError(401, "Invalid API key")
        mock_client.aio.models.generate_content = AsyncMock(side_effect=auth_err)

        sleep_mock = AsyncMock()
        provider = GeminiProvider(
            api_key="test-key",
            client=mock_client,
            max_retries=3,
            sleep_func=sleep_mock,
        )

        with pytest.raises(errors.APIError) as exc_info:
            await provider.complete([LLMMessage(role="user", content="Ping")])

        assert exc_info.value.code == 401
        assert mock_client.aio.models.generate_content.call_count == 1
        assert sleep_mock.call_count == 0

    asyncio.run(_run())


def test_gemini_is_retryable_conditions() -> None:
    """Verify retryable check against server errors and network errors."""
    provider = GeminiProvider(api_key="test-key", client=MagicMock())

    # 503 Server error
    err503 = errors.APIError(503, "Unavailable")
    assert provider._is_retryable(err503)

    # ServerError
    assert provider._is_retryable(errors.ServerError(500, "Crash"))

    # Generic network timeout
    class NetworkTimeoutError(Exception):
        pass

    assert provider._is_retryable(NetworkTimeoutError("Timeout"))

    # Non-retryable 400
    err400 = errors.APIError(400, "Bad Request")
    assert not provider._is_retryable(err400)

    # Generic runtime error
    assert not provider._is_retryable(RuntimeError("Unknown"))


def test_gemini_default_client_init() -> None:
    """Verify GeminiProvider initializes real genai.Client when client is None."""
    with patch("google.genai.Client") as mock_cls:
        provider = GeminiProvider(api_key="test-key", client=None)
        assert mock_cls.called
        assert provider.model == "gemini-2.0-flash"


def test_gemini_react_engine_integration() -> None:
    """Verify GeminiProvider works end-to-end within ReActEngine."""
    async def _run() -> None:
        mock_client = MagicMock()

        turn1 = MockGeminiResponse(
            candidates=[
                MockCandidate(
                    parts=[
                        SimpleNamespace(
                            text="Calculating 15 * 3",
                            function_call=SimpleNamespace(
                                name="calculator",
                                args={"expression": "15 * 3"},
                                id="call_react_1",
                            ),
                        )
                    ],
                    finish_reason="STOP",
                )
            ]
        )
        turn2 = MockGeminiResponse(
            candidates=[
                MockCandidate(
                    parts=[
                        SimpleNamespace(
                            text="The result is 45.",
                            function_call=None,
                        )
                    ],
                    finish_reason="STOP",
                )
            ]
        )

        mock_client.aio.models.generate_content = AsyncMock(side_effect=[turn1, turn2])

        provider = GeminiProvider(api_key="test-key", client=mock_client)
        registry = ToolRegistry()
        register_builtin_tools(registry)

        engine = ReActEngine(
            llm=provider,
            tools=registry,
            max_iterations=5,
        )

        final_answer, events = await engine.run("Calculate 15 * 3")
        assert final_answer == "The result is 45."
        assert any(e.type == "final" for e in events)
        assert mock_client.aio.models.generate_content.call_count == 2

    asyncio.run(_run())
