"""Shared contract test suite parametrized across all LLM providers.

Every provider (fake, openai, anthropic, gemini) must pass the identical contract suite.
Zero real network calls, zero real API keys.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.anthropic_provider import AnthropicProvider
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.llm.gemini_provider import GeminiProvider
from src.agent_builder.llm.openai import OpenAIProvider
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry

# -----------------------------------------------------------------------------
# Provider Fixture Factories
# -----------------------------------------------------------------------------

def _build_fake_text_provider() -> BaseLLMProvider:
    return FakeLLMProvider(
        model="fake-model",
        responses=[
            LLMResponse(
                content="Hello world response",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ],
    )


def _build_fake_tool_provider() -> BaseLLMProvider:
    return FakeLLMProvider(
        model="fake-model",
        responses=[
            [ToolCall(id="call_fake_1", name="calculator", arguments={"expression": "25 * 4"})]
        ],
    )


def _build_fake_react_provider() -> BaseLLMProvider:
    return FakeLLMProvider(
        model="fake-model",
        responses=[
            [ToolCall(id="call_fake_1", name="calculator", arguments={"expression": "25 * 4"})],
            "The result is 100.",
        ],
    )


def _build_openai_text_provider() -> BaseLLMProvider:
    payload = {
        "id": "chatcmpl-text-1",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello world response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    )
    return OpenAIProvider(
        model="gpt-4o-mini",
        api_key="sk-openai-secret-key-12345",
        http_client=client,
    )


def _build_openai_tool_provider() -> BaseLLMProvider:
    payload = {
        "id": "chatcmpl-tool-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_openai_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "25 * 4"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
    )
    return OpenAIProvider(
        model="gpt-4o-mini",
        api_key="sk-openai-secret-key-12345",
        http_client=client,
    )


def _build_openai_react_provider() -> BaseLLMProvider:
    p1 = {
        "id": "chatcmpl-turn-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Thinking about calculation",
                    "tool_calls": [
                        {
                            "id": "call_openai_react_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "25 * 4"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    p2 = {
        "id": "chatcmpl-turn-2",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The result is 100.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 35, "completion_tokens": 8, "total_tokens": 43},
    }

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        return httpx.Response(200, json=p1 if turn == 1 else p2)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIProvider(
        model="gpt-4o-mini",
        api_key="sk-openai-secret-key-12345",
        http_client=client,
    )


def _build_anthropic_text_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        id="msg_text_1",
        model="claude-3-5-haiku-20241022",
        content=[SimpleNamespace(type="text", text="Hello world response")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=14, output_tokens=7),
    )
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return AnthropicProvider(
        model="claude-3-5-haiku-20241022",
        api_key="sk-ant-anthropic-secret-key-54321",
        client=mock_client,
    )


def _build_anthropic_tool_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        id="msg_tool_1",
        model="claude-3-5-haiku-20241022",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="call_anthropic_1",
                name="calculator",
                input={"expression": "25 * 4"},
            )
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=18, output_tokens=9),
    )
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    return AnthropicProvider(
        model="claude-3-5-haiku-20241022",
        api_key="sk-ant-anthropic-secret-key-54321",
        client=mock_client,
    )


def _build_anthropic_react_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    t1 = SimpleNamespace(
        id="msg_react_1",
        model="claude-3-5-haiku-20241022",
        content=[
            SimpleNamespace(type="text", text="Thinking about calculation"),
            SimpleNamespace(
                type="tool_use",
                id="call_anthropic_react_1",
                name="calculator",
                input={"expression": "25 * 4"},
            ),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=22, output_tokens=11),
    )
    t2 = SimpleNamespace(
        id="msg_react_2",
        model="claude-3-5-haiku-20241022",
        content=[SimpleNamespace(type="text", text="The result is 100.")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=38, output_tokens=9),
    )
    mock_client.messages.create = AsyncMock(side_effect=[t1, t2])
    return AnthropicProvider(
        model="claude-3-5-haiku-20241022",
        api_key="sk-ant-anthropic-secret-key-54321",
        client=mock_client,
    )


def _build_gemini_text_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text="Hello world response", function_call=None)]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=13,
            candidates_token_count=6,
            total_token_count=19,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return GeminiProvider(
        model="gemini-2.0-flash",
        api_key="AIzaSyGeminiSecretKey67890",
        client=mock_client,
    )


def _build_gemini_tool_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text=None,
                            function_call=SimpleNamespace(
                                id="call_gemini_1",
                                name="calculator",
                                args={"expression": "25 * 4"},
                            ),
                        )
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=16,
            candidates_token_count=7,
            total_token_count=23,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return GeminiProvider(
        model="gemini-2.0-flash",
        api_key="AIzaSyGeminiSecretKey67890",
        client=mock_client,
    )


def _build_gemini_react_provider() -> BaseLLMProvider:
    mock_client = MagicMock()
    t1 = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="Thinking about calculation",
                            function_call=SimpleNamespace(
                                id="call_gemini_react_1",
                                name="calculator",
                                args={"expression": "25 * 4"},
                            ),
                        )
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=21,
            candidates_token_count=10,
            total_token_count=31,
        ),
    )
    t2 = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            text="The result is 100.",
                            function_call=None,
                        )
                    ]
                ),
                finish_reason="STOP",
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=36,
            candidates_token_count=8,
            total_token_count=44,
        ),
    )
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[t1, t2])
    return GeminiProvider(
        model="gemini-2.0-flash",
        api_key="AIzaSyGeminiSecretKey67890",
        client=mock_client,
    )


PROVIDER_BUILDERS = [
    ("fake", _build_fake_text_provider, _build_fake_tool_provider, _build_fake_react_provider),
    ("openai", _build_openai_text_provider, _build_openai_tool_provider, _build_openai_react_provider),
    ("anthropic", _build_anthropic_text_provider, _build_anthropic_tool_provider, _build_anthropic_react_provider),
    ("gemini", _build_gemini_text_provider, _build_gemini_tool_provider, _build_gemini_react_provider),
]


# -----------------------------------------------------------------------------
# Shared Contract Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("name,build_text,build_tool,build_react", PROVIDER_BUILDERS)
def test_contract_provider_metadata_and_secret_safety(
    name: str,
    build_text: Any,
    build_tool: Any,
    build_react: Any,
) -> None:
    """Verify provider metadata, inheritance, and secret safety in __repr__."""
    provider = build_text()
    assert isinstance(provider, BaseLLMProvider)
    assert provider.provider_name == name
    assert isinstance(provider.model, str) and len(provider.model) > 0

    # Ensure secret keys are never leaked in __repr__
    rep = repr(provider)
    assert "secret" not in rep.lower()
    assert "sk-" not in rep
    assert "AIzaSy" not in rep


@pytest.mark.parametrize("name,build_text,build_tool,build_react", PROVIDER_BUILDERS)
def test_contract_text_completion(
    name: str,
    build_text: Any,
    build_tool: Any,
    build_react: Any,
) -> None:
    """Verify text completion returns a valid standardized LLMResponse."""
    async def _run() -> None:
        provider = build_text()
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello world"),
        ]
        resp = await provider.complete(messages)

        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.content, str) and len(resp.content) > 0
        assert resp.content == "Hello world response"
        assert resp.tool_calls == []
        assert resp.finish_reason in ("stop", None)

        if resp.usage is not None:
            assert isinstance(resp.usage, dict)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in resp.usage:
                    val = resp.usage[key]
                    assert isinstance(val, int) and not isinstance(val, bool)
                    assert val >= 0

        assert resp.raw is None or isinstance(resp.raw, dict)

    asyncio.run(_run())


@pytest.mark.parametrize("name,build_text,build_tool,build_react", PROVIDER_BUILDERS)
def test_contract_tool_calling(
    name: str,
    build_text: Any,
    build_tool: Any,
    build_react: Any,
) -> None:
    """Verify tool calling returns valid ToolCall instances and finish_reason='tool_calls'."""
    async def _run() -> None:
        provider = build_tool()
        tool_spec = ToolSpec(
            name="calculator",
            description="Run math calculation",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        )
        messages = [LLMMessage(role="user", content="Calculate 25 * 4")]
        resp = await provider.complete(messages, tools=[tool_spec])

        assert isinstance(resp, LLMResponse)
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert isinstance(tc.id, str) and len(tc.id) > 0
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "25 * 4"}
        assert resp.finish_reason == "tool_calls"

    asyncio.run(_run())


@pytest.mark.parametrize("name,build_text,build_tool,build_react", PROVIDER_BUILDERS)
def test_contract_react_engine_execution(
    name: str,
    build_text: Any,
    build_tool: Any,
    build_react: Any,
) -> None:
    """Verify every provider executes end-to-end within ReActEngine."""
    async def _run() -> None:
        provider = build_react()
        registry = ToolRegistry()
        register_builtin_tools(registry)

        engine = ReActEngine(
            llm=provider,
            tools=registry,
            max_iterations=5,
        )

        final_answer, events = await engine.run("Calculate 25 * 4")
        assert final_answer == "The result is 100."
        assert any(e.type == "final" for e in events)

    asyncio.run(_run())
