"""Test suite for LLM provider abstraction, FakeLLMProvider, and OpenAIProvider."""

import asyncio
import json

import httpx
import pytest

from src.agent_builder.llm import (
    FakeLLMProvider,
    OpenAIProvider,
    create_llm_provider,
)
from src.agent_builder.models import (
    LLMMessage,
    LLMResponse,
    ToolCall,
    ToolSpec,
)


def test_fake_llm_provider_default():
    """Verify FakeLLMProvider fallback response and history tracking."""
    provider = FakeLLMProvider(default_response="System ready.")
    messages = [LLMMessage(role="user", content="Hello")]

    async def _run():
        resp = await provider.complete(messages)
        assert resp.content == "System ready."
        assert resp.finish_reason == "stop"
        assert resp.usage is not None
        assert provider.call_count == 1
        last = provider.last_call()
        assert last is not None
        assert last["messages"] == messages

    asyncio.run(_run())


def test_fake_llm_provider_scripted_sequence():
    """Verify FakeLLMProvider yields queued responses in FIFO order."""
    tool_call = ToolCall(name="calculator", arguments={"expression": "10 * 10"})
    provider = FakeLLMProvider(
        responses=[
            "First response",
            [tool_call],
            LLMResponse(content="Final scripted response", finish_reason="stop"),
        ]
    )

    async def _run():
        # Step 1: Text response
        r1 = await provider.complete([LLMMessage(role="user", content="Step 1")])
        assert r1.content == "First response"
        assert len(r1.tool_calls) == 0

        # Step 2: Tool call response
        r2 = await provider.complete([LLMMessage(role="user", content="Step 2")])
        assert r2.content is None
        assert len(r2.tool_calls) == 1
        assert r2.tool_calls[0].name == "calculator"
        assert r2.tool_calls[0].arguments == {"expression": "10 * 10"}

        # Step 3: Explicit LLMResponse
        r3 = await provider.complete([LLMMessage(role="user", content="Step 3")])
        assert r3.content == "Final scripted response"
        assert r3.finish_reason == "stop"

        # Step 4: Queue exhausted -> default response
        r4 = await provider.complete([LLMMessage(role="user", content="Step 4")])
        assert r4.content == provider.default_response

    asyncio.run(_run())


def test_fake_llm_provider_exception():
    """Verify FakeLLMProvider raises queued exceptions."""
    provider = FakeLLMProvider(
        responses=[RuntimeError("Simulated LLM rate limit exceeded")]
    )

    async def _run():
        with pytest.raises(RuntimeError, match="Simulated LLM rate limit exceeded"):
            await provider.complete([LLMMessage(role="user", content="Hi")])

    asyncio.run(_run())


def test_openai_message_serialization():
    """Verify LLMMessage objects are correctly formatted into OpenAI API schema."""
    provider = OpenAIProvider(api_key="test-key")
    messages = [
        LLMMessage(role="system", content="You are a helpful assistant."),
        LLMMessage(role="user", content="Compute 2+2"),
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_abc", name="calculator", arguments={"expression": "2+2"})
            ],
        ),
        LLMMessage(
            role="tool",
            content="4",
            tool_call_id="call_abc",
            name="calculator",
        ),
    ]

    formatted = provider._format_messages(messages)
    assert len(formatted) == 4
    assert formatted[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert formatted[1] == {"role": "user", "content": "Compute 2+2"}

    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["tool_calls"][0]["id"] == "call_abc"
    assert formatted[2]["tool_calls"][0]["type"] == "function"
    assert formatted[2]["tool_calls"][0]["function"]["name"] == "calculator"
    assert json.loads(formatted[2]["tool_calls"][0]["function"]["arguments"]) == {"expression": "2+2"}

    assert formatted[3]["role"] == "tool"
    assert formatted[3]["content"] == "4"
    assert formatted[3]["tool_call_id"] == "call_abc"
    assert formatted[3]["name"] == "calculator"


def test_openai_complete_mock_text():
    """Verify OpenAIProvider parses text responses using mock httpx transport."""
    mock_payload = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello there! How can I help you?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer mock-api-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0.5
        return httpx.Response(200, json=mock_payload)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        model="gpt-4o-mini",
        api_key="mock-api-key",
        temperature=0.5,
        http_client=mock_client,
    )

    async def _run():
        resp = await provider.complete([LLMMessage(role="user", content="Hi")])
        assert resp.content == "Hello there! How can I help you?"
        assert resp.finish_reason == "stop"
        assert resp.usage == {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
        assert resp.tool_calls == []
        assert resp.raw == mock_payload

    asyncio.run(_run())


def test_openai_complete_mock_tool_calls():
    """Verify OpenAIProvider parses tool call responses using mock httpx transport."""
    mock_payload = {
        "id": "chatcmpl-456",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz123",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "50 * 2"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "calculator"
        return httpx.Response(200, json=mock_payload)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="mock-key", http_client=mock_client)

    tools = [
        ToolSpec(
            name="calculator",
            description="Math tool",
            parameters={"type": "object", "properties": {"expression": {"type": "string"}}},
        )
    ]

    async def _run():
        resp = await provider.complete(
            messages=[LLMMessage(role="user", content="Compute 50*2")],
            tools=tools,
        )
        assert resp.content is None
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.id == "call_xyz123"
        assert tc.name == "calculator"
        assert tc.arguments == {"expression": "50 * 2"}

    asyncio.run(_run())


def test_openai_complete_http_error():
    """Verify OpenAIProvider raises descriptive RuntimeError on API status errors."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error": {"message": "Invalid API Key"}}')

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="invalid-key", http_client=mock_client)

    async def _run():
        with pytest.raises(RuntimeError, match=r"OpenAI API error \[401\]"):
            await provider.complete([LLMMessage(role="user", content="Hi")])

    asyncio.run(_run())


def test_create_llm_provider_factory():
    """Verify provider factory instantiates correct classes."""
    p_openai = create_llm_provider("openai", model="gpt-4o")
    assert isinstance(p_openai, OpenAIProvider)
    assert p_openai.model == "gpt-4o"
    assert p_openai.provider_name == "openai"

    p_fake = create_llm_provider("fake", model="fake-model")
    assert isinstance(p_fake, FakeLLMProvider)
    assert p_fake.provider_name == "fake"

    with pytest.raises(ValueError, match="Unsupported LLM provider: 'unsupported'"):
        create_llm_provider("unsupported", model="xyz")
