"""Test suite for observability: ExecutionTracer, trace_stream, and structured logging."""

import asyncio
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.llm.openai import OpenAIProvider
from src.agent_builder.models import (
    LLMResponse,
    RunEvent,
    ToolCall,
    ToolResult,
)
from src.agent_builder.observability import (
    ExecutionTracer,
    log_event,
    mask_sensitive_data,
    setup_logger,
    trace_stream,
    truncate_text,
)
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


def test_tracer_lifecycle_and_statuses():
    """Verify tracer start, observe, finalize, and status transitions for final, error codes, and cancel."""
    # 1. Final status
    tracer_final = ExecutionTracer(agent="test-agent")
    tracer_final.observe(RunEvent(type="thought", step=1, content="Thinking"))
    assert tracer_final.status == "running"
    tracer_final.observe(RunEvent(type="final", step=1, content="Done"))
    assert tracer_final.status == "final"
    tracer_final.finalize()
    assert tracer_final.status == "final"
    assert tracer_final.started_at is not None
    assert tracer_final.ended_at is not None

    # 2. Error codes
    for err_content, expected_code in [
        ("Execution timed out after 30.0s", "timeout"),
        ("Cycle detected: tool 'calc' called 3 times", "cycle"),
        ("Agent exceeded maximum iterations (10)", "max_iterations"),
        ("LLM Provider error: 500 Internal Server Error", "llm_error"),
        ("Generic failure", "error"),
    ]:
        t = ExecutionTracer()
        t.observe(RunEvent(type="error", step=2, content=err_content))
        assert t.status == expected_code
        t.finalize()
        assert t.status == expected_code

    # 3. Explicit error code in metadata
    t_meta = ExecutionTracer()
    t_meta.observe(RunEvent(type="error", step=1, content="Custom", metadata={"code": "custom_err"}))
    assert t_meta.status == "custom_err"

    # 4. Cancelled status on early finalize
    tracer_cancelled = ExecutionTracer()
    tracer_cancelled.start()
    tracer_cancelled.finalize()
    assert tracer_cancelled.status == "cancelled"


def test_metrics_with_fake_clock():
    """Verify exact tool durations, step count, tool calls, and error counts using injectable clock."""
    current_time = 1000.0
    wall_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def fake_monotonic() -> float:
        return current_time

    def fake_wall() -> datetime:
        return wall_time

    tracer = ExecutionTracer(
        agent="clock-bot",
        run_id="run-123",
        clock_monotonic=fake_monotonic,
        clock_wall=fake_wall,
    )

    tracer.start()
    assert tracer.started_at == wall_time

    # Step 1: Tool call
    current_time = 1000.5
    tracer.observe(RunEvent(type="tool_call", step=1, tool_call=ToolCall(name="calc", arguments={"expr": "2+2"})))

    # Step 1: Observation (fallback timing using clock: 1000.5 -> 1001.2 = 700ms)
    current_time = 1001.2
    tracer.observe(RunEvent(type="observation", step=1, tool_result=ToolResult(call_id="c1", tool_name="calc", output="4")))

    # Step 2: Failed tool call with explicit engine duration_ms
    current_time = 1002.0
    tracer.observe(RunEvent(type="tool_call", step=2, tool_call=ToolCall(name="calc", arguments={"expr": "1/0"})))
    current_time = 1002.5
    tracer.observe(RunEvent(
        type="observation",
        step=2,
        tool_result=ToolResult(call_id="c2", tool_name="calc", is_error=True, error_message="Div by 0", duration_ms=12.5),
    ))

    # Step 3: Final
    current_time = 1005.0
    tracer.observe(RunEvent(type="final", step=3, content="Finished"))
    tracer.finalize()

    trace_dict = tracer.to_dict()
    metrics = trace_dict["metrics"]

    assert metrics["step_count"] == 3
    assert metrics["tool_call_count"] == 2
    assert metrics["tool_calls_by_name"] == {"calc": 2}
    assert metrics["tool_errors_by_name"] == {"calc": 1}
    assert metrics["total_duration_ms"] == 5000.0  # 1005.0 - 1000.0

    # Step 1 used clock timing (700.0ms)
    assert trace_dict["steps"][0]["duration_ms"] == 700.0
    # Step 2 used engine duration_ms (12.5ms)
    assert trace_dict["steps"][1]["duration_ms"] == 12.5


def test_token_usage_aggregation():
    """Verify token usage accumulates when reported in metadata, and is None when omitted."""
    # 1. Not reported at all -> None
    tracer_none = ExecutionTracer()
    tracer_none.observe(RunEvent(type="thought", step=1, content="Hi"))
    tracer_none.observe(RunEvent(type="final", step=1, content="Bye"))
    assert tracer_none.token_usage is None
    assert tracer_none.to_dict()["metrics"]["token_usage"] is None

    # 2. Reported in metadata -> Summed
    tracer_usage = ExecutionTracer()
    tracer_usage.observe(RunEvent(
        type="thought",
        step=1,
        content="Step 1",
        metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ))
    tracer_usage.observe(RunEvent(
        type="final",
        step=2,
        content="Step 2",
        metadata={"usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}},
    ))
    assert tracer_usage.token_usage == {
        "prompt_tokens": 30,
        "completion_tokens": 15,
        "total_tokens": 45,
    }


def test_trace_stream_sync_and_async_pass_through():
    """Verify trace_stream passes events through unchanged and in order."""
    events = [
        RunEvent(type="thought", step=1, content="T1"),
        RunEvent(type="final", step=1, content="F1"),
    ]

    # Test sync generator
    tracer_sync = ExecutionTracer()
    yielded_sync = list(trace_stream(events, tracer_sync))
    assert yielded_sync == events
    assert tracer_sync.status == "final"

    # Test async generator
    async def _test_async():
        async def event_gen():
            for e in events:
                yield e

        tracer_async = ExecutionTracer()
        yielded_async = [ev async for ev in trace_stream(event_gen(), tracer_async)]
        assert yielded_async == events
        assert tracer_async.status == "final"

    asyncio.run(_test_async())


def test_trace_stream_early_close_marks_cancelled():
    """Verify closing stream early finalizes with cancelled status and does not raise."""
    async def _test_early_close():
        async def infinite_events():
            step = 1
            while True:
                yield RunEvent(type="thought", step=step, content="Loop")
                step += 1

        tracer = ExecutionTracer()
        stream = trace_stream(infinite_events(), tracer)

        # Consume 2 items then break / close
        count = 0
        async for _ in stream:
            count += 1
            if count == 2:
                await stream.aclose()
                break

        assert tracer._finalized is True
        assert tracer.status == "cancelled"

    asyncio.run(_test_early_close())


def test_concurrent_tracers_isolation():
    """Verify concurrent tracers do not share mutable state or run_id."""
    t1 = ExecutionTracer(agent="agent-1", run_id="id-1")
    t2 = ExecutionTracer(agent="agent-2", run_id="id-2")

    t1.observe(RunEvent(type="tool_call", step=1, tool_call=ToolCall(name="t1_tool")))
    t2.observe(RunEvent(type="tool_call", step=1, tool_call=ToolCall(name="t2_tool")))
    t2.observe(RunEvent(type="tool_call", step=2, tool_call=ToolCall(name="t2_tool")))

    assert t1.tool_call_count == 1
    assert t1.tool_calls_by_name == {"t1_tool": 1}
    assert t2.tool_call_count == 2
    assert t2.tool_calls_by_name == {"t2_tool": 2}
    assert t1.run_id != t2.run_id


def test_trace_export_json_and_file(tmp_path: Path):
    """Verify to_json produces valid JSON, excludes system prompt, writes file, and handles non-JSON natives."""
    tracer = ExecutionTracer(agent="export-bot", run_id="export-123")
    tracer.observe(RunEvent(type="thought", step=1, content="Thinking"))
    tracer.observe(RunEvent(type="final", step=1, content="Done"))
    tracer.finalize()

    out_file = tmp_path / "traces" / "trace.json"
    json_str = tracer.to_json(path=out_file)

    assert out_file.is_file()
    assert out_file.read_text(encoding="utf-8") == json_str

    parsed = json.loads(json_str)
    assert parsed["schema_version"] == 1
    assert parsed["run_id"] == "export-123"
    assert parsed["agent"] == "export-bot"
    assert parsed["status"] == "final"
    assert "system_prompt" not in parsed
    assert "system_prompt" not in parsed["metrics"]


def test_logger_formatting_and_masking():
    """Verify JSON line formatting, levels, masking of sensitive keys, and truncation."""
    stream = io.StringIO()
    logger = setup_logger(name="test_logger", level="DEBUG", json=True, log_content=False, stream=stream)

    # 1. Masking helper
    secret_dict = {
        "api_key": "sk-12345",
        "nested": {"PASSWORD": "secretpassword", "safe": "value"},
        "token_list": [{"auth_token": "bearer xyz"}],
    }
    masked = mask_sensitive_data(secret_dict)
    assert masked["api_key"] == "[REDACTED]"
    assert masked["nested"]["PASSWORD"] == "[REDACTED]"
    assert masked["nested"]["safe"] == "value"
    assert masked["token_list"][0]["auth_token"] == "[REDACTED]"

    # 2. Truncation helper
    long_str = "x" * 600
    truncated = truncate_text(long_str, max_chars=100)
    assert len(truncated) < 600
    assert "...[truncated 500 chars]" in truncated

    # 3. Log event dispatches
    event = RunEvent(
        type="tool_call",
        step=1,
        tool_call=ToolCall(name="db_query", arguments={"secret_token": "supersecret", "query": "SELECT 1"}),
    )
    log_event(logger, "run-abc", event)

    lines = [line for line in stream.getvalue().split("\n") if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["level"] == "DEBUG"
    assert record["run_id"] == "run-abc"
    assert record["event_type"] == "tool_call"
    assert record["step"] == 1
    assert "[REDACTED]" in record["message"]
    assert "supersecret" not in record["message"]


def test_logger_idempotency_and_root_untouched():
    """Verify setup_logger does not duplicate handlers and leaves root logger untouched."""
    root_handlers_before = list(logging.getLogger().handlers)

    s1 = io.StringIO()
    l1 = setup_logger(name="idempotent_test", stream=s1)
    s2 = io.StringIO()
    l2 = setup_logger(name="idempotent_test", stream=s2)

    assert l1 is l2
    # Only 1 handler attached on the logger
    assert len([h for h in l1.handlers if not isinstance(h, logging.NullHandler)]) == 1
    # Root logger untouched
    assert logging.getLogger().handlers == root_handlers_before


def test_end_to_end_react_engine_with_trace_stream():
    """Run real ReActEngine with FakeLLMProvider and built-in tools through trace_stream."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # 1. Normal run
    fake_llm = FakeLLMProvider(responses=[
        [ToolCall(name="calculator", arguments={"expression": "100 / 4"})],
        "Result is 25.",
    ])
    engine = ReActEngine(llm=fake_llm, tools=registry)

    tracer = ExecutionTracer(agent="math-agent")

    async def _run():
        events = []
        async for event in trace_stream(engine.run_stream("Compute 100/4"), tracer):
            events.append(event)
        return events

    yielded_events = asyncio.run(_run())
    assert len(yielded_events) == 3
    assert tracer.status == "final"
    assert tracer.tool_call_count == 1
    assert tracer.tool_calls_by_name == {"calculator": 1}
    assert tracer.tool_errors_by_name == {}
    assert len(tracer.steps) == 2

    # 2. Tool failure run
    fake_llm_fail = FakeLLMProvider(responses=[
        [ToolCall(name="calculator", arguments={"expression": "10 / 0"})],
        "Cannot divide by zero.",
    ])
    engine_fail = ReActEngine(llm=fake_llm_fail, tools=registry)
    tracer_fail = ExecutionTracer(agent="math-fail")

    async def _run_fail():
        async for _ in trace_stream(engine_fail.run_stream("Compute 10/0"), tracer_fail):
            pass

    asyncio.run(_run_fail())
    assert tracer_fail.status == "final"
    assert tracer_fail.tool_errors_by_name == {"calculator": 1}

    # 3. Max iterations run
    call = ToolCall(name="calculator", arguments={"expression": "1 + 1"})
    fake_llm_loop = FakeLLMProvider(default_response="Never stops")
    fake_llm_loop.set_responses([[call], [call], [call]])
    engine_loop = ReActEngine(llm=fake_llm_loop, tools=registry, max_iterations=2)
    tracer_loop = ExecutionTracer(agent="loop-agent")

    async def _run_loop():
        async for _ in trace_stream(engine_loop.run_stream("Loop forever"), tracer_loop):
            pass

    asyncio.run(_run_loop())
    assert tracer_loop.status == "max_iterations"


def test_engine_token_usage_forwarded_to_tracer():
    """Verify scripted LLMResponse usage is attached to events and aggregated by ExecutionTracer."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # Canned responses with explicit token usage
    step1_response = LLMResponse(
        content="I will compute 15 * 3",
        tool_calls=[ToolCall(name="calculator", arguments={"expression": "15 * 3"})],
        usage={"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
    )
    step2_response = LLMResponse(
        content="The answer is 45.",
        tool_calls=[],
        usage={"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70},
    )

    fake_llm = FakeLLMProvider(responses=[step1_response, step2_response])
    engine = ReActEngine(llm=fake_llm, tools=registry)
    tracer = ExecutionTracer(agent="usage-bot")

    async def _run():
        events = []
        async for ev in trace_stream(engine.run_stream("What is 15 * 3?"), tracer):
            events.append(ev)
        return events

    events = asyncio.run(_run())

    # Check event metadata
    # Step 1 thought event must carry step 1 usage
    assert events[0].type == "thought"
    assert events[0].metadata.get("usage") == {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}

    # Step 1 tool_call should not duplicate usage
    assert events[1].type == "tool_call"
    assert "usage" not in events[1].metadata

    # Step 2 final event must carry step 2 usage
    assert events[3].type == "final"
    assert events[3].metadata.get("usage") == {"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70}

    # Verify tracer aggregated token_usage is non-None and exact
    assert tracer.token_usage is not None
    assert tracer.token_usage == {
        "prompt_tokens": 100,  # 40 + 60
        "completion_tokens": 25,  # 15 + 10
        "total_tokens": 125,  # 55 + 70
    }

    metrics = tracer.to_dict()["metrics"]
    assert metrics["token_usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }


def test_engine_token_usage_without_thought_forwarded_to_tracer():
    """Verify tool_call event carries usage when LLM emits tool call without thought content."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # Tool call with content=None
    step1_response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(name="calculator", arguments={"expression": "7 + 8"})],
        usage={"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
    )
    step2_response = LLMResponse(
        content="Result is 15.",
        tool_calls=[],
        usage={"prompt_tokens": 35, "completion_tokens": 5, "total_tokens": 40},
    )

    fake_llm = FakeLLMProvider(responses=[step1_response, step2_response])
    engine = ReActEngine(llm=fake_llm, tools=registry)
    tracer = ExecutionTracer(agent="no-thought-bot")

    async def _run():
        events = []
        async for ev in trace_stream(engine.run_stream("Compute 7+8"), tracer):
            events.append(ev)
        return events

    events = asyncio.run(_run())

    # Step 1 tool_call event carries step 1 usage since thought was not emitted
    assert events[0].type == "tool_call"
    assert events[0].metadata.get("usage") == {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35}

    # Step 2 final event carries step 2 usage
    assert events[2].type == "final"
    assert events[2].metadata.get("usage") == {"prompt_tokens": 35, "completion_tokens": 5, "total_tokens": 40}

    assert tracer.token_usage == {
        "prompt_tokens": 60,
        "completion_tokens": 15,
        "total_tokens": 75,
    }


def test_tracer_with_openrouter_payload_shape():
    """Verify ExecutionTracer correctly reports token usage when OpenAIProvider receives OpenRouter payload."""
    openrouter_payload = {
        "id": "gen-openrouter-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 55,
            "completion_tokens": 12,
            "total_tokens": 67,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 0},
            "cost_details": {"upstream_inference_cost": 0.00025},
            "total_cost": 0.00030,
        },
    }

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=openrouter_payload)))
    llm = OpenAIProvider(api_key="mock-key", http_client=mock_client)
    engine = ReActEngine(llm=llm)
    tracer = ExecutionTracer(agent="openrouter-agent")

    async def _run():
        events = []
        async for ev in trace_stream(engine.run_stream("What is 6 * 7?"), tracer):
            events.append(ev)
        return events

    events = asyncio.run(_run())
    assert len(events) == 1
    assert events[0].type == "final"
    assert events[0].content == "The answer is 42."
    assert events[0].metadata.get("usage") == {
        "prompt_tokens": 55,
        "completion_tokens": 12,
        "total_tokens": 67,
    }

    assert tracer.status == "final"
    assert tracer.token_usage == {
        "prompt_tokens": 55,
        "completion_tokens": 12,
        "total_tokens": 67,
    }
    metrics = tracer.to_dict()["metrics"]
    assert metrics["token_usage"] == {
        "prompt_tokens": 55,
        "completion_tokens": 12,
        "total_tokens": 67,
    }



