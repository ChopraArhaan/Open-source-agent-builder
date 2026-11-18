"""Test suite for ReActEngine execution loop, stop conditions, and event streaming."""

import asyncio
from typing import Any

from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.models import (
    LLMMessage,
    LLMResponse,
    RunEvent,
    ToolCall,
)
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


def test_single_turn_final_answer():
    """Verify single-turn direct answer with no tool calls."""
    fake_llm = FakeLLMProvider(responses=["The capital of France is Paris."])
    engine = ReActEngine(llm=fake_llm)

    async def _run():
        final_answer, events = await engine.run("What is the capital of France?")
        assert final_answer == "The capital of France is Paris."
        assert len(events) == 1
        assert events[0].type == "final"
        assert events[0].step == 1
        assert events[0].content == "The capital of France is Paris."

    asyncio.run(_run())


def test_single_tool_execution_cycle():
    """Verify complete ReAct cycle: thought -> tool_call -> observation -> final."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeLLMProvider(
        responses=[
            [ToolCall(name="calculator", arguments={"expression": "25 * 4"})],
            "The result is 100.",
        ]
    )
    engine = ReActEngine(llm=fake_llm, tools=registry)

    async def _run():
        final_answer, events = await engine.run("Calculate 25 * 4")
        assert final_answer == "The result is 100."
        assert len(events) == 3

        # Step 1: Tool call
        assert events[0].type == "tool_call"
        assert events[0].step == 1
        assert events[0].tool_call is not None
        assert events[0].tool_call.name == "calculator"

        # Step 1: Observation
        assert events[1].type == "observation"
        assert events[1].step == 1
        assert events[1].tool_result is not None
        assert events[1].tool_result.output == "100"
        assert not events[1].tool_result.is_error

        # Step 2: Final answer
        assert events[2].type == "final"
        assert events[2].step == 2
        assert events[2].content == "The result is 100."

    asyncio.run(_run())


def test_thought_intermediate_reasoning():
    """Verify engine emits 'thought' event when LLM provides text reasoning alongside tool calls."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    tool_call = ToolCall(name="calculator", arguments={"expression": "10 + 20"})
    fake_llm = FakeLLMProvider(
        responses=[
            LLMResponse(
                content="I need to compute 10 + 20 first.",
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            ),
            "10 + 20 is 30.",
        ]
    )
    engine = ReActEngine(llm=fake_llm, tools=registry)

    async def _run():
        events: list[RunEvent] = []
        async for ev in engine.run_stream("What is 10 + 20?"):
            events.append(ev)

        assert len(events) == 4
        assert events[0].type == "thought"
        assert events[0].content == "I need to compute 10 + 20 first."
        assert events[1].type == "tool_call"
        assert events[2].type == "observation"
        assert events[3].type == "final"

    asyncio.run(_run())


def test_multi_step_react_loop():
    """Verify engine handles multiple iterative tool steps before finalizing."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeLLMProvider(
        responses=[
            [ToolCall(name="calculator", arguments={"expression": "10 + 10"})],
            [ToolCall(name="calculator", arguments={"expression": "20 * 3"})],
            "The final number is 60.",
        ]
    )
    engine = ReActEngine(llm=fake_llm, tools=registry, max_iterations=5)

    async def _run():
        final_answer, events = await engine.run("Compute (10 + 10) * 3 step by step")
        assert final_answer == "The final number is 60."

        event_types = [e.type for e in events]
        assert event_types == [
            "tool_call",
            "observation",
            "tool_call",
            "observation",
            "final",
        ]
        assert events[0].step == 1
        assert events[1].step == 1
        assert events[2].step == 2
        assert events[3].step == 2
        assert events[4].step == 3

    asyncio.run(_run())


def test_tool_error_resilience():
    """Verify tool errors become observations for the LLM and never crash the engine."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    fake_llm = FakeLLMProvider(
        responses=[
            # Step 1: Malformed expression causing error in calculator
            [ToolCall(name="calculator", arguments={"expression": "10 / 0"})],
            # Step 2: LLM sees division by zero error, recovers and explains
            "Division by zero is undefined.",
        ]
    )
    engine = ReActEngine(llm=fake_llm, tools=registry)

    async def _run():
        final_answer, events = await engine.run("Calculate 10 / 0")
        assert final_answer == "Division by zero is undefined."
        assert len(events) == 3

        # Observation event must record the error gracefully
        obs_event = events[1]
        assert obs_event.type == "observation"
        assert obs_event.tool_result is not None
        assert obs_event.tool_result.is_error
        assert "Division by zero" in (obs_event.tool_result.error_message or "")

        # Final answer completed normally
        assert events[2].type == "final"

    asyncio.run(_run())


def test_max_iterations_termination():
    """Verify engine halts with error event when max_iterations is reached."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    # Infinite tool loop
    call = ToolCall(name="current_time", arguments={})
    fake_llm = FakeLLMProvider(default_response="Never reaching final")
    # Always return tool calls
    fake_llm.set_responses([[call], [call], [call], [call]])

    engine = ReActEngine(llm=fake_llm, tools=registry, max_iterations=2)

    async def _run():
        final_answer, events = await engine.run("Keep looping")
        assert "Agent exceeded maximum iterations (2)" in final_answer

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Agent exceeded maximum iterations (2)" in (error_events[0].content or "")

    asyncio.run(_run())


def test_cycle_detection():
    """Verify engine detects consecutive identical tool calls and halts early."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    call = ToolCall(name="calculator", arguments={"expression": "2 + 2"})
    fake_llm = FakeLLMProvider(
        responses=[[call], [call], [call]]
    )
    engine = ReActEngine(
        llm=fake_llm,
        tools=registry,
        max_consecutive_identical_calls=3,
        max_iterations=10,
    )

    async def _run():
        final_answer, events = await engine.run("Loop forever")
        assert "Cycle detected" in final_answer

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Cycle detected" in (error_events[0].content or "")

    asyncio.run(_run())


def test_overall_timeout():
    """Verify engine halts when overall timeout is reached."""
    class SlowFakeLLM(FakeLLMProvider):
        async def complete(self, messages: list[LLMMessage], tools: Any = None, **kwargs: Any) -> LLMResponse:
            await asyncio.sleep(0.1)
            return await super().complete(messages, tools, **kwargs)

    slow_llm = SlowFakeLLM(default_response="Slow answer")
    engine = ReActEngine(llm=slow_llm, timeout_seconds=0.05)

    async def _run():
        final_answer, events = await engine.run("Slow question")
        assert "timed out" in final_answer
        assert any(e.type == "error" and "timed out" in (e.content or "") for e in events)

    asyncio.run(_run())


def test_llm_provider_error():
    """Verify unhandled LLM provider exception results in clean error event."""
    fake_llm = FakeLLMProvider(responses=[RuntimeError("API gateway connection refused")])
    engine = ReActEngine(llm=fake_llm)

    async def _run():
        final_answer, events = await engine.run("Hello")
        assert "LLM Provider error: API gateway connection refused" in final_answer
        assert len(events) == 1
        assert events[0].type == "error"

    asyncio.run(_run())


def test_system_prompt_and_session_messages():
    """Verify system prompt and prior conversation context are passed to LLM."""
    fake_llm = FakeLLMProvider(responses=["Acknowledged."])
    engine = ReActEngine(llm=fake_llm, system_prompt="You are a specialized code reviewer.")

    prior_history = [
        LLMMessage(role="user", content="Previous question"),
        LLMMessage(role="assistant", content="Previous answer"),
    ]

    async def _run():
        await engine.run("New question", session_messages=prior_history)
        last_call = fake_llm.last_call()
        assert last_call is not None
        messages = last_call["messages"]
        assert len(messages) == 4
        assert messages[0].role == "system"
        assert messages[0].content == "You are a specialized code reviewer."
        assert messages[1].role == "user"
        assert messages[1].content == "Previous question"
        assert messages[2].role == "assistant"
        assert messages[2].content == "Previous answer"
        assert messages[3].role == "user"
        assert messages[3].content == "New question"

    asyncio.run(_run())
