"""Test suite for ToolRegistry, tool execution, validation, security, and built-ins."""

import asyncio
import time

import pytest
from pydantic import BaseModel, Field

from src.agent_builder.models import (
    LLMMessage,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from src.agent_builder.tools.builtins import (
    calculator,
    register_builtin_tools,
)
from src.agent_builder.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    name: str = Field(description="User name")
    age: int = Field(default=18, description="User age")


def test_schema_output_excludes_timeout():
    """Verify tool schema matches OpenAI function-calling format and excludes runtime timeout."""
    registry = ToolRegistry()
    registry.register(
        name="dummy_tool",
        description="A dummy test tool",
        args_schema=DummyInput,
        func=lambda name, age=18: f"Hello {name}, age {age}",
        timeout_seconds=15.0,
    )

    spec = registry.get_spec("dummy_tool")
    assert isinstance(spec, ToolSpec)
    assert spec.name == "dummy_tool"
    assert spec.timeout_seconds == 15.0

    openai_schema = spec.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "dummy_tool"
    assert openai_schema["function"]["description"] == "A dummy test tool"
    assert "properties" in openai_schema["function"]["parameters"]
    assert "name" in openai_schema["function"]["parameters"]["properties"]

    # Verify runtime fields like timeout_seconds are completely absent from OpenAI schema
    assert "timeout_seconds" not in openai_schema
    assert "timeout_seconds" not in openai_schema["function"]
    assert "timeout" not in openai_schema["function"]

    schemas = registry.get_openai_schemas()
    assert len(schemas) == 1
    assert schemas[0] == openai_schema


def test_sync_and_async_tool_execution():
    """Verify sync and async tools execute properly through the registry."""
    registry = ToolRegistry()

    def sync_greet(name: str, age: int = 18) -> str:
        return f"Sync: {name} is {age}"

    async def async_greet(name: str, age: int = 18) -> str:
        await asyncio.sleep(0.01)
        return f"Async: {name} is {age}"

    registry.register(
        name="sync_greet",
        description="Sync greet",
        args_schema=DummyInput,
        func=sync_greet,
    )
    registry.register(
        name="async_greet",
        description="Async greet",
        args_schema=DummyInput,
        func=async_greet,
    )

    async def _run():
        res_sync = await registry.execute("sync_greet", {"name": "Alice", "age": 25})
        assert not res_sync.is_error
        assert res_sync.output == "Sync: Alice is 25"
        assert res_sync.duration_ms is not None

        res_async = await registry.execute("async_greet", {"name": "Bob", "age": 30})
        assert not res_async.is_error
        assert res_async.output == "Async: Bob is 30"
        assert res_async.duration_ms is not None

    asyncio.run(_run())


def test_tool_call_object_execution():
    """Verify registry accepts ToolCall model directly."""
    registry = ToolRegistry()

    def add_one(age: int, name: str = "Test") -> int:
        return age + 1

    registry.register("add_one", "Add 1 to age", DummyInput, add_one)

    async def _run():
        call = ToolCall(id="call_123", name="add_one", arguments={"name": "Test", "age": 99})
        result = await registry.execute(call)
        assert result.call_id == "call_123"
        assert result.tool_name == "add_one"
        assert not result.is_error
        assert result.output == 100

    asyncio.run(_run())


def test_validation_failure():
    """Verify missing required parameters or wrong types return is_error=True and never raise."""
    registry = ToolRegistry()
    registry.register(
        name="dummy_tool",
        description="A dummy test tool",
        args_schema=DummyInput,
        func=lambda name, age=18: f"{name}-{age}",
    )

    async def _run():
        # Missing required field 'name'
        res_missing = await registry.execute("dummy_tool", {"age": 20})
        assert res_missing.is_error
        assert res_missing.error_message is not None
        assert "Validation failed" in res_missing.error_message
        assert res_missing.output is None

        # Invalid type for 'age'
        res_invalid_type = await registry.execute("dummy_tool", {"name": "Alice", "age": "not-an-int"})
        assert res_invalid_type.is_error
        assert res_invalid_type.error_message is not None
        assert "Validation failed" in res_invalid_type.error_message

    asyncio.run(_run())


def test_tool_exception_handling():
    """Verify tool functions throwing unhandled exceptions return ToolResult with is_error=True."""
    registry = ToolRegistry()

    def buggy_func(name: str, age: int = 18) -> str:
        raise RuntimeError("Database connection suddenly dropped!")

    registry.register("buggy", "A buggy tool", DummyInput, buggy_func)

    async def _run():
        res = await registry.execute("buggy", {"name": "Alice"})
        assert res.is_error
        assert "Database connection suddenly dropped!" in (res.error_message or "")
        assert res.as_text().startswith("Error:")

    asyncio.run(_run())


def test_async_tool_timeout():
    """Verify async tool exceeding timeout returns ToolResult with is_error=True."""
    registry = ToolRegistry()

    async def slow_async_func(name: str, age: int = 18) -> str:
        await asyncio.sleep(0.3)
        return "Finished"

    registry.register(
        name="slow_async",
        description="Slow async tool",
        args_schema=DummyInput,
        func=slow_async_func,
        timeout_seconds=0.05,
    )

    async def _run():
        res = await registry.execute("slow_async", {"name": "Alice"})
        assert res.is_error
        assert "timed out after" in (res.error_message or "")

    asyncio.run(_run())


def test_sync_tool_timeout():
    """Verify sync tool exceeding timeout returns ToolResult with is_error=True."""
    registry = ToolRegistry()

    def slow_sync_func(name: str, age: int = 18) -> str:
        time.sleep(0.3)
        return "Finished sync"

    registry.register(
        name="slow_sync",
        description="Slow sync tool",
        args_schema=DummyInput,
        func=slow_sync_func,
        timeout_seconds=0.05,
    )

    async def _run():
        res = await registry.execute("slow_sync", {"name": "Alice"})
        assert res.is_error
        assert "timed out after" in (res.error_message or "")

    asyncio.run(_run())


def test_duplicate_registration_raises():
    """Verify registering the same tool name twice raises ValueError."""
    registry = ToolRegistry()
    registry.register("test_tool", "desc", DummyInput, lambda name, age=18: "ok")

    with pytest.raises(ValueError, match="already registered"):
        registry.register("test_tool", "duplicate desc", DummyInput, lambda name, age=18: "dup")


def test_unknown_tool_execution():
    """Verify executing an unregistered tool returns is_error=True without raising."""
    registry = ToolRegistry()

    async def _run():
        res = await registry.execute("ghost_tool", {"foo": "bar"})
        assert res.is_error
        assert "Unknown tool" in (res.error_message or "")

    asyncio.run(_run())


def test_calculator_exploits_and_hardening():
    """Verify calculator tool is safe against exploits, DoS, and malicious AST nodes."""
    # 1. Normal safe operations
    assert calculator("2 + 2") == "4"
    assert calculator("10 + 5 * 2") == "20"
    assert calculator("(10 + 5) * 2") == "30"
    assert calculator("sqrt(16) + 3") == "7"
    assert calculator("min(10, 20) + max(1, 5)") == "15"
    assert calculator("abs(-42)") == "42"

    # 2. Huge exponent exploit (DoS prevention)
    with pytest.raises(ValueError, match="exceeds maximum allowed exponent limit"):
        calculator("9 ** 99999999")

    # 3. Huge base exploit
    with pytest.raises(ValueError, match="exceeds maximum allowed base limit"):
        calculator("1000000 ** 2")

    # 4. Long input string exploit
    with pytest.raises(ValueError, match="exceeds maximum length"):
        calculator("1 + " * 150)

    # 5. Malicious code injection attempts
    with pytest.raises(ValueError, match="Function 'Attribute' is not permitted"):
        calculator("__import__('os').system('echo pwned')")

    with pytest.raises(ValueError, match="Function 'open' is not permitted"):
        calculator("open('/etc/passwd')")

    with pytest.raises(ValueError, match="Function 'eval' is not permitted"):
        calculator("eval('1 + 1')")

    with pytest.raises(ValueError, match="Function 'exec' is not permitted"):
        calculator("exec('x = 1')")

    # 6. Disallowed language constructs
    with pytest.raises(ValueError):
        calculator("[x for x in range(10)]")

    # 7. Division by zero
    with pytest.raises(ValueError, match="Division by zero"):
        calculator("10 / 0")

    # 8. Negative square root
    with pytest.raises(ValueError, match="square root of a negative number"):
        calculator("sqrt(-4)")


def test_builtin_tools_registration_and_execution():
    """Verify built-in tools register and execute properly on a registry."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    assert registry.has("calculator")
    assert registry.has("current_time")
    assert registry.has("mock_search")

    async def _run():
        # Calculator
        calc_res = await registry.execute("calculator", {"expression": "25 * 4"})
        assert not calc_res.is_error
        assert calc_res.output == "100"

        # Current time
        time_res = await registry.execute("current_time", {})
        assert not time_res.is_error
        assert "UTC" in time_res.output

        # Mock search
        search_res = await registry.execute("mock_search", {"query": "react agent", "max_results": 2})
        assert not search_res.is_error
        assert isinstance(search_res.output, list)
        assert len(search_res.output) <= 2
        assert any("react" in item["title"].lower() or "agent" in item["title"].lower() for item in search_res.output)

    asyncio.run(_run())


def test_models_validation_constraints():
    """Verify constraints on shared models."""
    # ToolResult requires error_message if is_error=True
    with pytest.raises(ValueError, match="error_message is required when is_error=True"):
        ToolResult(call_id="c1", tool_name="t1", is_error=True, error_message=None)

    # LLMMessage with role='tool' requires tool_call_id
    with pytest.raises(ValueError, match="tool messages require tool_call_id"):
        LLMMessage(role="tool", content="Observation data", tool_call_id=None)

    # Valid message
    msg = LLMMessage(role="tool", content="Observation data", tool_call_id="c1")
    assert msg.role == "tool"

    # ToolResult.as_text
    err_res = ToolResult(call_id="c1", tool_name="t1", is_error=True, error_message="Failed")
    assert err_res.as_text() == "Error: Failed"

    ok_res_str = ToolResult(call_id="c1", tool_name="t1", output="Success data")
    assert ok_res_str.as_text() == "Success data"

    ok_res_obj = ToolResult(call_id="c1", tool_name="t1", output={"count": 5})
    assert ok_res_obj.as_text() == '{"count": 5}'
