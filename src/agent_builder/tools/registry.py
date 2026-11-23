"""Tool registry and execution engine with Pydantic validation and timeout enforcement."""

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ValidationError

from src.agent_builder.models import ToolCall, ToolResult, ToolSpec, _new_call_id


class ToolRegistry:
    """Registry managing tool definitions, schema generation, and safe execution.

    Thread Limitation Note on Sync Timeouts:
    -----------------------------------------
    Synchronous tool functions are wrapped via `asyncio.to_thread()` to run inside
    a background worker thread. When a per-tool timeout triggers, `asyncio.wait_for`
    aborts the awaiting coroutine and immediately returns a `ToolResult` with
    `is_error=True`. However, standard Python threads cannot be forcefully terminated
    or preempted from the outside; the worker thread will continue running in the
    background until the underlying synchronous task completes.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._specs: dict[str, ToolSpec] = {}
        self._schemas: dict[str, type[BaseModel]] = {}
        self._raw_funcs: dict[str, Callable[..., Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        func: Callable[..., Any],
        timeout_seconds: float = 30.0,
    ) -> None:
        """Register a tool function with its Pydantic input schema.

        Raises:
            ValueError: If a tool with the same name is already registered,
                        or if timeout_seconds is <= 0.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        # Generate JSON schema from Pydantic model
        schema_dict = args_schema.model_json_schema()

        spec = ToolSpec(
            name=name,
            description=description,
            parameters=schema_dict,
            timeout_seconds=timeout_seconds,
        )

        # Wrap sync functions into async coroutines
        if inspect.iscoroutinefunction(func):
            async_func = func
        else:
            async def async_func(*args: Any, **kwargs: Any) -> Any:
                return await asyncio.to_thread(func, *args, **kwargs)

        self._tools[name] = async_func
        self._specs[name] = spec
        self._schemas[name] = args_schema
        self._raw_funcs[name] = func

    def register_tool(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        timeout_seconds: float = 30.0,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a tool on this registry instance."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                name=name,
                description=description,
                args_schema=args_schema,
                func=func,
                timeout_seconds=timeout_seconds,
            )
            return func
        return decorator

    def get_spec(self, name: str) -> ToolSpec:
        """Retrieve the ToolSpec for a given tool name."""
        if name not in self._specs:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._specs[name]

    def list_specs(self) -> list[ToolSpec]:
        """List all registered tool specifications."""
        return list(self._specs.values())

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Return all registered tools formatted as OpenAI function-calling schemas.

        Note: Deliberately excludes runtime metadata like timeout_seconds.
        """
        return [spec.to_openai_schema() for spec in self._specs.values()]

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    async def execute(
        self,
        name_or_call: str | ToolCall,
        arguments: dict[str, Any] | str | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        """Execute a tool safely.

        Validates arguments against the tool's Pydantic schema, enforces
        per-tool timeouts, and traps all exceptions into a ToolResult.
        Never raises exceptions directly.
        """
        start_time = time.perf_counter()

        # Normalize invocation parameters
        raw_args: dict[str, Any] | str
        if isinstance(name_or_call, ToolCall):
            tool_name = name_or_call.name
            raw_args = name_or_call.arguments
            effective_call_id = name_or_call.id
        else:
            tool_name = name_or_call
            raw_args = arguments if arguments is not None else {}
            effective_call_id = call_id or _new_call_id()

        # 1. Check if tool exists
        if tool_name not in self._tools:
            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                is_error=True,
                error_message=f"Unknown tool: '{tool_name}'",
                duration_ms=0.0,
            )

        # 2. Parse arguments if passed as JSON string
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except (json.JSONDecodeError, ValueError) as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    call_id=effective_call_id,
                    tool_name=tool_name,
                    is_error=True,
                    error_message=f"Failed to parse arguments JSON for tool '{tool_name}': {e}",
                    duration_ms=round(duration_ms, 2),
                )

        if not isinstance(raw_args, dict):
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                is_error=True,
                error_message=f"Arguments for tool '{tool_name}' must be a dictionary or JSON object",
                duration_ms=round(duration_ms, 2),
            )

        # 3. Validate arguments against Pydantic schema
        schema_cls = self._schemas[tool_name]
        try:
            validated_args = schema_cls.model_validate(raw_args)
        except ValidationError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                is_error=True,
                error_message=f"Validation failed for tool '{tool_name}': {e}",
                duration_ms=round(duration_ms, 2),
            )

        # 4. Invoke tool with timeout
        spec = self._specs[tool_name]
        async_func = self._tools[tool_name]
        raw_func = self._raw_funcs[tool_name]

        try:
            # Check if underlying function expects the BaseModel instance or unpacked kwargs
            sig = inspect.signature(raw_func)
            if len(sig.parameters) == 1:
                param = next(iter(sig.parameters.values()))
                if param.annotation == schema_cls:
                    call_coro = async_func(validated_args)
                else:
                    call_coro = async_func(**validated_args.model_dump())
            else:
                call_coro = async_func(**validated_args.model_dump())

            output = await asyncio.wait_for(call_coro, timeout=spec.timeout_seconds)
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                output=output,
                is_error=False,
                duration_ms=round(duration_ms, 2),
            )

        except (asyncio.TimeoutError, TimeoutError):
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                is_error=True,
                error_message=f"Tool '{tool_name}' timed out after {spec.timeout_seconds:.2f}s",
                duration_ms=round(duration_ms, 2),
            )
        except Exception as e:  # noqa: BLE001 - Resilient loop: tools must never crash the engine
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                call_id=effective_call_id,
                tool_name=tool_name,
                is_error=True,
                error_message=f"Tool '{tool_name}' execution error: {e}",
                duration_ms=round(duration_ms, 2),
            )


# Default global registry instance
default_registry = ToolRegistry()


def register_tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    timeout_seconds: float = 30.0,
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator to register a tool on the default (or specified) registry."""
    target_registry = registry if registry is not None else default_registry
    return target_registry.register_tool(
        name=name,
        description=description,
        args_schema=args_schema,
        timeout_seconds=timeout_seconds,
    )
