"""Raw ReAct execution engine with event streaming and guardrails."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.models import LLMMessage, RunEvent, ToolCall
from src.agent_builder.tools.registry import ToolRegistry


class ReActEngine:
    """Reasoning + Acting execution loop with unified event streaming.

    Follows a single code path: all executions run through `run_stream()`, which yields
    observable `RunEvent` instances (`thought`, `tool_call`, `observation`, `final`, `error`).
    The convenience `run()` method simply collects this stream.
    """

    def __init__(
        self,
        llm: BaseLLMProvider,
        tools: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        timeout_seconds: float = 120.0,
        max_consecutive_identical_calls: int = 3,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.max_consecutive_identical_calls = max_consecutive_identical_calls

    def _hash_call(self, tool_call: ToolCall) -> str:
        """Create a deterministic signature of a tool call for cycle detection."""
        args_str = json.dumps(tool_call.arguments, sort_keys=True)
        return f"{tool_call.name}:{args_str}"

    async def run_stream(
        self,
        prompt: str,
        session_messages: list[LLMMessage] | None = None,
    ) -> AsyncGenerator[RunEvent, None]:
        """Stream execution events step-by-step through the ReAct loop."""
        start_time = time.perf_counter()

        # Build initial conversation context
        messages: list[LLMMessage] = []
        if self.system_prompt:
            messages.append(LLMMessage(role="system", content=self.system_prompt))

        if session_messages:
            messages.extend(session_messages)

        messages.append(LLMMessage(role="user", content=prompt))

        # Retrieve available tool specifications
        tool_specs = self.tools.list_specs() if self.tools else None

        # Cycle detection state
        consecutive_call_signature: str | None = None
        consecutive_call_count = 0

        for step in range(1, self.max_iterations + 1):
            # 1. Guardrail: Overall timeout check
            remaining_time = self.timeout_seconds - (time.perf_counter() - start_time)
            if remaining_time <= 0:
                yield RunEvent(
                    type="error",
                    step=step,
                    content=f"Execution timed out after {self.timeout_seconds:.1f}s",
                )
                return

            # 2. Call LLM provider with remaining timeout
            try:
                response = await asyncio.wait_for(
                    self.llm.complete(messages=messages, tools=tool_specs),
                    timeout=remaining_time,
                )
            except (asyncio.TimeoutError, TimeoutError):
                yield RunEvent(
                    type="error",
                    step=step,
                    content=f"Execution timed out after {self.timeout_seconds:.1f}s",
                )
                return
            except Exception as e:  # noqa: BLE001
                yield RunEvent(
                    type="error",
                    step=step,
                    content=f"LLM Provider error: {e}",
                )
                return

            # Post-call timeout verification
            if (time.perf_counter() - start_time) >= self.timeout_seconds:
                yield RunEvent(
                    type="error",
                    step=step,
                    content=f"Execution timed out after {self.timeout_seconds:.1f}s",
                )
                return

            # 3. Check for intermediate thought reasoning
            if response.content and response.tool_calls:
                yield RunEvent(
                    type="thought",
                    step=step,
                    content=response.content,
                )

            # 4. Check for final answer (no tools requested)
            if not response.tool_calls:
                final_content = response.content or ""
                messages.append(LLMMessage(role="assistant", content=final_content))
                yield RunEvent(
                    type="final",
                    step=step,
                    content=final_content,
                )
                return

            # 5. Process tool call requests
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                # Cycle check: prevent infinite loops calling identical tools/args
                call_sig = self._hash_call(tool_call)
                if call_sig == consecutive_call_signature:
                    consecutive_call_count += 1
                else:
                    consecutive_call_signature = call_sig
                    consecutive_call_count = 1

                if consecutive_call_count >= self.max_consecutive_identical_calls:
                    yield RunEvent(
                        type="error",
                        step=step,
                        content=(
                            f"Cycle detected: tool '{tool_call.name}' called {consecutive_call_count} "
                            "times consecutively with identical arguments"
                        ),
                        tool_call=tool_call,
                    )
                    return

                # Emit tool call event
                yield RunEvent(
                    type="tool_call",
                    step=step,
                    tool_call=tool_call,
                )

                # Execute tool safely via registry (never raises)
                if self.tools:
                    tool_result = await self.tools.execute(tool_call)
                else:
                    from src.agent_builder.models import ToolResult
                    tool_result = ToolResult(
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        is_error=True,
                        error_message="No tool registry configured on engine",
                        duration_ms=0.0,
                    )

                # Emit observation event
                yield RunEvent(
                    type="observation",
                    step=step,
                    tool_result=tool_result,
                )

                # Add observation to messages for LLM context
                messages.append(
                    LLMMessage(
                        role="tool",
                        content=tool_result.as_text(),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                )

        # 6. Exceeded max iterations
        yield RunEvent(
            type="error",
            step=self.max_iterations,
            content=f"Agent exceeded maximum iterations ({self.max_iterations})",
        )

    async def run(
        self,
        prompt: str,
        session_messages: list[LLMMessage] | None = None,
    ) -> tuple[str, list[RunEvent]]:
        """Synchronously collect the stream of execution events.

        Returns:
            A tuple of (final_text, list_of_all_events).
        """
        events: list[RunEvent] = []
        final_answer = ""

        async for event in self.run_stream(prompt, session_messages=session_messages):
            events.append(event)
            if event.type == "final":
                final_answer = event.content or ""
            elif event.type == "error" and not final_answer:
                final_answer = f"Error: {event.content}"

        return final_answer, events
