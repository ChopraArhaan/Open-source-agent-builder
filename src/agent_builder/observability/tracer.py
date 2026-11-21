"""Execution tracer for agent runs, computing step metrics and lifecycle traces."""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Generator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent_builder.models import RunEvent
from src.agent_builder.observability.logger import log_event


def _extract_error_code(event: RunEvent) -> str:
    """Extract a normalized error code from an error RunEvent."""
    meta = getattr(event, "metadata", None) or getattr(event, "data", None) or {}
    if isinstance(meta, dict) and "code" in meta:
        return str(meta["code"])

    content = (event.content or "").lower()
    if "timed out" in content or "timeout" in content:
        return "timeout"
    if "cycle detected" in content or "cycle" in content:
        return "cycle"
    if "exceeded maximum iterations" in content or "max_iterations" in content:
        return "max_iterations"
    if "llm provider error" in content or "provider error" in content:
        return "llm_error"

    return "error"


class ExecutionTracer:
    """Non-intrusive observer collecting telemetry, metrics, and event traces per agent run.

    Timing Caveat:
    -------------
    When tool duration is not directly provided in `ToolResult.duration_ms`, the tracer
    measures duration between `tool_call` and `observation` events as seen by the observer.
    In streaming contexts, this duration includes event propagation and consumer latency.
    """

    def __init__(
        self,
        agent: str = "agent",
        run_id: str | None = None,
        logger: logging.Logger | None = None,
        log_enabled: bool = False,
        clock_monotonic: Callable[[], float] = time.monotonic,
        clock_wall: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.agent = agent
        self.run_id = run_id or str(uuid.uuid4())
        self.logger = logger
        self.log_enabled = log_enabled
        self.clock_monotonic = clock_monotonic
        self.clock_wall = clock_wall

        # Lifecycle state
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self._start_mono: float | None = None
        self._end_mono: float | None = None
        self.status: str = "running"
        self._finalized: bool = False

        # Step and tool tracking
        self.steps: list[dict[str, Any]] = []
        self._active_steps: dict[int, dict[str, Any]] = {}
        self._pending_tool_calls: dict[str, tuple[int, float]] = {}  # tool_name -> (step, start_mono)

        # Aggregated metrics
        self._observed_steps: set[int] = set()
        self.tool_call_count: int = 0
        self.tool_calls_by_name: dict[str, int] = {}
        self.tool_errors_by_name: dict[str, int] = {}
        self.token_usage: dict[str, int] | None = None

    def start(self) -> None:
        """Mark the beginning of the execution run."""
        if self.started_at is None:
            self.started_at = self.clock_wall()
            self._start_mono = self.clock_monotonic()
            if self.log_enabled and self.logger:
                self.logger.info(
                    f"Run started for agent '{self.agent}'",
                    extra={"run_id": self.run_id, "event_type": "run_start"},
                )

    def _get_or_create_step(self, step_num: int) -> dict[str, Any]:
        """Retrieve or initialize the record for a given step number."""
        if step_num not in self._active_steps:
            record: dict[str, Any] = {
                "step": step_num,
                "thought": None,
                "tool_name": None,
                "tool_arguments": None,
                "observation": None,
                "ok": None,
                "duration_ms": None,
            }
            self._active_steps[step_num] = record
            self.steps.append(record)
        return self._active_steps[step_num]

    def observe(self, event: RunEvent) -> None:
        """Process an observed RunEvent and update metrics."""
        self.start()

        if event.step is not None:
            self._observed_steps.add(event.step)

        # Log event if logging is active
        if self.log_enabled and self.logger:
            log_event(self.logger, self.run_id, event)

        # Accumulate token usage if exposed in metadata
        meta = getattr(event, "metadata", None) or getattr(event, "data", None) or {}
        if isinstance(meta, dict):
            usage = meta.get("usage") or meta.get("token_usage")
            if isinstance(usage, dict):
                if self.token_usage is None:
                    self.token_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

        # Update step records and metrics based on event type
        step_num = event.step or 1

        if event.type == "thought":
            record = self._get_or_create_step(step_num)
            record["thought"] = event.content

        elif event.type == "tool_call":
            record = self._get_or_create_step(step_num)
            tool_call = event.tool_call
            tool_name = tool_call.name if tool_call else "unknown"
            record["tool_name"] = tool_name
            record["tool_arguments"] = tool_call.arguments if tool_call else {}

            self.tool_call_count += 1
            self.tool_calls_by_name[tool_name] = self.tool_calls_by_name.get(tool_name, 0) + 1
            self._pending_tool_calls[tool_name] = (step_num, self.clock_monotonic())

        elif event.type == "observation":
            record = self._get_or_create_step(step_num)
            tool_result = event.tool_result
            tool_name = tool_result.tool_name if tool_result else "unknown"
            record["observation"] = tool_result.as_text() if tool_result else ""
            record["ok"] = not tool_result.is_error if tool_result else False

            if tool_result and tool_result.is_error:
                self.tool_errors_by_name[tool_name] = self.tool_errors_by_name.get(tool_name, 0) + 1

            # Determine duration
            if tool_result and tool_result.duration_ms is not None:
                record["duration_ms"] = tool_result.duration_ms
            elif tool_name in self._pending_tool_calls:
                _, call_mono = self._pending_tool_calls.pop(tool_name)
                record["duration_ms"] = round((self.clock_monotonic() - call_mono) * 1000, 2)

        elif event.type == "final":
            self.status = "final"
            record = self._get_or_create_step(step_num)
            if not record["thought"]:
                record["thought"] = event.content

        elif event.type == "error":
            self.status = _extract_error_code(event)
            record = self._get_or_create_step(step_num)
            record["observation"] = event.content
            record["ok"] = False

    def finalize(self) -> None:
        """Mark run as ended and lock final status."""
        if self._finalized:
            return

        self._finalized = True
        self.ended_at = self.clock_wall()
        self._end_mono = self.clock_monotonic()

        # If ended without final or error event, mark as cancelled
        if self.status == "running":
            self.status = "cancelled"

        if self.log_enabled and self.logger:
            duration_ms = self.total_duration_ms
            self.logger.info(
                f"Run finished for agent '{self.agent}' with status '{self.status}' in {duration_ms:.1f}ms",
                extra={
                    "run_id": self.run_id,
                    "event_type": "run_end",
                    "status": self.status,
                    "duration_ms": duration_ms,
                },
            )

    @property
    def total_duration_ms(self) -> float:
        """Calculate total execution time in milliseconds."""
        if self._start_mono is None:
            return 0.0
        end = self._end_mono if self._end_mono is not None else self.clock_monotonic()
        return round((end - self._start_mono) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Export the trace into a structured dictionary."""
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "agent": self.agent,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status,
            "metrics": {
                "step_count": len(self._observed_steps),
                "tool_call_count": self.tool_call_count,
                "tool_calls_by_name": dict(self.tool_calls_by_name),
                "tool_errors_by_name": dict(self.tool_errors_by_name),
                "total_duration_ms": self.total_duration_ms,
                "token_usage": self.token_usage,
            },
            "steps": self.steps,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize trace to JSON. If path is provided, writes to disk."""
        data_str = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(data_str, encoding="utf-8")
        return data_str


def trace_stream(
    events: Any,
    tracer: ExecutionTracer,
) -> Any:
    """Pass-through stream wrapper that observes RunEvents without altering them.

    Supports both AsyncIterable and synchronous Iterable event sequences.
    Always finalizes the tracer in a `finally` block, marking the trace
    as `cancelled` if the stream is closed early.
    """
    if hasattr(events, "__aiter__"):
        async def _async_gen() -> AsyncGenerator[RunEvent, None]:
            tracer.start()
            try:
                async for event in events:
                    tracer.observe(event)
                    yield event
            finally:
                tracer.finalize()

        return _async_gen()

    def _sync_gen() -> Generator[RunEvent, None, None]:
        tracer.start()
        try:
            for event in events:
                tracer.observe(event)
                yield event
        finally:
            tracer.finalize()

    return _sync_gen()
