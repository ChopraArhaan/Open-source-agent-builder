"""Structured logging and telemetry for agent state transitions and tool executions."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.agent_builder.models import RunEvent

LOGGER_NAME = "agent_builder"

# Default NullHandler so importing the package never prints anything
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "token",
    "password",
    "secret",
    "authorization",
    "cookie",
}


def mask_sensitive_data(data: Any) -> Any:
    """Recursively mask sensitive values in dictionaries and lists."""
    if isinstance(data, dict):
        masked: dict[str, Any] = {}
        for k, v in data.items():
            key_lower = str(k).lower()
            if isinstance(v, (dict, list)):
                masked[k] = mask_sensitive_data(v)
            elif any(keyword in key_lower for keyword in SENSITIVE_KEYWORDS):
                masked[k] = "[REDACTED]"
            else:
                masked[k] = v
        return masked
    if isinstance(data, list):
        return [mask_sensitive_data(item) for item in data]
    return data


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate long string content with an informative marker."""
    if len(text) > max_chars:
        excess = len(text) - max_chars
        return f"{text[:max_chars]}...[truncated {excess} chars]"
    return text


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", None),
            "event_type": getattr(record, "event_type", None),
            "step": getattr(record, "step", None),
        }
        # Add any custom extra attributes
        for key, value in record.__dict__.items():
            if key not in payload and not key.startswith("_") and key not in (
                "args", "msg", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process",
            ):
                payload[key] = value

        return json.dumps(payload, default=str)


def setup_logger(
    name: str = LOGGER_NAME,
    level: str = "INFO",
    json: bool = True,
    log_content: bool = False,
    stream: Any = None,
) -> logging.Logger:
    """Configure or update a dedicated logger for agent telemetry.

    Idempotent: Re-calling with different options updates the existing logger
    without duplicating handlers. Never modifies the root logger.
    """
    target_logger = logging.getLogger(name)
    target_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    target_logger.propagate = False

    # Find or create our dedicated handler
    handler_attr = "_agent_builder_handler"
    handler: logging.Handler | None = getattr(target_logger, handler_attr, None)

    if handler is None:
        handler = logging.StreamHandler(stream)
        setattr(target_logger, handler_attr, handler)
        target_logger.addHandler(handler)
    else:
        if stream is not None and hasattr(handler, "setStream"):
            handler.setStream(stream)

    if json:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [run:%(run_id)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    # Store configuration options on logger
    target_logger._log_content = log_content  # type: ignore[attr-defined]

    return target_logger


def log_event(
    logger: logging.Logger,
    run_id: str,
    event: RunEvent,
    log_content: bool | None = None,
) -> None:
    """Format and dispatch a log record for a RunEvent."""
    effective_log_content = (
        log_content
        if log_content is not None
        else getattr(logger, "_log_content", False)
    )

    step = event.step
    event_type = event.type
    extra = {
        "run_id": run_id,
        "event_type": event_type,
        "step": step,
    }

    if event_type == "thought":
        content = event.content or ""
        if not effective_log_content:
            content = truncate_text(content)
        logger.debug(f"Step {step}: Thought: {content}", extra=extra)

    elif event_type == "tool_call":
        tool_call = event.tool_call
        tool_name = tool_call.name if tool_call else "unknown"
        raw_args = tool_call.arguments if tool_call else {}
        args = raw_args if effective_log_content else mask_sensitive_data(raw_args)
        logger.debug(f"Step {step}: Tool call '{tool_name}' with args {args}", extra=extra)

    elif event_type == "observation":
        res = event.tool_result
        tool_name = res.tool_name if res else "unknown"
        if res and res.is_error:
            err_msg = res.error_message or "Unknown tool error"
            logger.warning(f"Step {step}: Tool '{tool_name}' error: {err_msg}", extra=extra)
        else:
            output_text = res.as_text() if res else ""
            if not effective_log_content:
                output_text = truncate_text(output_text)
            logger.debug(f"Step {step}: Tool '{tool_name}' observation: {output_text}", extra=extra)

    elif event_type == "final":
        final_text = event.content or ""
        if not effective_log_content:
            final_text = truncate_text(final_text)
        logger.info(f"Step {step}: Final answer: {final_text}", extra=extra)

    elif event_type == "error":
        err_text = event.content or "Unknown execution error"
        logger.error(f"Step {step}: Agent error: {err_text}", extra=extra)
