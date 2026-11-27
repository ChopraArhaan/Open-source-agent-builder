"""Data models and exception hierarchy for persistent conversation sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DEFAULT_MAX_MESSAGE_CHARS = 100_000
DEFAULT_MAX_MESSAGES_PER_SESSION = 1_000
DEFAULT_MAX_METADATA_BYTES = 16_384


class SessionStoreError(Exception):
    """Base exception for all session store errors."""


class SessionNotFoundError(SessionStoreError):
    """Raised when a requested session does not exist."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' not found")


class SessionValidationError(SessionStoreError):
    """Raised when session data or identifiers fail validation."""


class SessionLimitError(SessionStoreError):
    """Raised when a session or message exceeds configured size/count limits."""


class SchemaVersionError(SessionStoreError):
    """Raised when the database schema version is newer than supported."""


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def validate_session_id(session_id: str) -> str:
    """Validate that session_id is a canonical UUIDv4 string."""
    if not isinstance(session_id, str) or not session_id:
        raise SessionValidationError("session_id must be a non-empty UUIDv4 string")
    try:
        parsed = uuid.UUID(session_id)
    except (ValueError, AttributeError, TypeError) as exc:
        raise SessionValidationError(f"Invalid session_id format: {session_id!r}") from exc

    if parsed.version != 4 or str(parsed) != session_id.lower():
        raise SessionValidationError(
            f"session_id must be a valid UUIDv4 string: {session_id!r}"
        )
    return str(parsed)


def validate_metadata(
    metadata: dict[str, Any] | None,
    max_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> dict[str, Any]:
    """Validate that metadata is JSON-safe and within the byte limit."""
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise SessionValidationError("Session metadata must be a dictionary")
    try:
        serialized = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise SessionValidationError(f"Session metadata is not JSON-safe: {exc}") from exc

    byte_len = len(serialized.encode("utf-8"))
    if byte_len > max_bytes:
        raise SessionLimitError(
            f"Session metadata size ({byte_len} bytes) exceeds limit ({max_bytes} bytes)"
        )
    # Round-trip through json.loads to guarantee JSON-normalized copy
    return json.loads(serialized)  # type: ignore[no-any-return]


class StoredMessage(BaseModel):
    """A single persisted user or assistant turn message.

    System prompts and internal ReAct scratchpad events (thoughts, tool calls,
    observations) are intentionally excluded so stored history remains concise.
    """

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    seq: int = Field(default=0, ge=0)


class Session(BaseModel):
    """A persisted conversation session bound to an agent."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[StoredMessage] = Field(default_factory=list)

    @field_validator("session_id")
    @classmethod
    def _check_session_id(cls, value: str) -> str:
        return validate_session_id(value)
