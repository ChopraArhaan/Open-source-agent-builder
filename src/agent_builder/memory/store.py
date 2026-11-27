"""Async session store interface and in-memory implementation."""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from src.agent_builder.memory.models import (
    DEFAULT_MAX_MESSAGE_CHARS,
    DEFAULT_MAX_MESSAGES_PER_SESSION,
    DEFAULT_MAX_METADATA_BYTES,
    Session,
    SessionLimitError,
    SessionNotFoundError,
    SessionValidationError,
    StoredMessage,
    utc_now,
    validate_metadata,
    validate_session_id,
)


class SessionStore(ABC):
    """Abstract async interface for persistent conversation session stores."""

    @abstractmethod
    async def create_session(
        self,
        agent: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create and persist a new conversation session."""
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session:
        """Fetch a session and its ordered messages.

        Raises:
            SessionValidationError: If session_id is not a valid UUIDv4.
            SessionNotFoundError: If no session exists with session_id.
        """
        ...

    @abstractmethod
    async def list_sessions(
        self,
        agent: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions ordered by updated_at descending, optionally filtered by agent."""
        ...

    @abstractmethod
    async def append_messages(
        self,
        session_id: str,
        messages: Sequence[StoredMessage],
    ) -> list[StoredMessage]:
        """Atomically append messages to an existing session.

        Sequence numbers are assigned atomically inside the store transaction.
        All messages commit together or none do.
        """
        ...

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """Return ordered messages for a session (or the most recent `limit` messages)."""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all of its messages.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        ...

    @abstractmethod
    async def purge_older_than(self, cutoff: datetime) -> int:
        """Delete sessions whose updated_at is strictly older than cutoff.

        Returns:
            Number of deleted sessions.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying connections or resources."""
        ...


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class InMemorySessionStore(SessionStore):
    """Thread/async-safe in-memory session store for tests and ephemeral runs."""

    def __init__(
        self,
        max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
        max_messages_per_session: int = DEFAULT_MAX_MESSAGES_PER_SESSION,
        max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.max_message_chars = max_message_chars
        self.max_messages_per_session = max_messages_per_session
        self.max_metadata_bytes = max_metadata_bytes
        self._clock = clock or utc_now
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        agent: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        if not isinstance(agent, str) or not agent.strip():
            raise SessionValidationError("agent must be a non-empty string")

        resolved_id = (
            validate_session_id(session_id)
            if session_id is not None
            else str(uuid.uuid4())
        )
        clean_metadata = validate_metadata(metadata, max_bytes=self.max_metadata_bytes)

        async with self._lock:
            if resolved_id in self._sessions:
                raise SessionValidationError(f"Session '{resolved_id}' already exists")

            now = _ensure_utc(self._clock())
            session = Session(
                session_id=resolved_id,
                agent=agent.strip(),
                created_at=now,
                updated_at=now,
                metadata=clean_metadata,
                messages=[],
            )
            self._sessions[resolved_id] = session
            return session.model_copy(deep=True)

    async def get_session(self, session_id: str) -> Session:
        resolved_id = validate_session_id(session_id)
        async with self._lock:
            session = self._sessions.get(resolved_id)
            if session is None:
                raise SessionNotFoundError(resolved_id)
            return session.model_copy(deep=True)

    async def list_sessions(
        self,
        agent: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        if limit < 1:
            raise SessionValidationError("limit must be >= 1")
        if offset < 0:
            raise SessionValidationError("offset must be >= 0")

        async with self._lock:
            items = list(self._sessions.values())
            if agent is not None:
                items = [s for s in items if s.agent == agent]

            items.sort(key=lambda s: (s.updated_at, s.session_id), reverse=True)
            sliced = items[offset : offset + limit]
            return [s.model_copy(deep=True) for s in sliced]

    async def append_messages(
        self,
        session_id: str,
        messages: Sequence[StoredMessage],
    ) -> list[StoredMessage]:
        resolved_id = validate_session_id(session_id)

        async with self._lock:
            session = self._sessions.get(resolved_id)
            if session is None:
                raise SessionNotFoundError(resolved_id)

            if not messages:
                return []

            if len(session.messages) + len(messages) > self.max_messages_per_session:
                raise SessionLimitError(
                    f"Appending {len(messages)} message(s) would exceed limit of "
                    f"{self.max_messages_per_session} messages per session"
                )

            next_seq = (session.messages[-1].seq + 1) if session.messages else 1
            staged: list[StoredMessage] = []
            for offset, msg in enumerate(messages):
                if len(msg.content) > self.max_message_chars:
                    raise SessionLimitError(
                        f"Message length ({len(msg.content)} chars) exceeds limit "
                        f"({self.max_message_chars} chars)"
                    )
                staged.append(
                    StoredMessage(
                        role=msg.role,
                        content=msg.content,
                        created_at=_ensure_utc(msg.created_at),
                        seq=next_seq + offset,
                    )
                )

            # Commit atomically
            session.messages.extend(staged)
            session.updated_at = _ensure_utc(self._clock())
            return [m.model_copy(deep=True) for m in staged]

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        resolved_id = validate_session_id(session_id)
        if limit is not None and limit < 0:
            raise SessionValidationError("limit must be >= 0")

        async with self._lock:
            session = self._sessions.get(resolved_id)
            if session is None:
                raise SessionNotFoundError(resolved_id)

            if limit == 0:
                return []
            msgs = session.messages if limit is None else session.messages[-limit:]
            return [m.model_copy(deep=True) for m in msgs]

    async def delete_session(self, session_id: str) -> None:
        resolved_id = validate_session_id(session_id)
        async with self._lock:
            if resolved_id not in self._sessions:
                raise SessionNotFoundError(resolved_id)
            del self._sessions[resolved_id]

    async def purge_older_than(self, cutoff: datetime) -> int:
        cutoff_utc = _ensure_utc(cutoff)
        async with self._lock:
            to_delete = [
                sid
                for sid, session in self._sessions.items()
                if _ensure_utc(session.updated_at) < cutoff_utc
            ]
            for sid in to_delete:
                del self._sessions[sid]
            return len(to_delete)

    async def close(self) -> None:
        return None
