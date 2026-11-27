"""Async session store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from src.agent_builder.memory.models import Session, StoredMessage


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
