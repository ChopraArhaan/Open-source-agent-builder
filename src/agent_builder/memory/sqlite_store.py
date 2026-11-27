"""SQLite-backed async session store using stdlib sqlite3."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from src.agent_builder.memory.models import (
    DEFAULT_MAX_MESSAGE_CHARS,
    DEFAULT_MAX_MESSAGES_PER_SESSION,
    DEFAULT_MAX_METADATA_BYTES,
    SchemaVersionError,
    Session,
    SessionLimitError,
    SessionNotFoundError,
    SessionStoreError,
    SessionValidationError,
    StoredMessage,
    utc_now,
    validate_metadata,
    validate_session_id,
)
from src.agent_builder.memory.store import SessionStore

CURRENT_SCHEMA_VERSION = 1


def _to_iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds")


def _from_iso_utc(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SQLiteSessionStore(SessionStore):
    """Persistent SQLite session store backed by stdlib sqlite3.

    Thread-safety choice: uses a single-worker ThreadPoolExecutor so every
    sqlite3 call executes off the asyncio event loop on the exact thread that
    opened the connection. Explicit `BEGIN IMMEDIATE` transactions ensure
    atomic multi-message appends and contiguous sequence numbers.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
        max_messages_per_session: int = DEFAULT_MAX_MESSAGES_PER_SESSION,
        max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES,
        busy_timeout_ms: int = 5000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        raw_path = (
            str(db_path)
            if db_path is not None
            else os.getenv("AGENT_BUILDER_DB_PATH", "data/sessions.db")
        )
        self.db_path = raw_path
        self.is_memory = raw_path == ":memory:"
        self.max_message_chars = max_message_chars
        self.max_messages_per_session = max_messages_per_session
        self.max_metadata_bytes = max_metadata_bytes
        self.busy_timeout_ms = busy_timeout_ms
        self._clock = clock or utc_now

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="agent-builder-sqlite",
        )
        self._conn: sqlite3.Connection | None = None
        self._closed = False

        # Initialize connection and schema synchronously on the worker thread
        self._executor.submit(self._init_sync).result()

    def _init_sync(self) -> None:
        if not self.is_memory:
            path_obj = Path(self.db_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            if not path_obj.exists():
                try:
                    fd = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(fd)

        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=True,
        )
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")

            if not self.is_memory:
                mode_row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
                mode = str(mode_row[0]).lower() if mode_row else ""
                if mode != "wal":
                    raise SessionStoreError(
                        f"Failed to enable WAL journal mode (got {mode!r})"
                    )

            self._migrate_sync(conn)
        except Exception:
            conn.close()
            raise

        self._conn = conn

    def _migrate_sync(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0]) if row else 0

        if version > CURRENT_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema version {version} is newer than supported "
                f"version {CURRENT_SCHEMA_VERSION}"
            )

        if version == 0:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        agent TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at "
                    "ON sessions(updated_at DESC, session_id DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        seq INTEGER NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                        UNIQUE (session_id, seq)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session_seq "
                    "ON messages(session_id, seq)"
                )
                conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _require_conn(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise SessionStoreError("SQLiteSessionStore is closed")
        return self._conn

    async def _run(self, fn: Callable[[], Any]) -> Any:
        if self._closed:
            raise SessionStoreError("SQLiteSessionStore is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    def _create_session_sync(
        self,
        agent: str,
        session_id: str,
        clean_metadata: dict[str, Any],
    ) -> Session:
        conn = self._require_conn()
        now_dt = self._clock()
        now_iso = _to_iso_utc(now_dt)
        meta_json = json.dumps(clean_metadata, allow_nan=False, separators=(",", ":"))

        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                raise SessionValidationError(f"Session '{session_id}' already exists")

            conn.execute(
                """
                INSERT INTO sessions (session_id, agent, created_at, updated_at, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, agent, now_iso, now_iso, meta_json),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return Session(
            session_id=session_id,
            agent=agent,
            created_at=_from_iso_utc(now_iso),
            updated_at=_from_iso_utc(now_iso),
            metadata=clean_metadata,
            messages=[],
        )

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
        return cast(
            Session,
            await self._run(
                lambda: self._create_session_sync(agent.strip(), resolved_id, clean_metadata)
            ),
        )

    def _get_session_sync(self, session_id: str) -> Session:
        conn = self._require_conn()
        row = conn.execute(
            """
            SELECT session_id, agent, created_at, updated_at, metadata_json
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)

        msg_rows = conn.execute(
            """
            SELECT role, content, created_at, seq
            FROM messages
            WHERE session_id = ?
            ORDER BY seq ASC
            """,
            (session_id,),
        ).fetchall()

        messages = [
            StoredMessage(
                role=cast(Literal["user", "assistant"], m[0]),
                content=m[1],
                created_at=_from_iso_utc(m[2]),
                seq=int(m[3]),
            )
            for m in msg_rows
        ]
        return Session(
            session_id=row[0],
            agent=row[1],
            created_at=_from_iso_utc(row[2]),
            updated_at=_from_iso_utc(row[3]),
            metadata=json.loads(row[4]),
            messages=messages,
        )

    async def get_session(self, session_id: str) -> Session:
        resolved_id = validate_session_id(session_id)
        return cast(Session, await self._run(lambda: self._get_session_sync(resolved_id)))

    def _list_sessions_sync(
        self,
        agent: str | None,
        limit: int,
        offset: int,
    ) -> list[Session]:
        conn = self._require_conn()
        if agent is not None:
            rows = conn.execute(
                """
                SELECT session_id, agent, created_at, updated_at, metadata_json
                FROM sessions
                WHERE agent = ?
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ? OFFSET ?
                """,
                (agent, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT session_id, agent, created_at, updated_at, metadata_json
                FROM sessions
                ORDER BY updated_at DESC, session_id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        sessions: list[Session] = []
        for row in rows:
            sid = row[0]
            msg_rows = conn.execute(
                """
                SELECT role, content, created_at, seq
                FROM messages
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (sid,),
            ).fetchall()
            messages = [
                StoredMessage(
                    role=cast(Literal["user", "assistant"], m[0]),
                    content=m[1],
                    created_at=_from_iso_utc(m[2]),
                    seq=int(m[3]),
                )
                for m in msg_rows
            ]
            sessions.append(
                Session(
                    session_id=sid,
                    agent=row[1],
                    created_at=_from_iso_utc(row[2]),
                    updated_at=_from_iso_utc(row[3]),
                    metadata=json.loads(row[4]),
                    messages=messages,
                )
            )
        return sessions

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
        return cast(
            list[Session],
            await self._run(lambda: self._list_sessions_sync(agent, limit, offset)),
        )

    def _append_messages_sync(
        self,
        session_id: str,
        messages: Sequence[StoredMessage],
    ) -> list[StoredMessage]:
        conn = self._require_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if exists is None:
                raise SessionNotFoundError(session_id)

            if not messages:
                conn.execute("COMMIT")
                return []

            stats = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            current_count = int(stats[0]) if stats else 0
            max_seq = int(stats[1]) if stats else 0

            if current_count + len(messages) > self.max_messages_per_session:
                raise SessionLimitError(
                    f"Appending {len(messages)} message(s) would exceed limit of "
                    f"{self.max_messages_per_session} messages per session"
                )

            staged: list[StoredMessage] = []
            rows_to_insert: list[tuple[str, int, str, str, str]] = []
            for offset, msg in enumerate(messages):
                if len(msg.content) > self.max_message_chars:
                    raise SessionLimitError(
                        f"Message length ({len(msg.content)} chars) exceeds limit "
                        f"({self.max_message_chars} chars)"
                    )
                seq = max_seq + 1 + offset
                created_iso = _to_iso_utc(msg.created_at)
                rows_to_insert.append(
                    (session_id, seq, msg.role, msg.content, created_iso)
                )
                staged.append(
                    StoredMessage(
                        role=msg.role,
                        content=msg.content,
                        created_at=_from_iso_utc(created_iso),
                        seq=seq,
                    )
                )

            conn.executemany(
                """
                INSERT INTO messages (session_id, seq, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

            now_iso = _to_iso_utc(self._clock())
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now_iso, session_id),
            )
            conn.execute("COMMIT")
            return staged
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def append_messages(
        self,
        session_id: str,
        messages: Sequence[StoredMessage],
    ) -> list[StoredMessage]:
        resolved_id = validate_session_id(session_id)
        return cast(
            list[StoredMessage],
            await self._run(lambda: self._append_messages_sync(resolved_id, messages)),
        )

    def _get_messages_sync(
        self,
        session_id: str,
        limit: int | None,
    ) -> list[StoredMessage]:
        conn = self._require_conn()
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if exists is None:
            raise SessionNotFoundError(session_id)

        if limit == 0:
            return []

        if limit is not None:
            rows = conn.execute(
                """
                SELECT role, content, created_at, seq
                FROM (
                    SELECT role, content, created_at, seq
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY seq DESC
                    LIMIT ?
                )
                ORDER BY seq ASC
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT role, content, created_at, seq
                FROM messages
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            ).fetchall()

        return [
            StoredMessage(
                role=cast(Literal["user", "assistant"], r[0]),
                content=r[1],
                created_at=_from_iso_utc(r[2]),
                seq=int(r[3]),
            )
            for r in rows
        ]

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        resolved_id = validate_session_id(session_id)
        if limit is not None and limit < 0:
            raise SessionValidationError("limit must be >= 0")
        return cast(
            list[StoredMessage],
            await self._run(lambda: self._get_messages_sync(resolved_id, limit)),
        )

    def _delete_session_sync(self, session_id: str) -> None:
        conn = self._require_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            if cur.rowcount == 0:
                raise SessionNotFoundError(session_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def delete_session(self, session_id: str) -> None:
        resolved_id = validate_session_id(session_id)
        await self._run(lambda: self._delete_session_sync(resolved_id))

    def _purge_older_than_sync(self, cutoff_iso: str) -> int:
        conn = self._require_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "DELETE FROM sessions WHERE updated_at < ?",
                (cutoff_iso,),
            )
            deleted = int(cur.rowcount)
            conn.execute("COMMIT")
            return deleted
        except Exception:
            conn.execute("ROLLBACK")
            raise

    async def purge_older_than(self, cutoff: datetime) -> int:
        cutoff_iso = _to_iso_utc(cutoff)
        return cast(int, await self._run(lambda: self._purge_older_than_sync(cutoff_iso)))

    def _close_sync(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    async def close(self) -> None:
        if self._closed:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._close_sync)
        self._closed = True
        self._executor.shutdown(wait=True)
