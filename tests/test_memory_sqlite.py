"""SQLite-specific tests for SQLiteSessionStore."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from src.agent_builder.memory import (
    SchemaVersionError,
    SessionStoreError,
    SQLiteSessionStore,
    StoredMessage,
)
from src.agent_builder.memory.sqlite_store import CURRENT_SCHEMA_VERSION


def test_sqlite_persistence_across_close_and_reopen(tmp_path: Path) -> None:
    """Data written before close() must persist and be readable after reopening the file."""
    db_file = tmp_path / "nested" / "dir" / "sessions.db"

    async def _run() -> None:
        store1 = SQLiteSessionStore(db_path=db_file)
        s = await store1.create_session(agent="math-tutor", metadata={"k": "v"})
        await store1.append_messages(
            s.session_id,
            [
                StoredMessage(role="user", content="What is 2+2?"),
                StoredMessage(role="assistant", content="4"),
            ],
        )
        await store1.close()
        # Idempotent close
        await store1.close()

        store2 = SQLiteSessionStore(db_path=db_file)
        try:
            fetched = await store2.get_session(s.session_id)
            assert fetched.session_id == s.session_id
            assert fetched.metadata == {"k": "v"}
            assert [m.content for m in fetched.messages] == ["What is 2+2?", "4"]
            assert [m.seq for m in fetched.messages] == [1, 2]
        finally:
            await store2.close()

    asyncio.run(_run())


def test_sqlite_pragmas_wal_fk_busy_timeout_and_user_version(tmp_path: Path) -> None:
    """Verify WAL journal mode, foreign_keys=ON, busy_timeout, and user_version are set."""
    db_file = tmp_path / "pragmas.db"

    async def _run() -> None:
        store = SQLiteSessionStore(db_path=db_file, busy_timeout_ms=4321)
        try:
            def _inspect() -> tuple[str, int, int, int]:
                conn = store._require_conn()
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                bt = conn.execute("PRAGMA busy_timeout").fetchone()[0]
                uv = conn.execute("PRAGMA user_version").fetchone()[0]
                return str(mode).lower(), int(fk), int(bt), int(uv)

            mode, fk, bt, uv = await store._run(_inspect)
            assert mode == "wal"
            assert fk == 1
            assert bt == 4321
            assert uv == CURRENT_SCHEMA_VERSION
        finally:
            await store.close()

    asyncio.run(_run())


def test_sqlite_rejects_newer_schema_version(tmp_path: Path) -> None:
    """Opening a DB with user_version > CURRENT_SCHEMA_VERSION must raise SchemaVersionError."""
    db_file = tmp_path / "future.db"
    conn = sqlite3.connect(db_file)
    conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 10}")
    conn.close()

    with pytest.raises(SchemaVersionError, match="newer than supported"):
        SQLiteSessionStore(db_path=db_file)


def test_sqlite_cascade_delete_removes_child_messages(tmp_path: Path) -> None:
    """Deleting a session cascades to delete its rows from the messages table."""
    db_file = tmp_path / "cascade.db"

    async def _run() -> None:
        store = SQLiteSessionStore(db_path=db_file)
        try:
            s = await store.create_session(agent="alpha")
            await store.append_messages(
                s.session_id,
                [
                    StoredMessage(role="user", content="hello"),
                    StoredMessage(role="assistant", content="world"),
                ],
            )
            await store.delete_session(s.session_id)

            count = await store._run(
                lambda: store._require_conn()
                .execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (s.session_id,))
                .fetchone()[0]
            )
            assert count == 0
        finally:
            await store.close()

    asyncio.run(_run())


def test_sqlite_parameterized_query_safety_and_unicode(tmp_path: Path) -> None:
    """SQL injection strings, quotes, unicode, emoji, and long text round-trip safely."""
    db_file = tmp_path / "safety.db"
    tricky_strings = [
        "'; DROP TABLE sessions; --",
        "\" OR 1=1 --",
        "Unicode: café, naïve, 日本語, العربية, 🚀🤖🔥",
        "Long text: " + ("x" * 20_000),
    ]

    async def _run() -> None:
        store = SQLiteSessionStore(db_path=db_file)
        try:
            s = await store.create_session(
                agent="alpha",
                metadata={"sqli": "'; DROP TABLE messages; --", "emoji": "🔥"},
            )
            msgs = [
                StoredMessage(role="user" if i % 2 == 0 else "assistant", content=txt)
                for i, txt in enumerate(tricky_strings)
            ]
            await store.append_messages(s.session_id, msgs)

            fetched = await store.get_session(s.session_id)
            assert fetched.metadata == {"sqli": "'; DROP TABLE messages; --", "emoji": "🔥"}
            assert [m.content for m in fetched.messages] == tricky_strings
        finally:
            await store.close()

    asyncio.run(_run())


def test_sqlite_operations_after_close_raise(tmp_path: Path) -> None:
    """Using a closed SQLiteSessionStore raises SessionStoreError."""
    async def _run() -> None:
        store = SQLiteSessionStore(db_path=tmp_path / "closed.db")
        await store.close()
        with pytest.raises(SessionStoreError, match="closed"):
            await store.create_session(agent="alpha")

    asyncio.run(_run())


def test_sqlite_concurrent_stores_with_busy_timeout(tmp_path: Path) -> None:
    """Two separate SQLiteSessionStore instances on the same WAL DB append safely."""
    db_file = tmp_path / "shared_wal.db"

    async def _run() -> None:
        store_a = SQLiteSessionStore(db_path=db_file, busy_timeout_ms=5000)
        store_b = SQLiteSessionStore(db_path=db_file, busy_timeout_ms=5000)
        try:
            s = await store_a.create_session(agent="alpha")

            async def _writer(store: SQLiteSessionStore, prefix: str) -> None:
                for i in range(10):
                    await store.append_messages(
                        s.session_id,
                        [StoredMessage(role="user", content=f"{prefix}-{i}")],
                    )

            await asyncio.gather(_writer(store_a, "A"), _writer(store_b, "B"))
            msgs = await store_a.get_messages(s.session_id)
            assert len(msgs) == 20
            assert [m.seq for m in msgs] == list(range(1, 21))
        finally:
            await store_a.close()
            await store_b.close()

    asyncio.run(_run())


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX file permission bits are not enforced on Windows",
)
def test_sqlite_new_file_permissions_owner_only_and_existing_unchanged(
    tmp_path: Path,
) -> None:
    """New DB files are created with 0o600; pre-existing files keep their permissions."""
    new_db = tmp_path / "new_secure.db"
    store_new = SQLiteSessionStore(db_path=new_db)
    asyncio.run(store_new.close())
    assert stat.S_IMODE(os.stat(new_db).st_mode) == 0o600

    existing_db = tmp_path / "existing.db"
    fd = os.open(existing_db, os.O_CREAT | os.O_RDWR, 0o644)
    os.close(fd)
    os.chmod(existing_db, 0o644)

    store_existing = SQLiteSessionStore(db_path=existing_db)
    asyncio.run(store_existing.close())
    assert stat.S_IMODE(os.stat(existing_db).st_mode) == 0o644
