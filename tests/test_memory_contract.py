"""Shared contract test suite parametrized across all SessionStore implementations."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.agent_builder.memory import (
    InMemorySessionStore,
    SessionLimitError,
    SessionNotFoundError,
    SessionStore,
    SessionValidationError,
    SQLiteSessionStore,
    StoredMessage,
)


class FakeClock:
    """Deterministic injectable clock for testing timestamps and retention."""

    def __init__(self, initial: datetime | None = None) -> None:
        self.now = initial or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int = 1) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


def _make_in_memory(
    tmp_path: Path,
    clock: Callable[[], datetime],
    max_message_chars: int = 100_000,
    max_messages_per_session: int = 1_000,
    max_metadata_bytes: int = 16_384,
) -> SessionStore:
    return InMemorySessionStore(
        max_message_chars=max_message_chars,
        max_messages_per_session=max_messages_per_session,
        max_metadata_bytes=max_metadata_bytes,
        clock=clock,
    )


def _make_sqlite_memory(
    tmp_path: Path,
    clock: Callable[[], datetime],
    max_message_chars: int = 100_000,
    max_messages_per_session: int = 1_000,
    max_metadata_bytes: int = 16_384,
) -> SessionStore:
    return SQLiteSessionStore(
        db_path=":memory:",
        max_message_chars=max_message_chars,
        max_messages_per_session=max_messages_per_session,
        max_metadata_bytes=max_metadata_bytes,
        clock=clock,
    )


def _make_sqlite_file(
    tmp_path: Path,
    clock: Callable[[], datetime],
    max_message_chars: int = 100_000,
    max_messages_per_session: int = 1_000,
    max_metadata_bytes: int = 16_384,
) -> SessionStore:
    return SQLiteSessionStore(
        db_path=tmp_path / "contract_sessions.db",
        max_message_chars=max_message_chars,
        max_messages_per_session=max_messages_per_session,
        max_metadata_bytes=max_metadata_bytes,
        clock=clock,
    )


STORE_FACTORIES = [
    pytest.param(_make_in_memory, id="in_memory"),
    pytest.param(_make_sqlite_memory, id="sqlite_memory"),
    pytest.param(_make_sqlite_file, id="sqlite_file"),
]


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_create_get_delete_session(factory: Callable[..., SessionStore], tmp_path: Path) -> None:
    """Create, retrieve, and delete a session, verifying SessionNotFoundError afterward."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            created = await store.create_session(
                agent="math-tutor",
                metadata={"env": "test", "count": 42, "nested": {"ok": True}},
            )
            assert created.agent == "math-tutor"
            assert created.metadata == {"env": "test", "count": 42, "nested": {"ok": True}}
            assert created.messages == []
            assert created.created_at == clock.now

            fetched = await store.get_session(created.session_id)
            assert fetched.session_id == created.session_id
            assert fetched.agent == "math-tutor"
            assert fetched.metadata == created.metadata

            await store.delete_session(created.session_id)
            with pytest.raises(SessionNotFoundError):
                await store.get_session(created.session_id)
            with pytest.raises(SessionNotFoundError):
                await store.delete_session(created.session_id)
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_invalid_ids_and_missing_session(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """Invalid UUIDv4 IDs raise SessionValidationError; unknown UUIDv4 raises SessionNotFoundError."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            for bad_id in ["", "not-a-uuid", "12345", str(uuid.uuid1())]:
                with pytest.raises(SessionValidationError):
                    await store.get_session(bad_id)
                with pytest.raises(SessionValidationError):
                    await store.delete_session(bad_id)
                with pytest.raises(SessionValidationError):
                    await store.get_messages(bad_id)
                with pytest.raises(SessionValidationError):
                    await store.append_messages(bad_id, [StoredMessage(role="user", content="hi")])

            unknown_id = str(uuid.uuid4())
            with pytest.raises(SessionNotFoundError):
                await store.get_session(unknown_id)
            with pytest.raises(SessionNotFoundError):
                await store.get_messages(unknown_id)
            with pytest.raises(SessionNotFoundError):
                await store.append_messages(unknown_id, [StoredMessage(role="user", content="hi")])
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_append_ordering_and_limit(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """Appended messages receive contiguous sequence numbers and respect get_messages(limit)."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            s = await store.create_session(agent="math-tutor")
            clock.advance(5)
            batch1 = await store.append_messages(
                s.session_id,
                [
                    StoredMessage(role="user", content="Q1"),
                    StoredMessage(role="assistant", content="A1"),
                ],
            )
            assert [m.seq for m in batch1] == [1, 2]

            clock.advance(5)
            batch2 = await store.append_messages(
                s.session_id,
                [
                    StoredMessage(role="user", content="Q2"),
                    StoredMessage(role="assistant", content="A2"),
                ],
            )
            assert [m.seq for m in batch2] == [3, 4]

            all_msgs = await store.get_messages(s.session_id)
            assert [m.content for m in all_msgs] == ["Q1", "A1", "Q2", "A2"]
            assert [m.seq for m in all_msgs] == [1, 2, 3, 4]

            recent_2 = await store.get_messages(s.session_id, limit=2)
            assert [m.content for m in recent_2] == ["Q2", "A2"]
            assert [m.seq for m in recent_2] == [3, 4]

            empty = await store.get_messages(s.session_id, limit=0)
            assert empty == []
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_atomic_append_failure_writes_nothing(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """If a batch fails validation on the second message, nothing in the batch is persisted."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock, max_message_chars=10)
        try:
            s = await store.create_session(agent="math-tutor")
            with pytest.raises(SessionLimitError):
                await store.append_messages(
                    s.session_id,
                    [
                        StoredMessage(role="user", content="ok"),
                        StoredMessage(role="assistant", content="this message is way too long"),
                    ],
                )
            msgs = await store.get_messages(s.session_id)
            assert msgs == []
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_list_sessions_filtering_and_pagination(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """list_sessions filters by agent, orders by updated_at desc, and paginates."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            s1 = await store.create_session(agent="alpha")
            clock.advance(10)
            s2 = await store.create_session(agent="beta")
            clock.advance(10)
            s3 = await store.create_session(agent="alpha")

            # Touch s1 so it becomes the most recently updated
            clock.advance(10)
            await store.append_messages(s1.session_id, [StoredMessage(role="user", content="hi")])

            all_sessions = await store.list_sessions()
            assert [s.session_id for s in all_sessions] == [s1.session_id, s3.session_id, s2.session_id]

            alpha_only = await store.list_sessions(agent="alpha")
            assert [s.session_id for s in alpha_only] == [s1.session_id, s3.session_id]

            page1 = await store.list_sessions(limit=2, offset=0)
            page2 = await store.list_sessions(limit=2, offset=2)
            assert [s.session_id for s in page1] == [s1.session_id, s3.session_id]
            assert [s.session_id for s in page2] == [s2.session_id]
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_purge_older_than_with_injectable_clock(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """purge_older_than deletes only sessions strictly older than cutoff."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            old_session = await store.create_session(agent="alpha")
            clock.advance(100)
            new_session = await store.create_session(agent="alpha")

            cutoff = clock.now - timedelta(seconds=50)
            deleted = await store.purge_older_than(cutoff)
            assert deleted == 1

            with pytest.raises(SessionNotFoundError):
                await store.get_session(old_session.session_id)
            kept = await store.get_session(new_session.session_id)
            assert kept.session_id == new_session.session_id
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_size_limits_and_validation(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """Enforce max_messages_per_session and max_metadata_bytes."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock, max_messages_per_session=2, max_metadata_bytes=64)
        try:
            with pytest.raises(SessionLimitError):
                await store.create_session(agent="alpha", metadata={"big": "x" * 100})

            with pytest.raises(SessionValidationError):
                await store.create_session(agent="alpha", metadata={"bad": float("nan")})

            s = await store.create_session(agent="alpha")
            await store.append_messages(
                s.session_id,
                [
                    StoredMessage(role="user", content="1"),
                    StoredMessage(role="assistant", content="2"),
                ],
            )
            with pytest.raises(SessionLimitError):
                await store.append_messages(s.session_id, [StoredMessage(role="user", content="3")])
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_validation_edge_cases(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """Test duplicate session_id, empty agent, invalid metadata type, empty append, bad pagination."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            with pytest.raises(SessionValidationError):
                await store.create_session(agent="")
            with pytest.raises(SessionValidationError):
                await store.create_session(agent="alpha", metadata=["not-a-dict"])  # type: ignore[arg-type]

            sid = str(uuid.uuid4())
            await store.create_session(agent="alpha", session_id=sid)
            with pytest.raises(SessionValidationError):
                await store.create_session(agent="alpha", session_id=sid)

            # Empty append returns [] and leaves session valid
            assert await store.append_messages(sid, []) == []

            with pytest.raises(SessionValidationError):
                await store.list_sessions(limit=0)
            with pytest.raises(SessionValidationError):
                await store.list_sessions(offset=-1)
            with pytest.raises(SessionValidationError):
                await store.get_messages(sid, limit=-1)
        finally:
            await store.close()

    asyncio.run(_run())


@pytest.mark.parametrize("factory", STORE_FACTORIES)
def test_contract_concurrent_appends_unique_contiguous_seq(
    factory: Callable[..., SessionStore], tmp_path: Path
) -> None:
    """N concurrent appends to the same session must yield contiguous unique sequence numbers."""
    async def _run() -> None:
        clock = FakeClock()
        store = factory(tmp_path, clock)
        try:
            s = await store.create_session(agent="alpha")

            async def _append_one(idx: int) -> list[StoredMessage]:
                return await store.append_messages(
                    s.session_id,
                    [
                        StoredMessage(role="user", content=f"u-{idx}"),
                        StoredMessage(role="assistant", content=f"a-{idx}"),
                    ],
                )

            results = await asyncio.gather(*[_append_one(i) for i in range(20)])
            assert len(results) == 20

            msgs = await store.get_messages(s.session_id)
            assert len(msgs) == 40
            assert [m.seq for m in msgs] == list(range(1, 41))
        finally:
            await store.close()

    asyncio.run(_run())
