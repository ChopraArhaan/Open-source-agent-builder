"""API integration, concurrency, and crash/cancel-safety tests for sessions."""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from src.agent_builder.api.app import create_app
from src.agent_builder.config.schema import AgentConfig, ExecutionConfig, LLMConfig
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.memory import InMemorySessionStore, SQLiteSessionStore
from src.agent_builder.models import LLMMessage, LLMResponse, ToolCall, ToolSpec
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


def _make_app_and_client(
    responses: list[Any] | None = None,
    llm_override: BaseLLMProvider | None = None,
    store: InMemorySessionStore | None = None,
    max_history_messages: int | None = None,
    max_history_chars: int | None = None,
    api_key: str | None = None,
) -> tuple[Any, TestClient, FakeLLMProvider | BaseLLMProvider, InMemorySessionStore]:
    registry = ToolRegistry()
    register_builtin_tools(registry)

    configs = {
        "math-agent": AgentConfig(
            name="math-agent",
            version="1.0.0",
            description="Math tutor",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="You are a math tutor.",
            tools=["calculator"],
            execution=ExecutionConfig(max_iterations=5),
        ),
        "other-agent": AgentConfig(
            name="other-agent",
            version="1.0.0",
            description="Other agent",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="You are another agent.",
            tools=[],
        ),
    }

    effective_llm = llm_override or FakeLLMProvider(responses=responses or [])
    effective_store = store or InMemorySessionStore()

    app = create_app(
        registry=registry,
        llm_override=effective_llm,
        configs=configs,
        session_store=effective_store,
        max_history_messages=max_history_messages,
        max_history_chars=max_history_chars,
        api_key=api_key,
    )
    client = TestClient(app)
    return app, client, effective_llm, effective_store


def test_stateless_run_unchanged() -> None:
    """Requests without session_id remain stateless and persist nothing."""
    _, client, _, store = _make_app_and_client(responses=["42"])
    resp = client.post("/agents/math-agent/run", json={"prompt": "Answer?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_answer"] == "42"
    assert data["session_id"] is None
    assert asyncio.run(store.list_sessions()) == []


def test_session_crud_routes_and_pagination() -> None:
    """Verify POST, GET (single & list), and DELETE /sessions routes."""
    _, client, _, _ = _make_app_and_client()

    # Unknown agent -> 404
    bad_create = client.post("/sessions", json={"agent": "nonexistent"})
    assert bad_create.status_code == 404

    # Create 2 sessions
    r1 = client.post("/sessions", json={"agent": "math-agent", "metadata": {"tag": "a"}})
    assert r1.status_code == 201
    s1 = r1.json()
    assert s1["agent"] == "math-agent"
    assert s1["metadata"] == {"tag": "a"}
    assert s1["messages"] == []

    r2 = client.post("/sessions", json={"agent": "other-agent"})
    assert r2.status_code == 201
    s2 = r2.json()

    # List all and filter by agent
    all_list = client.get("/sessions").json()
    assert len(all_list) == 2
    math_list = client.get("/sessions", params={"agent": "math-agent"}).json()
    assert len(math_list) == 1
    assert math_list[0]["session_id"] == s1["session_id"]

    # Pagination
    paged = client.get("/sessions", params={"limit": 1, "offset": 1}).json()
    assert len(paged) == 1

    # Get single
    get_r = client.get(f"/sessions/{s1['session_id']}")
    assert get_r.status_code == 200
    assert get_r.json()["session_id"] == s1["session_id"]

    # Invalid ID -> 422, missing ID -> 404
    assert client.get("/sessions/not-a-uuid").status_code == 422
    assert client.get(f"/sessions/{uuid.uuid4()}").status_code == 404

    # Delete
    del_r = client.delete(f"/sessions/{s2['session_id']}")
    assert del_r.status_code == 204
    assert client.get(f"/sessions/{s2['session_id']}").status_code == 404
    assert client.delete(f"/sessions/{s2['session_id']}").status_code == 404


def test_multi_turn_run_forwards_history_and_excludes_scratchpad() -> None:
    """Turn 2 LLM call receives Turn 1 user+assistant messages, without tool scratchpad."""
    tool_call = ToolCall(id="call_1", name="calculator", arguments={"expression": "6 * 7"})
    _, client, fake_llm, _ = _make_app_and_client(
        responses=[
            [tool_call],
            "6 * 7 is 42.",
            "Earlier you asked about 6 * 7, which is 42.",
        ]
    )
    assert isinstance(fake_llm, FakeLLMProvider)

    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]

    # Turn 1 (uses tool call -> scratchpad must NOT be persisted)
    r1 = client.post(
        "/agents/math-agent/run",
        json={"prompt": "What is 6 * 7?", "session_id": sid},
    )
    assert r1.status_code == 200
    assert r1.json()["final_answer"] == "6 * 7 is 42."
    assert r1.json()["session_id"] == sid

    # Verify stored session has only 2 messages (user + final assistant)
    sess = client.get(f"/sessions/{sid}").json()
    assert [(m["role"], m["content"]) for m in sess["messages"]] == [
        ("user", "What is 6 * 7?"),
        ("assistant", "6 * 7 is 42."),
    ]

    # Turn 2
    r2 = client.post(
        "/agents/math-agent/run",
        json={"prompt": "What did I just ask?", "session_id": sid},
    )
    assert r2.status_code == 200

    # Assert on FakeLLMProvider's recorded call for Turn 2
    last_call = fake_llm.last_call()
    assert last_call is not None
    sent_messages = last_call["messages"]
    assert [(m.role, m.content) for m in sent_messages] == [
        ("system", "You are a math tutor."),
        ("user", "What is 6 * 7?"),
        ("assistant", "6 * 7 is 42."),
        ("user", "What did I just ask?"),
    ]


def test_windowed_history_sent_to_engine_while_full_history_persisted() -> None:
    """Engine receives only windowed history, while store retains all turns."""
    _, client, fake_llm, _ = _make_app_and_client(
        responses=["A1", "A2", "A3"],
        max_history_messages=2,  # 1 pair (2 messages)
    )
    assert isinstance(fake_llm, FakeLLMProvider)
    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]

    client.post("/agents/math-agent/run", json={"prompt": "Q1", "session_id": sid})
    client.post("/agents/math-agent/run", json={"prompt": "Q2", "session_id": sid})
    client.post("/agents/math-agent/run", json={"prompt": "Q3", "session_id": sid})

    # Full history in store has all 6 messages
    sess = client.get(f"/sessions/{sid}").json()
    assert len(sess["messages"]) == 6

    # Turn 3 LLM call received only Turn 2 (Q2, A2) + current Q3
    last_call = fake_llm.last_call()
    assert last_call is not None
    assert [(m.role, m.content) for m in last_call["messages"]] == [
        ("system", "You are a math tutor."),
        ("user", "Q2"),
        ("assistant", "A2"),
        ("user", "Q3"),
    ]


def test_run_unknown_session_404_and_wrong_agent_409() -> None:
    """Unknown session returns 404; mismatched agent returns 409."""
    _, client, _, _ = _make_app_and_client(responses=["ok"])

    r_404 = client.post(
        "/agents/math-agent/run",
        json={"prompt": "Hi", "session_id": str(uuid.uuid4())},
    )
    assert r_404.status_code == 404

    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]
    r_409 = client.post(
        "/agents/other-agent/run",
        json={"prompt": "Hi", "session_id": sid},
    )
    assert r_409.status_code == 409


def test_error_ending_run_persists_nothing() -> None:
    """When an agent run ends in an error, nothing is persisted to the session."""
    _, client, _, _ = _make_app_and_client(
        responses=[RuntimeError("Simulated LLM failure")]
    )
    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]

    resp = client.post(
        "/agents/math-agent/run",
        json={"prompt": "This will fail", "session_id": sid},
    )
    assert resp.status_code == 200
    assert "Error:" in resp.json()["final_answer"]

    sess = client.get(f"/sessions/{sid}").json()
    assert sess["messages"] == []


def test_sse_stream_success_persists_and_includes_session_id() -> None:
    """Successful SSE stream persists turn and includes session_id on final event."""
    _, client, _, _ = _make_app_and_client(responses=["Streamed answer"])
    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]

    with client.stream(
        "POST",
        "/agents/math-agent/stream",
        json={"prompt": "Stream this", "session_id": sid},
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "event: final" in body
    assert sid in body

    sess = client.get(f"/sessions/{sid}").json()
    assert [(m["role"], m["content"]) for m in sess["messages"]] == [
        ("user", "Stream this"),
        ("assistant", "Streamed answer"),
    ]


def test_sse_stream_error_and_midstream_cancel_persist_nothing() -> None:
    """Error stream or mid-stream disconnect/cancel persists nothing."""
    async def _run() -> None:
        # 1. Error stream
        app_err, _, _, store_err = _make_app_and_client(
            responses=[RuntimeError("Stream crash")]
        )
        s_err = await store_err.create_session(agent="math-agent")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_err),
            base_url="http://test",
        ) as ac:
            r = await ac.post(
                "/agents/math-agent/stream",
                json={"prompt": "Fail stream", "session_id": s_err.session_id},
            )
            assert "event: error" in r.text
        assert await store_err.get_messages(s_err.session_id) == []

        # 2. Mid-stream disconnect via ASGI send aborting after the first SSE chunk
        class _TwoStepSlowProvider(BaseLLMProvider):
            def __init__(self) -> None:
                super().__init__(model="two-step-slow")
                self.turn = 0

            @property
            def provider_name(self) -> str:
                return "two-step-slow"

            async def complete(
                self,
                messages: list[LLMMessage],
                tools: list[ToolSpec] | None = None,
                **kwargs: Any,
            ) -> LLMResponse:
                self.turn += 1
                if self.turn == 1:
                    return LLMResponse(
                        content="Using calculator",
                        tool_calls=[ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})],
                        finish_reason="tool_calls",
                    )
                await asyncio.sleep(0.05)
                return LLMResponse(content="Should not persist", finish_reason="stop")

        app_disc, _, _, store_disc = _make_app_and_client(llm_override=_TwoStepSlowProvider())
        s_disc = await store_disc.create_session(agent="math-agent")

        import json as _json

        body_bytes = _json.dumps(
            {"prompt": "Disconnect early", "session_id": s_disc.session_id}
        ).encode("utf-8")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/agents/math-agent/stream",
            "raw_path": b"/agents/math-agent/stream",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body_bytes)).encode("ascii")),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "app": app_disc,
        }
        req_sent = False

        async def _receive() -> dict[str, Any]:
            nonlocal req_sent
            if not req_sent:
                req_sent = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return {"type": "http.disconnect"}

        body_chunks = 0

        async def _send(message: dict[str, Any]) -> None:
            nonlocal body_chunks
            if message["type"] == "http.response.body":
                body_chunks += 1
                if body_chunks >= 1:
                    raise asyncio.CancelledError("Simulated client disconnect")

        try:
            await app_disc(scope, _receive, _send)
        except asyncio.CancelledError:
            pass

        assert body_chunks == 1
        assert await store_disc.get_messages(s_disc.session_id) == []
        import gc

        gc.collect()
        await asyncio.sleep(0.01)
        assert app_disc.state.session_locks.active_lock_count == 0

    asyncio.run(_run())


class _OrderedSlowProvider(BaseLLMProvider):
    """Test provider that delays slightly and echoes the prompt + history count."""

    def __init__(self, delay: float = 0.02) -> None:
        super().__init__(model="slow-fake")
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def provider_name(self) -> str:
        return "slow-fake"

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            user_prompt = messages[-1].content or ""
            prior_count = len([m for m in messages if m.role in ("user", "assistant")]) - 1
            return LLMResponse(
                content=f"reply:{user_prompt}:prior={prior_count}",
                finish_reason="stop",
            )
        finally:
            self.in_flight -= 1


def test_concurrent_same_session_serialized_and_locks_cleaned() -> None:
    """Concurrent runs on the same session are serialized with no lost/interleaved turns."""
    async def _run() -> None:
        provider = _OrderedSlowProvider(delay=0.01)
        app, _, _, store = _make_app_and_client(llm_override=provider)
        session = await store.create_session(agent="math-agent")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            tasks = [
                ac.post(
                    "/agents/math-agent/run",
                    json={"prompt": f"msg-{i}", "session_id": session.session_id},
                )
                for i in range(6)
            ]
            responses = await asyncio.gather(*tasks)

        assert all(r.status_code == 200 for r in responses)
        # Because runs on the same session were serialized, max_in_flight must be 1
        assert provider.max_in_flight == 1

        msgs = await store.get_messages(session.session_id)
        assert len(msgs) == 12
        assert [m.seq for m in msgs] == list(range(1, 13))
        # Each user message is immediately followed by its matching reply
        for pair_idx in range(6):
            u_msg = msgs[pair_idx * 2]
            a_msg = msgs[pair_idx * 2 + 1]
            assert u_msg.role == "user"
            assert a_msg.role == "assistant"
            assert a_msg.content == f"reply:{u_msg.content}:prior={pair_idx * 2}"

        # Lock dictionary must be completely cleaned up
        assert app.state.session_locks.active_lock_count == 0

    asyncio.run(_run())


def test_concurrent_different_sessions_run_in_parallel() -> None:
    """Concurrent runs on different sessions execute in parallel without blocking each other."""
    async def _run() -> None:
        provider = _OrderedSlowProvider(delay=0.05)
        app, _, _, store = _make_app_and_client(llm_override=provider)
        sessions = [await store.create_session(agent="math-agent") for _ in range(4)]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            await asyncio.gather(
                *[
                    ac.post(
                        "/agents/math-agent/run",
                        json={"prompt": "hi", "session_id": s.session_id},
                    )
                    for s in sessions
                ]
            )

        assert provider.max_in_flight > 1
        assert app.state.session_locks.active_lock_count == 0

    asyncio.run(_run())


def test_session_routes_protected_by_api_key() -> None:
    """When AGENT_BUILDER_API_KEY is set, /sessions routes require authentication."""
    _, client, _, _ = _make_app_and_client(api_key="secret-token")
    assert client.get("/sessions").status_code == 401
    assert client.post("/sessions", json={"agent": "math-agent"}).status_code == 401
    assert client.get("/sessions", headers={"X-API-Key": "secret-token"}).status_code == 200


def test_lifespan_sqlite_and_ttl_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """App lifespan initializes SQLiteSessionStore, purges expired sessions, and closes cleanly."""
    db_file = tmp_path / "lifespan.db"
    monkeypatch.setenv("AGENT_BUILDER_DB_PATH", str(db_file))
    monkeypatch.setenv("AGENT_BUILDER_SESSION_TTL_DAYS", "1")

    # Seed an expired session before app startup
    async def _seed() -> str:
        s_store = SQLiteSessionStore(db_path=db_file)
        sess = await s_store.create_session(agent="math-agent")
        old_iso = (sess.updated_at - timedelta(days=5)).isoformat(timespec="microseconds")
        await s_store._run(
            lambda: s_store._require_conn().execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (old_iso, sess.session_id),
            )
        )
        await s_store.close()
        return sess.session_id

    expired_id = asyncio.run(_seed())

    registry = ToolRegistry()
    register_builtin_tools(registry)
    configs = {
        "math-agent": AgentConfig(
            name="math-agent",
            version="1.0.0",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="Math",
            tools=[],
        )
    }
    app = create_app(registry=registry, llm_override=FakeLLMProvider(responses=["ok"]), configs=configs)
    with TestClient(app) as client:
        # Expired session was purged at startup
        assert client.get(f"/sessions/{expired_id}").status_code == 404
        created = client.post("/sessions", json={"agent": "math-agent"}).json()
        r = client.post(
            "/agents/math-agent/run",
            json={"prompt": "Hello", "session_id": created["session_id"]},
        )
        assert r.status_code == 200


def test_session_message_size_limit_returns_422() -> None:
    """Exceeding max_message_chars during /run returns HTTP 422."""
    store = InMemorySessionStore(max_message_chars=10)
    _, client, _, _ = _make_app_and_client(responses=["ok"], store=store)
    sid = client.post("/sessions", json={"agent": "math-agent"}).json()["session_id"]

    resp = client.post(
        "/agents/math-agent/run",
        json={"prompt": "This prompt exceeds ten characters", "session_id": sid},
    )
    assert resp.status_code == 422
