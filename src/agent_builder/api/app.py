"""FastAPI application factory and route definitions for serving agents."""

import asyncio
import json
import math
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Annotated, cast

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.agent_builder.api.schemas import (
    AgentSummary,
    CreateSessionRequest,
    RunRequest,
    RunResponse,
    SessionResponse,
    SessionSummaryResponse,
    ToolSummary,
)
from src.agent_builder.config.loader import (
    build_engine_from_config,
    scan_agents_dir,
)
from src.agent_builder.config.schema import AgentConfig
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.memory import (
    SessionLimitError,
    SessionNotFoundError,
    SessionStore,
    SessionValidationError,
    SQLiteSessionStore,
    StoredMessage,
    select_history,
    to_llm_messages,
)
from src.agent_builder.memory.models import validate_session_id
from src.agent_builder.tools import default_registry
from src.agent_builder.tools.registry import ToolRegistry


class _SessionLockLease:
    """Class-based async context manager safe against GeneratorExit in SSE streams."""

    def __init__(self, manager: "_SessionLockManager", session_id: str) -> None:
        self._manager = manager
        self._session_id = session_id
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        self._lock = self._manager._checkout(self._session_id)
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            self._lock.release()
            self._manager._checkin(self._session_id)


class _SessionLockManager:
    """Reference-counted per-session asyncio lock manager that leaks no memory."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[asyncio.Lock, int]] = {}

    def _checkout(self, session_id: str) -> asyncio.Lock:
        lock, count = self._entries.get(session_id, (asyncio.Lock(), 0))
        self._entries[session_id] = (lock, count + 1)
        return lock

    def _checkin(self, session_id: str) -> None:
        entry = self._entries.get(session_id)
        if entry is None:
            return
        cur_lock, cur_count = entry
        remaining = cur_count - 1
        if remaining <= 0 and not cur_lock.locked():
            self._entries.pop(session_id, None)
        else:
            self._entries[session_id] = (cur_lock, remaining)

    def acquire(self, session_id: str) -> _SessionLockLease:
        return _SessionLockLease(self, session_id)

    @property
    def active_lock_count(self) -> int:
        return len(self._entries)


def get_session_store(request: Request) -> SessionStore:
    """FastAPI dependency returning the configured SessionStore."""
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        store = SQLiteSessionStore()
        request.app.state.session_store = store
    return cast(SessionStore, store)


def get_optional_session_store(request: Request) -> SessionStore | None:
    """FastAPI dependency returning SessionStore if initialized (for stateless-safe routes)."""
    return cast(SessionStore | None, getattr(request.app.state, "session_store", None))


def _parse_optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    val = int(raw)
    if val < 0:
        raise ValueError(f"{name} must be >= 0")
    return val


def _parse_optional_ttl_days(name: str = "AGENT_BUILDER_SESSION_TTL_DAYS") -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {name} value {raw!r}: expected a number of days"
        ) from exc
    if not math.isfinite(val):
        raise ValueError(
            f"Invalid {name} value {raw!r}: expected a finite number of days"
        )
    return val if val > 0 else None


def create_app(
    agents_dir: str | Path | None = None,
    registry: ToolRegistry | None = None,
    llm_override: BaseLLMProvider | None = None,
    configs: dict[str, AgentConfig] | None = None,
    api_key: str | None = None,
    cors_origins: list[str] | str | None = None,
    session_store: SessionStore | None = None,
    max_history_messages: int | None = None,
    max_history_chars: int | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        agents_dir: Directory path to scan for agent YAML configurations.
        registry: Tool registry to use for tool lookups. Defaults to default_registry.
        llm_override: Optional LLM provider override for zero-network testing.
        configs: Optional explicit dictionary of {name: AgentConfig} to bypass directory scanning.
        api_key: Optional server authentication API key. Defaults to AGENT_BUILDER_API_KEY env var.
        cors_origins: Allowed CORS origins list or comma-separated string. Defaults to CORS_ORIGINS env var.
        session_store: Optional SessionStore instance (e.g. InMemorySessionStore for tests).
        max_history_messages: Optional history window message count budget.
        max_history_chars: Optional history window character count budget.
    """
    owns_store = session_store is None

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
        if getattr(app_instance.state, "session_store", None) is None:
            app_instance.state.session_store = SQLiteSessionStore()

        ttl_days = _parse_optional_ttl_days("AGENT_BUILDER_SESSION_TTL_DAYS")
        if ttl_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            await app_instance.state.session_store.purge_older_than(cutoff)

        try:
            yield
        finally:
            if owns_store and getattr(app_instance.state, "session_store", None) is not None:
                await app_instance.state.session_store.close()

    app = FastAPI(
        title="Open Source Agent Builder API",
        version="0.1.0",
        description="A transparent engine for running and serving autonomous AI agents.",
        lifespan=lifespan,
    )

    # 1. Configurable CORS: Default to no allowed origins; never allow '*' with credentials
    effective_cors_origins: list[str] = []
    if cors_origins is not None:
        if isinstance(cors_origins, str):
            effective_cors_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
        else:
            effective_cors_origins = [o.strip() for o in cors_origins if o.strip()]
    elif os.getenv("CORS_ORIGINS"):
        raw_cors = os.getenv("CORS_ORIGINS", "")
        effective_cors_origins = [o.strip() for o in raw_cors.split(",") if o.strip()]

    if effective_cors_origins:
        # Never allow credentials when wildcard origin is present
        allow_creds = "*" not in effective_cors_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=effective_cors_origins,
            allow_credentials=allow_creds,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[],
            allow_credentials=False,
            allow_methods=[],
            allow_headers=[],
        )

    # 2. Optional API Key Authentication
    effective_api_key = api_key if api_key is not None else os.getenv("AGENT_BUILDER_API_KEY")

    async def verify_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> None:
        if not effective_api_key:
            return

        provided_key: str | None = None
        if x_api_key:
            provided_key = x_api_key
        elif authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()

        if not provided_key or not secrets.compare_digest(provided_key, effective_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    auth_dependencies = [Depends(verify_api_key)] if effective_api_key else []

    # 3. Registry, Configurations, Session Store, and Window Budgets
    effective_registry = registry if registry is not None else default_registry

    loaded_configs: dict[str, AgentConfig] = {}
    if configs is not None:
        loaded_configs.update(configs)
    else:
        resolved_dir = (
            Path(agents_dir)
            if agents_dir is not None
            else Path(os.getenv("AGENTS_DIR", "agents"))
        )
        if resolved_dir.is_dir():
            loaded_configs.update(scan_agents_dir(resolved_dir, strict=False)[0])

    effective_max_msgs = (
        max_history_messages
        if max_history_messages is not None
        else _parse_optional_int_env("AGENT_BUILDER_MAX_HISTORY_MESSAGES")
    )
    effective_max_chars = (
        max_history_chars
        if max_history_chars is not None
        else _parse_optional_int_env("AGENT_BUILDER_MAX_HISTORY_CHARS")
    )

    lock_manager = _SessionLockManager()

    # Store state on app
    app.state.configs = loaded_configs
    app.state.registry = effective_registry
    app.state.llm_override = llm_override
    app.state.api_key = effective_api_key
    app.state.session_store = session_store
    app.state.session_locks = lock_manager

    # 4. Unauthenticated System / Health Endpoints
    @app.get("/health", tags=["System"])
    @app.get("/api/v1/health", tags=["System"])
    async def health() -> dict[str, str]:
        """System health and ready check (always publicly accessible)."""
        return {"status": "ok", "version": "0.1.0"}

    # 5. Core API Endpoints (protected by auth if configured)
    api_router = APIRouter(dependencies=auth_dependencies)

    @api_router.get("/agents", response_model=list[AgentSummary], tags=["Agents"])
    async def list_agents() -> list[AgentSummary]:
        """List all discovered agent configurations."""
        return [
            AgentSummary(
                name=cfg.name,
                version=cfg.version,
                description=cfg.description,
                tools=cfg.tools,
                provider=cfg.llm.provider,
                model=cfg.llm.model,
            )
            for cfg in app.state.configs.values()
        ]

    @api_router.get("/agents/{name}", response_model=AgentConfig, tags=["Agents"])
    async def get_agent(name: str) -> AgentConfig:
        """Get the full declarative specification for a given agent."""
        if name not in app.state.configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )
        return app.state.configs[name]

    @api_router.get("/tools", response_model=list[ToolSummary], tags=["Tools"])
    async def list_tools() -> list[ToolSummary]:
        """List all registered tools with their specifications."""
        specs = app.state.registry.list_specs()
        return [
            ToolSummary(
                name=s.name,
                description=s.description,
                parameters=s.parameters,
                timeout_seconds=s.timeout_seconds,
            )
            for s in specs
        ]

    # 6. Session Management Endpoints
    @api_router.post(
        "/sessions",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Sessions"],
    )
    async def create_session(
        request: CreateSessionRequest,
        store: Annotated[SessionStore, Depends(get_session_store)],
    ) -> SessionResponse:
        """Create a new persisted conversation session bound to an agent."""
        if request.agent not in app.state.configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{request.agent}' not found",
            )
        try:
            session = await store.create_session(
                agent=request.agent,
                metadata=request.metadata,
            )
        except (SessionValidationError, SessionLimitError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        return SessionResponse(
            session_id=session.session_id,
            agent=session.agent,
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata=session.metadata,
            messages=session.messages,
        )

    @api_router.get(
        "/sessions",
        response_model=list[SessionSummaryResponse],
        tags=["Sessions"],
    )
    async def list_sessions(
        store: Annotated[SessionStore, Depends(get_session_store)],
        agent: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> list[SessionSummaryResponse]:
        """List persisted sessions ordered by updated_at descending."""
        sessions = await store.list_sessions(agent=agent, limit=limit, offset=offset)
        return [
            SessionSummaryResponse(
                session_id=s.session_id,
                agent=s.agent,
                created_at=s.created_at,
                updated_at=s.updated_at,
                metadata=s.metadata,
                message_count=len(s.messages),
            )
            for s in sessions
        ]

    @api_router.get(
        "/sessions/{session_id}",
        response_model=SessionResponse,
        tags=["Sessions"],
    )
    async def get_session(
        session_id: str,
        store: Annotated[SessionStore, Depends(get_session_store)],
    ) -> SessionResponse:
        """Retrieve a session and its ordered message history."""
        try:
            session = await store.get_session(session_id)
        except SessionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        return SessionResponse(
            session_id=session.session_id,
            agent=session.agent,
            created_at=session.created_at,
            updated_at=session.updated_at,
            metadata=session.metadata,
            messages=session.messages,
        )

    @api_router.delete(
        "/sessions/{session_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Sessions"],
    )
    async def delete_session(
        session_id: str,
        store: Annotated[SessionStore, Depends(get_session_store)],
    ) -> Response:
        """Delete a session and all of its messages."""
        try:
            await store.delete_session(session_id)
        except SessionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @api_router.post("/agents/{name}/run", response_model=RunResponse, tags=["Execution"])
    async def run_agent(
        name: str,
        request: RunRequest,
        raw_request: Request,
        store: Annotated[SessionStore | None, Depends(get_optional_session_store)],
    ) -> RunResponse:
        """Run an agent synchronously and collect the full event trace."""
        if name not in app.state.configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )

        config = app.state.configs[name]
        engine = build_engine_from_config(
            config,
            llm=app.state.llm_override,
            registry=app.state.registry,
        )

        # Stateless mode when session_id is omitted
        if request.session_id is None:
            final_answer, events = await engine.run(
                prompt=request.prompt,
                session_messages=request.session_messages,
            )
            return RunResponse(
                agent_name=name,
                final_answer=final_answer,
                events=events,
            )

        active_store = store if store is not None else get_session_store(raw_request)

        # Stateful session mode: serialize per-session runs
        try:
            validated_id = validate_session_id(request.session_id)
        except SessionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        async with lock_manager.acquire(validated_id):
            try:
                session = await active_store.get_session(validated_id)
            except SessionNotFoundError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc),
                ) from exc

            if session.agent != name:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=(
                        f"Session '{validated_id}' belongs to agent '{session.agent}', "
                        f"not '{name}'"
                    ),
                )

            windowed = select_history(
                session.messages,
                max_messages=effective_max_msgs,
                max_chars=effective_max_chars,
            )
            history_llm = to_llm_messages(windowed)
            if request.session_messages:
                history_llm.extend(request.session_messages)

            final_answer, events = await engine.run(
                prompt=request.prompt,
                session_messages=history_llm,
            )

            has_final = any(e.type == "final" for e in events)
            has_error = any(e.type == "error" for e in events)
            if has_final and not has_error:
                try:
                    await active_store.append_messages(
                        validated_id,
                        [
                            StoredMessage(role="user", content=request.prompt),
                            StoredMessage(role="assistant", content=final_answer),
                        ],
                    )
                except SessionLimitError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                        detail=str(exc),
                    ) from exc

            return RunResponse(
                agent_name=name,
                final_answer=final_answer,
                events=events,
                session_id=validated_id,
            )

    @api_router.post("/agents/{name}/stream", tags=["Execution"])
    async def stream_agent(
        name: str,
        request: RunRequest,
        raw_request: Request,
        store: Annotated[SessionStore | None, Depends(get_optional_session_store)],
    ) -> StreamingResponse:
        """Stream an agent's execution step-by-step using Server-Sent Events (SSE)."""
        if name not in app.state.configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )

        config = app.state.configs[name]
        engine = build_engine_from_config(
            config,
            llm=app.state.llm_override,
            registry=app.state.registry,
        )

        # Stateless streaming mode
        if request.session_id is None:
            async def stateless_sse_generator() -> AsyncGenerator[str, None]:
                async for event in engine.run_stream(
                    prompt=request.prompt,
                    session_messages=request.session_messages,
                ):
                    payload = event.model_dump(mode="json")
                    yield f"event: {event.type}\ndata: {json.dumps(payload)}\n\n"

            return StreamingResponse(
                stateless_sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        active_store = store if store is not None else get_session_store(raw_request)

        # Validate session upfront so HTTP 422/404/409 are returned before SSE headers
        try:
            validated_id = validate_session_id(request.session_id)
            initial_session = await active_store.get_session(validated_id)
        except SessionValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        if initial_session.agent != name:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Session '{validated_id}' belongs to agent '{initial_session.agent}', "
                    f"not '{name}'"
                ),
            )

        async def session_sse_generator() -> AsyncGenerator[str, None]:
            async with lock_manager.acquire(validated_id):
                session = await active_store.get_session(validated_id)
                windowed = select_history(
                    session.messages,
                    max_messages=effective_max_msgs,
                    max_chars=effective_max_chars,
                )
                history_llm = to_llm_messages(windowed)
                if request.session_messages:
                    history_llm.extend(request.session_messages)

                final_content: str | None = None
                has_error = False

                async for event in engine.run_stream(
                    prompt=request.prompt,
                    session_messages=history_llm,
                ):
                    if event.type == "error":
                        has_error = True
                    elif event.type == "final":
                        final_content = event.content or ""

                    payload = event.model_dump(mode="json")
                    if event.type == "final":
                        payload["session_id"] = validated_id
                    yield f"event: {event.type}\ndata: {json.dumps(payload)}\n\n"

                # Persist ONLY after the stream completes normally with a final event
                if final_content is not None and not has_error:
                    await active_store.append_messages(
                        validated_id,
                        [
                            StoredMessage(role="user", content=request.prompt),
                            StoredMessage(role="assistant", content=final_content),
                        ],
                    )

        return StreamingResponse(
            session_sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Mount routes both under /api/v1 and at root / for backwards compatibility
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(api_router)

    return app


# Default ASGI application instance
app = create_app()
