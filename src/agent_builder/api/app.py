"""FastAPI application factory and route definitions for serving agents."""

import json
import os
import secrets
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.agent_builder.api.schemas import (
    AgentSummary,
    RunRequest,
    RunResponse,
    ToolSummary,
)
from src.agent_builder.config.loader import (
    build_engine_from_config,
    scan_agents_dir,
)
from src.agent_builder.config.schema import AgentConfig
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.tools import default_registry
from src.agent_builder.tools.registry import ToolRegistry


def create_app(
    agents_dir: str | Path | None = None,
    registry: ToolRegistry | None = None,
    llm_override: BaseLLMProvider | None = None,
    configs: dict[str, AgentConfig] | None = None,
    api_key: str | None = None,
    cors_origins: list[str] | str | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        agents_dir: Directory path to scan for agent YAML configurations.
        registry: Tool registry to use for tool lookups. Defaults to default_registry.
        llm_override: Optional LLM provider override for zero-network testing.
        configs: Optional explicit dictionary of {name: AgentConfig} to bypass directory scanning.
        api_key: Optional server authentication API key. Defaults to AGENT_BUILDER_API_KEY env var.
        cors_origins: Allowed CORS origins list or comma-separated string. Defaults to CORS_ORIGINS env var.
    """
    app = FastAPI(
        title="Open Source Agent Builder API",
        version="0.1.0",
        description="A transparent engine for running and serving autonomous AI agents.",
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

    # 3. Registry & Configurations
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

    # Store state on app
    app.state.configs = loaded_configs
    app.state.registry = effective_registry
    app.state.llm_override = llm_override
    app.state.api_key = effective_api_key

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

    @api_router.post("/agents/{name}/run", response_model=RunResponse, tags=["Execution"])
    async def run_agent(name: str, request: RunRequest) -> RunResponse:
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

        final_answer, events = await engine.run(
            prompt=request.prompt,
            session_messages=request.session_messages,
        )

        return RunResponse(
            agent_name=name,
            final_answer=final_answer,
            events=events,
        )

    @api_router.post("/agents/{name}/stream", tags=["Execution"])
    async def stream_agent(name: str, request: RunRequest) -> StreamingResponse:
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

        async def sse_generator() -> AsyncGenerator[str, None]:
            async for event in engine.run_stream(
                prompt=request.prompt,
                session_messages=request.session_messages,
            ):
                payload = event.model_dump(mode="json")
                yield f"event: {event.type}\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            sse_generator(),
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
