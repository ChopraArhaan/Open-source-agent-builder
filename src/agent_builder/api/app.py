"""FastAPI application factory and route definitions for serving agents."""

import json
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
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
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        agents_dir: Directory path to scan for agent YAML configurations.
        registry: Tool registry to use for tool lookups. Defaults to default_registry.
        llm_override: Optional LLM provider override for zero-network testing.
        configs: Optional explicit dictionary of {name: AgentConfig} to bypass directory scanning.
    """
    app = FastAPI(
        title="Open Source Agent Builder API",
        version="0.1.0",
        description="A transparent engine for running and serving autonomous AI agents.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    effective_registry = registry if registry is not None else default_registry

    # Load configurations
    loaded_configs: dict[str, AgentConfig] = {}
    if configs is not None:
        loaded_configs.update(configs)
    elif agents_dir is not None:
        path = Path(agents_dir)
        if path.is_dir():
            loaded_configs.update(scan_agents_dir(path, strict=True))
    else:
        # Default fallback to ./agents if it exists
        default_dir = Path("agents")
        if default_dir.is_dir():
            loaded_configs.update(scan_agents_dir(default_dir, strict=False)[0])

    # Store state on app
    app.state.configs = loaded_configs
    app.state.registry = effective_registry
    app.state.llm_override = llm_override

    @app.get("/health", tags=["System"])
    async def health() -> dict[str, str]:
        """System health and ready check."""
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/agents", response_model=list[AgentSummary], tags=["Agents"])
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

    @app.get("/agents/{name}", response_model=AgentConfig, tags=["Agents"])
    async def get_agent(name: str) -> AgentConfig:
        """Get the full declarative specification for a given agent."""
        if name not in app.state.configs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{name}' not found",
            )
        return app.state.configs[name]

    @app.get("/tools", response_model=list[ToolSummary], tags=["Tools"])
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

    @app.post("/agents/{name}/run", response_model=RunResponse, tags=["Execution"])
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

    @app.post("/agents/{name}/stream", tags=["Execution"])
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

    return app


# Default ASGI application instance
app = create_app()
