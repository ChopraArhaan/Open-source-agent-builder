# Project: Open Source Agent Builder

A lightweight, transparent engine for building, running, and serving autonomous AI agents.
Stack: Python 3.10+, FastAPI, Pydantic v2, httpx, PyYAML. No LangChain, AutoGen, CrewAI, or similar frameworks.

## Principles

- **No black boxes.** Every prompt, state transition, and tool call must be loggable and inspectable.
- **Minimal dependencies.** Allowed: pydantic, fastapi, uvicorn, httpx, pyyaml, pytest, pytest-asyncio. Ask before adding anything else.
- **Async-first.** Sync tool functions are wrapped once at the registry, not special-cased elsewhere.
- **Resilient loop.** Tool errors become observations for the LLM. They never crash the loop.
- **Config over constants.** Model names, providers, and limits come from config, never hardcoded.

## Architecture (build order)

1. `tools/` ToolRegistry, `@register_tool`, JSON Schema generation
2. `llm/` BaseLLMProvider + provider implementations (native function calling first)
3. `engine/` ReAct loop, scratchpad, stop conditions, RunEvent stream
4. `config/` AgentConfig (YAML) + loader that scans `agents/`
5. `api/` FastAPI: `/run`, `/stream` (SSE), `/agents`, `/tools`
6. Observability, tests, docs

**Single code path:** the engine yields a stream of `RunEvent`s (`thought`, `tool_call`, `observation`, `final`, `error`). `/run` collects them; `/stream` forwards them.

## Layout

src/agent_builder/{models.py, tools/, llm/, engine/, config/, api/}
agents/ example YAML agents
tests/

## Workflow

- Work **only** on the phase I name. Do not build ahead.
- Before coding: give a plan in under 10 lines and list the files you will create or change.
- If a requirement is ambiguous, ask one concise question instead of guessing.
- Do not modify shared models in `models.py` without asking me first.

## Quality

- Type hints everywhere. Run `ruff` and `mypy` if configured.
- Write tests alongside code. A phase is done only when `pytest` passes.
- Use `FakeLLMProvider` for engine tests. No real network calls in tests.

## Git & GitHub

- After each phase passes `pytest`, commit with `phase-N: <summary>`.
- Push to a feature branch named `phase-N-<name>` and open a PR to `main`.
- Never push directly to `main`, never force-push, and never commit `.env` or secrets.

## End-of-phase report

When a phase is finished, summarize:

1. What was built
2. What is untested or incomplete
3. Open questions or risks
