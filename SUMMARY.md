# 📑 Architecture & Engineering Summary: Open Source Agent Builder

---

## 1. Executive Summary & Vision

### What is Open Source Agent Builder?
**Open Source Agent Builder** is a modular, transparent, and lightweight engine for building, running, and deploying autonomous AI agents from first principles using Python and FastAPI.

### Why Build This From Scratch?
Many current agent frameworks (e.g., LangChain, AutoGen, CrewAI, LlamaIndex) suffer from:
1. **Opaque Abstractions**: High levels of indirection make debugging prompt loops and execution state extremely painful.
2. **Dependency Bloat**: Pulling hundreds of transitive packages introduces security vulnerabilities and performance overhead.
3. **Brittle Production Guarantees**: Frameworks frequently make breaking changes across minor versions.
4. **Lack of Native API First Design**: Many frameworks focus on notebook demos rather than clean REST / SSE API services.

**Our Core Philosophy:**
- **No Black Boxes**: Every prompt, state transition, and tool call is visible, logged, and inspectable.
- **Pure Python & FastAPI**: Built on robust, standard Python libraries (`pydantic`, `httpx`, `fastapi`, `pyyaml`).
- **Declarative Configuration**: Define agents in readable YAML, separating agent behavior from execution runtime.
- **Production-Ready**: Native async support, streaming via Server-Sent Events (SSE), and clean error boundaries.

---

## 2. Core Architectural Components

### 2.1. Declarative Agent Specification (YAML)
Agents are defined declaratively in `.yaml` files. A Pydantic schema validates configurations on startup:
- **Identity & Persona**: Name, description, system prompt, role instructions.
- **LLM Settings**: Provider (`openai`, `anthropic`, `gemini`, `ollama`), model name, temperature, max tokens.
- **Tool Bindings**: Whitelist of registered tools the agent is permitted to execute.
- **Guardrails & Execution Policy**: Maximum iterations, execution timeout, fallback behavior on loop limits.

### 2.2. Pydantic-Powered Tool Registry
A registry that turns standard async/sync Python functions into agent tools:
- **Type Safety**: Parameters are defined using Pydantic `BaseModel` classes with descriptions.
- **Auto-Schema Generation**: Automatic translation into standard JSON Schema for LLM function calling.
- **Execution Wrapper**: Automatic error trapping (returning friendly error observations instead of crashing the loop).
- **Security Guardrails**: Parameter sanitation, allowed origin checks, and execution timeout limits.

### 2.3. Unified LLM Gateway
An abstract client layer (`BaseLLMProvider`) ensuring provider independence:
- Direct support for:
  - **OpenAI** (GPT-4o, GPT-4o-mini)
  - **Google Gemini** (Gemini 2.5/3.0 via google-genai / HTTP)
  - **Anthropic Claude** (Claude 3.5 Sonnet / Haiku)
  - **Local Ollama** (Llama 3, Mistral, Qwen)
- Dual Execution Paradigm:
  1. **Native Function Calling**: High precision, structured arguments directly from API.
  2. **Text ReAct Mode**: Traditional `Thought: ... \nAction: ... \nAction Input: ...` parsing for models lacking native tool call APIs.

### 2.4. Raw ReAct Runtime Engine
The core execution heartbeat implementing the **Reasoning + Acting** loop:
```
           ┌────────────────────────────────────────┐
           ▼                                        │
┌─────────────────────┐    Tool Needed?    ┌──────────────────┐
│ 1. LLM Step         │ ─────────────────► │ 2. Tool Registry │
│ Thought + Action    │                    │ Execute Tool     │
└──────────┬──────────┘                    └────────┬─────────┘
           │ No (Done)                              │
           │                                        │ Observation
           ▼                                        ▼
┌─────────────────────┐                    ┌──────────────────┐
│ 3. Final Answer     │                    │ 4. Append to     │
│ Return to Caller    │                    │    Scratchpad    │
└─────────────────────┘                    └──────────────────┘
```
- **Cycle Detection**: Detects repeated identical tool calls or oscillation loops.
- **Scratchpad Buffer**: Maintains step-by-step history of `Thought -> Action -> Observation`.
- **Termination Conditions**: Stop when `Final Answer` is reached, `max_iterations` exceeded, or timeout triggered.

### 2.5. FastAPI Serving Layer
Exposes the agent engine as a clean, production-grade microservice:
- `POST /api/v1/agents/{id}/run`: Synchronous execution returning the final result and trajectory trace.
- `POST /api/v1/agents/{id}/stream`: Real-time streaming via Server-Sent Events (SSE) delivering thoughts, tool calls, and observations live as they occur.
- `GET /api/v1/agents`: List all discovered agents and their active configs.
- `GET /api/v1/tools`: Inspect registered tools and their JSON schemas.

---

## 3. Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| **Language** | Python 3.10+ | Type hints, `asyncio`, ecosystem support |
| **Data Validation** | Pydantic v2 | High-performance schema validation & JSON Schema export |
| **API Framework** | FastAPI + Uvicorn | High performance, automatic OpenAPI docs, native async |
| **HTTP Client** | HTTPX | Fast, async HTTP requests for tool calls and API gateways |
| **Config Format** | PyYAML / YAML | Human-readable, declarative, portable |
| **Streaming** | SSE (Server-Sent Events) | Clean browser-friendly real-time event streaming |

---

## 4. Implementation Approaches: How We Can Build This

### Approach A: Bottom-Up (Core Engine First) — *Recommended*
1. **Tool Registry**: Build `ToolRegistry` and `@register_tool` decorator with Pydantic validation.
2. **LLM Provider**: Build a lightweight base LLM client (e.g. OpenAI/Gemini) that formats prompts and tool calls.
3. **ReAct Engine**: Build the core loop (`Engine.run`) that handles iterations, tool execution, and stop conditions.
4. **YAML Loader**: Add declarative configuration loading and validation.
5. **FastAPI Layer**: Wrap the engine with REST and SSE endpoints.

*Why this works best:* Ensures the execution mechanics are completely sound, unit-tested, and robust before adding API layers.

---

### Approach B: Schema & API-First (Top-Down)
1. Define the YAML agent specifications and OpenAPI route contracts first.
2. Build mock responses for `/run` and `/stream`.
3. Flesh out the execution engine underneath.

*Why consider this:* Great if building in collaboration with a frontend/UI team needing mock endpoints immediately.

---

### Approach C: Hybrid Dual-Loop (Text ReAct + Native Function Calling)
Support two operating modes in the engine:
1. **Mode 1 (Native Tool Calls)**: Fast, schema-strict, leverages modern provider APIs.
2. **Mode 2 (Prompt ReAct)**: Traditional text-based ReAct parsing for open-source / local models that struggle with native tool calling.

---

## 5. Phased Roadmap for Development

- [ ] **Phase 1: Foundation & Tool Registry**
  - Project setup (`pyproject.toml`, directory structure, `.env`).
  - `@register_tool` decorator, Pydantic argument validation, schema generator.
  - Built-in test tools (`calculator`, `mock_search`, `current_time`).

- [ ] **Phase 2: LLM Client Gateway**
  - Unified `BaseLLMClient` with support for tool schemas.
  - Implement provider clients (OpenAI-compatible, Google Gemini).

- [ ] **Phase 3: The ReAct Execution Engine**
  - Loop controller, trajectory/scratchpad tracking, iteration safety limits.
  - Error handling for tool failures with graceful recovery.

- [ ] **Phase 4: Declarative Agent Engine**
  - YAML schema definition (`AgentConfig`).
  - Agent discovery loader (scans `agents/` directory).

- [ ] **Phase 5: FastAPI Runtime & Streaming**
  - REST endpoints (`/run`, `/agents`, `/tools`).
  - SSE streaming endpoint (`/stream`) with real-time events.

- [ ] **Phase 6: Observability, Tests & Documentation**
  - Structured logs, trajectory exports, unit and integration tests.

---

*Note: This document is for internal architectural reference and planning.*
