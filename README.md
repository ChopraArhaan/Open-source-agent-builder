# 🤖 Open Source Agent Builder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.0+-red.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Build, run, and deploy autonomous AI agents from scratch.**
> Built cleanly with pure Python and FastAPI — featuring a raw ReAct execution loop, dynamic Pydantic tool registry, declarative YAML configs, and non-intrusive observability. Zero framework bloat, maximum transparency.

---

## 💡 Overview

Many contemporary AI agent frameworks hide core mechanics behind thick abstractions, complex inheritance trees, and brittle dependencies. **Open Source Agent Builder** is built from first principles to demystify agentic systems and deliver a clean, production-grade API service for custom agents.

It provides everything you need to build, test, trace, and serve autonomous agents without relying on heavyweight orchestration libraries.

---

## ✨ Key Features

- **⚡ Zero Framework Bloat**: Direct LLM calls and explicit state handling. No LangChain or LlamaIndex black boxes.
- **🔄 Raw ReAct Execution Loop**: Fully observable **Thought ➔ Action ➔ Observation** runtime loop with configurable iteration limits, loop detection, and early exits.
- **🛠️ Pydantic-Powered Tool Registry**: Declare tools using type-safe Pydantic models. Automatically generates OpenAI function-calling schemas with timeout and exception guardrails.
- **📄 Declarative Agent YAMLs**: Define personas, instructions, tools, system prompts, and model hyper-parameters in human-readable YAML specs with strict validation (`extra="forbid"`).
- **🚀 Production-Ready FastAPI Runtime**: Expose your agents instantly via REST endpoints and real-time Server-Sent Events (SSE) streaming (`POST /agents/{name}/stream`).
- **🔌 Multi-Provider LLM Gateway**: Standardized client layer supporting OpenAI-compatible APIs (OpenAI, Ollama, Groq, vLLM) and a deterministic `FakeLLMProvider` for zero-network testing.
- **🔍 Non-Intrusive Observability**: Execution tracing (`ExecutionTracer`), stream wrapper (`trace_stream`), JSON export, and structured logging with automated secret masking and string truncation.

---

## 📂 Project Structure

```text
open-source-agent-builder/
├── agents/                       # Declarative agent configurations (YAML)
│   ├── math_tutor.yaml
│   └── researcher.yaml
├── src/agent_builder/
│   ├── models.py                 # Core domain models (ToolSpec, ToolResult, RunEvent)
│   ├── tools/                    # Tool registry and built-in tools
│   │   ├── registry.py           # ToolRegistry & @register_tool decorator
│   │   └── builtins.py           # Safe AST calculator, current_time, mock_search
│   ├── llm/                      # Unified LLM provider interface
│   │   ├── base.py               # BaseLLMProvider abstract interface
│   │   ├── openai.py             # OpenAI-compatible httpx provider
│   │   ├── anthropic_provider.py # Native Anthropic Messages API provider
│   │   ├── gemini_provider.py    # Native Google Gemini provider
│   │   └── fake.py               # Scriptable offline test provider
│   ├── memory/                   # Persistent session stores & history windowing
│   │   ├── models.py             # Session, StoredMessage & error hierarchy
│   │   ├── store.py              # SessionStore ABC & InMemorySessionStore
│   │   ├── sqlite_store.py       # SQLiteSessionStore (stdlib sqlite3, WAL)
│   │   └── window.py             # Pair-safe history windowing (select_history)
│   ├── engine/                   # ReAct execution core
│   │   └── react.py              # Raw ReAct execution loop & RunEvent stream
│   ├── config/                   # Declarative configuration system
│   │   ├── schema.py             # Pydantic v2 AgentConfig, LLMConfig, ExecutionConfig
│   │   └── loader.py             # YAML loader, agents/ scanner, and engine builder
│   ├── api/                      # FastAPI serving layer (REST & SSE)
│   │   ├── app.py                # App factory & route handlers
│   │   └── schemas.py            # API request & response schemas
│   └── observability/            # Telemetry, metrics, and structured logging
│       ├── tracer.py             # ExecutionTracer & trace_stream wrapper
│       └── logger.py             # Structured JSON logger & secret redactor
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_tools.py
│   ├── test_llm.py
│   ├── test_llm_anthropic.py
│   ├── test_llm_gemini.py
│   ├── test_llm_contract.py
│   ├── test_engine.py
│   ├── test_config.py
│   ├── test_api.py
│   ├── test_api_sessions.py
│   ├── test_memory_contract.py
│   ├── test_memory_sqlite.py
│   ├── test_memory_window.py
│   └── test_observability.py
├── .env.example                  # Environment variable template
├── pyproject.toml                # Project metadata & dependencies
├── README.md                     # Project documentation
└── SUMMARY.md                    # Architecture & engineering breakdown
```

---

## ⚡ Installation & Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ChopraArhaan/Open-source-agent-builder.git
cd Open-source-agent-builder

# Install package with dependencies (core OpenAI & Fake providers)
pip install -e .

# Or install with native provider SDKs
pip install -e ".[anthropic]"       # Native Anthropic Claude support
pip install -e ".[gemini]"          # Native Google Gemini support
pip install -e ".[all-providers]"   # All provider SDKs

# Or install with development and test dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

Copy the `.env.example` file to `.env` and set your provider API key:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start the FastAPI Server

```bash
uvicorn agent_builder.api.app:app --reload --port 8000
```

---

## 🔌 Supported LLM Providers

Open Source Agent Builder provides a standardized multi-provider gateway supporting native tool calling, token usage telemetry, and automatic exponential backoff with jitter across top LLM providers:

| Provider | Provider String | Install Extra | Required Environment Variable | Default / Recommended Models |
|---|---|---|---|---|
| **OpenAI** | `openai` | *(Included in core)* | `OPENAI_API_KEY` | `gpt-4o-mini`, `gpt-4o` |
| **Anthropic** | `anthropic` | `agent-builder[anthropic]` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022`, `claude-3-7-sonnet-20250219` |
| **Google Gemini** | `gemini` | `agent-builder[gemini]` | `GEMINI_API_KEY` | `gemini-2.0-flash`, `gemini-2.5-flash` |
| **Offline Fake** | `fake` | *(Included in core)* | *None* | `fake-model` |

### Provider Error Handling & Retry Policies
- **Transient Failures (Retried)**: Rate limits (HTTP 429), server errors (HTTP 500, 502, 503, 504), network dropouts, and timeouts are automatically retried with exponential backoff and randomized jitter (`max_retries=3`, `initial_delay=1.0s`, `max_delay=60.0s`).
- **Permanent Client Errors (Fast Fail)**: Invalid authentication keys (HTTP 401), permission errors (HTTP 403), and bad requests (HTTP 400) immediately abort without retry.
- **Secret Safety**: API keys are never exposed in `__repr__`, debug logs, or exception traces.

### Example: Anthropic Claude Agent (`agents/claude_analyst.yaml`)

```yaml
name: "claude-analyst"
version: "1.0.0"
description: "Data analysis agent powered by Anthropic Claude Messages API."

llm:
  provider: "anthropic"
  model: "claude-3-5-haiku-20241022"
  temperature: 0.2
  max_tokens: 4096

system_prompt: |
  You are an analytical assistant. Perform precise calculations using available tools
  and provide concise, well-structured summaries.

tools:
  - calculator

execution:
  max_iterations: 10
  timeout_seconds: 60.0
  max_repeats: 3
```

### Example: Google Gemini Agent (`agents/gemini_assistant.yaml`)

```yaml
name: "gemini-assistant"
version: "1.0.0"
description: "General assistant powered by native Google Gemini SDK."

llm:
  provider: "gemini"
  model: "gemini-2.0-flash"
  temperature: 0.2
  max_tokens: 2048

system_prompt: |
  You are a helpful assistant powered by Google Gemini. Use tools when required
  to evaluate expressions accurately.

tools:
  - calculator

execution:
  max_iterations: 10
  timeout_seconds: 60.0
  max_repeats: 3
```

---

## 📄 Declarative Agent Specification

Agents are defined declaratively in YAML files placed in the `agents/` directory:

```yaml
name: "math-tutor"
version: "1.0.0"
description: "Interactive math tutor that solves arithmetic problems step-by-step and validates calculations."

llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1000

system_prompt: |
  You are a patient and rigorous math tutor. Break down mathematical problems
  into clear, easy-to-follow steps. Always evaluate arithmetic expressions
  using the calculator tool rather than calculating in your head to guarantee accuracy.

tools:
  - calculator

execution:
  max_iterations: 10
  timeout_seconds: 60.0
  max_repeats: 3
```

---

## 🛠️ Defining Custom Tools

Declare tools using standard Pydantic models. Sync and async functions are wrapped uniformly:

```python
from pydantic import BaseModel, Field
from agent_builder.tools import register_tool

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression (e.g. '2 + 2 * 3')")

@register_tool(
    name="calculator",
    description="Safely evaluate mathematical expressions using AST.",
    args_schema=CalculatorInput,
    timeout_seconds=5.0,
)
def calculator(expression: str) -> str:
    # Safe computation logic here
    return str(eval_expression(expression))
```

---

## 🌐 API Reference

The server exposes the following endpoints:

| Method | Endpoint | Description | Request Body | Response Body |
|---|---|---|---|---|
| `GET` | `/health` | Service health status | None | `{"status": "ok", "version": "0.1.0"}` |
| `GET` | `/agents` | List all discovered agents | None | `list[AgentSummary]` |
| `GET` | `/agents/{name}` | Get declarative agent YAML config | None | `AgentConfig` (or 404) |
| `GET` | `/tools` | List registered tools and schemas | None | `list[ToolSummary]` |
| `POST` | `/agents/{name}/run` | Run agent synchronously | `RunRequest` | `RunResponse` |
| `POST` | `/agents/{name}/stream` | Stream agent execution via SSE | `RunRequest` | `text/event-stream` |

### Request & Response Models

#### `RunRequest`
```json
{
  "prompt": "Calculate (25 * 4) + 50",
  "session_messages": []
}
```

#### `RunResponse`
```json
{
  "agent_name": "math-tutor",
  "final_answer": "The result is 150.",
  "events": [
    {
      "type": "tool_call",
      "step": 1,
      "content": null,
      "tool_call": {
        "id": "call_abc123",
        "name": "calculator",
        "arguments": {"expression": "(25 * 4) + 50"}
      },
      "tool_result": null,
      "metadata": {},
      "timestamp": "2026-11-21T20:00:00Z"
    },
    {
      "type": "observation",
      "step": 1,
      "content": null,
      "tool_call": null,
      "tool_result": {
        "call_id": "call_abc123",
        "tool_name": "calculator",
        "output": "150",
        "is_error": false,
        "error_message": null,
        "duration_ms": 0.4
      },
      "metadata": {},
      "timestamp": "2026-11-21T20:00:01Z"
    },
    {
      "type": "final",
      "step": 2,
      "content": "The result is 150.",
      "tool_call": null,
      "tool_result": null,
      "metadata": {},
      "timestamp": "2026-11-21T20:00:02Z"
    }
  ]
}
```

---

## 📡 Server-Sent Events (SSE) Streaming

The `/agents/{name}/stream` endpoint delivers live progress chunks in standard SSE format (`event: <type>\ndata: <json>\n\n`).

### Supported Event Types
- `thought`: Intermediate reasoning emitted by the model prior to taking an action.
- `tool_call`: Tool invocation request with function name and arguments.
- `observation`: The outcome returned by the tool execution.
- `final`: The final completion answer returned by the agent.
- `error`: Loop termination errors (e.g. `timeout`, `cycle`, `max_iterations`, `llm_error`).

### Verified Client Examples

#### 1. Python (`httpx` async client)

```python
import asyncio
import json
import httpx

async def stream_agent(prompt: str):
    url = "http://localhost:8000/agents/math-tutor/stream"
    payload = {"prompt": prompt}

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=60.0) as response:
            current_event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:].strip())
                    print(f"[{current_event}] Step {data.get('step')}: {data.get('type')}")
                    if data.get("type") == "final":
                        print(f"Final Answer: {data.get('content')}")

if __name__ == "__main__":
    asyncio.run(stream_agent("What is 25 * 4?"))
```

#### 2. cURL (`curl -N`)

```bash
curl -N -s -X POST "http://localhost:8000/agents/math-tutor/stream" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Calculate (25 * 4) + 50"}'
```

---

## 🔍 Observability, Tracing & Logging

Open Source Agent Builder includes a non-intrusive observer system that computes execution metrics and structured logs without modifying the engine.

### 1. Enabling Structured Logging

Use `setup_logger` to configure logging without touching the root logger:

```python
from agent_builder.observability import setup_logger

# Configure JSON formatting to stdout
logger = setup_logger(level="INFO", json=True, log_content=False)
```

**Security & Sanitization:**
- **Key Redaction**: Any dictionary keys matching `api_key`, `token`, `password`, `secret`, `authorization`, or `cookie` are automatically replaced with `"[REDACTED]"`.
- **String Truncation**: Strings exceeding 500 characters are truncated with `...[truncated N chars]`.
- Set `log_content=True` to disable redaction and truncation when debugging locally.

### 2. Tracing an Execution (`ExecutionTracer`)

Wrap any event generator with `trace_stream`:

```python
from agent_builder.engine.react import ReActEngine
from agent_builder.observability import ExecutionTracer, trace_stream

tracer = ExecutionTracer(agent="math-tutor")

async for event in trace_stream(engine.run_stream("Calculate 12 * 8"), tracer):
    # Process events as they stream
    pass

# Export trace to dict or JSON file
trace_dict = tracer.to_dict()
tracer.to_json(path="traces/math_run.json")
```

### Real Sample Trace JSON

```json
{
  "schema_version": 1,
  "run_id": "beeaf168-30da-4c7e-88a3-cd54243b6e37",
  "agent": "math-tutor",
  "started_at": "2026-11-21T20:41:53.955414+00:00",
  "ended_at": "2026-11-21T20:41:53.956410+00:00",
  "status": "final",
  "metrics": {
    "step_count": 2,
    "tool_call_count": 1,
    "tool_calls_by_name": {
      "calculator": 1
    },
    "tool_errors_by_name": {},
    "total_duration_ms": 1.2,
    "token_usage": null
  },
  "steps": [
    {
      "step": 1,
      "thought": null,
      "tool_name": "calculator",
      "tool_arguments": {
        "expression": "12 * 8"
      },
      "observation": "96",
      "ok": true,
      "duration_ms": 0.4
    },
    {
      "step": 2,
      "thought": "The answer is 96.",
      "tool_name": null,
      "tool_arguments": null,
      "observation": null,
      "ok": null,
      "duration_ms": null
    }
  ]
}
```

> **Telemetry Caveats**:
> - **Tool Duration**: Prefers `ToolResult.duration_ms` when recorded by the tool runner. Fallback measures time between `tool_call` and `observation` as seen by the observer (which includes event dispatch latency in streaming mode).
> - **Token Usage**: Summed only when reported by the LLM provider in metadata (`usage`). If not reported, token usage is `null` (never `0`) so missing telemetry is not mistaken for a free run.

---

## 🧠 Sessions & Persistent Memory

Open Source Agent Builder supports optional multi-turn conversation sessions backed by SQLite (with WAL mode) or an in-memory store for testing.

### Design Principles
- **Stateless by default**: Calling `/agents/{name}/run` or `/agents/{name}/stream` without `session_id` remains completely stateless and persists nothing.
- **Clean turn history**: Only `user` prompts and `final` `assistant` answers are stored. System prompts come from the agent YAML config, and internal ReAct scratchpad events (`thought`, `tool_call`, `observation`) are excluded.
- **Crash & cancel safety**: A turn (`user` + `assistant`) is appended in one atomic transaction **only after** execution produces a `final` event. Errors or mid-stream client disconnects persist nothing.
- **Per-session serialization**: Concurrent requests targeting the same `session_id` are serialized via reference-counted per-session async locks; requests on different sessions run in parallel.

### Session API Routes

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/sessions` | Create a new session bound to an agent (`201 Created`) |
| `GET` | `/sessions` | List sessions (`?agent=...&limit=50&offset=0`) |
| `GET` | `/sessions/{session_id}` | Fetch a session with its ordered messages |
| `DELETE` | `/sessions/{session_id}` | Delete a session and cascade-delete its messages (`204 No Content`) |

Passing `session_id` in `POST /agents/{name}/run` or `POST /agents/{name}/stream` loads windowed history into the engine and persists the completed turn. Unknown `session_id`s return `404`, and mismatched agent names return `409 Conflict`.

### Multi-Turn Examples

#### 1. cURL

```bash
# 1. Create a session bound to math-tutor
SESSION_ID=$(curl -s -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{"agent": "math-tutor", "metadata": {"channel": "cli"}}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['session_id'])")

# 2. Turn 1
curl -s -X POST "http://localhost:8000/agents/math-tutor/run" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"What is 12 * 8?\", \"session_id\": \"$SESSION_ID\"}"

# 3. Turn 2 (receives Turn 1 context automatically)
curl -s -X POST "http://localhost:8000/agents/math-tutor/run" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Add 4 to that result.\", \"session_id\": \"$SESSION_ID\"}"

# 4. Inspect persisted session
curl -s "http://localhost:8000/sessions/$SESSION_ID"
```

#### 2. Python (`httpx`)

```python
import httpx

with httpx.Client(base_url="http://localhost:8000") as client:
    # 1. Create session
    session = client.post(
        "/sessions",
        json={"agent": "math-tutor", "metadata": {"source": "python"}},
    ).json()
    session_id = session["session_id"]

    # 2. Run multi-turn conversation
    r1 = client.post(
        "/agents/math-tutor/run",
        json={"prompt": "What is 12 * 8?", "session_id": session_id},
    ).json()
    print("Turn 1:", r1["final_answer"])

    r2 = client.post(
        "/agents/math-tutor/run",
        json={"prompt": "Add 4 to that result.", "session_id": session_id},
    ).json()
    print("Turn 2:", r2["final_answer"])

    # 3. Fetch stored history
    stored = client.get(f"/sessions/{session_id}").json()
    print("Stored messages:", len(stored["messages"]))
```

### Configuration, Windowing & Retention

| Environment Variable | Default | Description |
|---|---|---|
| `AGENT_BUILDER_DB_PATH` | `data/sessions.db` | SQLite database file path (or `:memory:`) |
| `AGENT_BUILDER_SESSION_TTL_DAYS` | *(unset / off)* | If > 0, purges sessions older than N days on startup |
| `AGENT_BUILDER_MAX_HISTORY_MESSAGES` | *(unset / unlimited)* | Max historical messages sent to the engine per run |
| `AGENT_BUILDER_MAX_HISTORY_CHARS` | *(unset / unlimited)* | Max historical characters sent to the engine per run |

- **History Windowing**: Full history always remains in the store. Before each run, `select_history()` selects the most recent complete `(user, assistant)` turns fitting within `max_messages` and `max_chars` without ever splitting a turn pair or starting with an `assistant` message.
- **Session Limits**: Enforces default limits of `100,000` chars per message, `1,000` messages per session, and `16,384` bytes of JSON metadata.

### Session Store Limitations
- **Plaintext storage**: The SQLite database is unencrypted at rest (though newly created DB files are set to owner-only permissions where supported by the OS).
- **Single-process only**: Designed for a single server process using SQLite WAL mode and in-process per-session async locks; not intended for multi-node horizontal scaling without an external database.
- **No authentication on sessions**: Sessions have no authentication (a `session_id` is an unguessable UUIDv4, but it is not access control).
- **Character-based windowing**: History windowing (`AGENT_BUILDER_MAX_HISTORY_CHARS`) counts characters, not tokens.

---

## ⚠️ Known Limitations

- **Key-based redaction only**: Sensitive data masking in structured logs inspects dictionary keys matching sensitive patterns (`api_key`, `token`, `password`, `secret`, `authorization`, `cookie`). It does not parse or inspect unstructured free text or raw strings for secrets.
- **Traces contain unredacted arguments and outputs**: Traces include raw tool arguments and outputs and are not redacted, so don't share traces from sensitive runs.
- **Provider-dependent token metrics**: Token usage counts are accumulated strictly when the LLM provider reports them in `usage` metadata. If a provider omits token usage or returns `None`, metrics report `null` rather than `0` to prevent false assumptions of zero-cost runs.
- **Trace tool durations may include client latency**: While `ToolResult.duration_ms` is prioritized for precise tool execution timing, observer fallback durations (measured from `tool_call` to `observation` events) can include SSE event streaming latency and client consumer processing time.

---

## 🛣️ Roadmap

- [x] Architecture design & project spec
- [x] Pydantic tool registry with JSON Schema export (Phase 1)
- [x] Unified LLM Gateway & Fake test provider (Phase 2)
- [x] Core ReAct execution loop & event stream (Phase 3)
- [x] Declarative YAML parser & validator (Phase 4)
- [x] FastAPI service with REST & SSE endpoints (Phase 5)
- [x] Observability, execution tracing & structured logging (Phase 6)
- [x] Multi-provider expansions (Anthropic, Gemini native SDKs)
- [x] Persistent agent memory & conversation session store
- [ ] Interactive Web UI / Playground

---

## 📜 License

MIT License. Feel free to use, modify, and distribute.
