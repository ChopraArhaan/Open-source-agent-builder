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
│   │   └── fake.py               # Scriptable offline test provider
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
│   ├── test_engine.py
│   ├── test_config.py
│   ├── test_api.py
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

# Install package with dependencies
pip install -e .

# Or install with test dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

Copy the `.env.example` file to `.env` and set your provider API key:

```bash
cp .env.example .env
```

```ini
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the FastAPI Server

```bash
uvicorn src.agent_builder.api.app:app --reload --port 8000
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
from src.agent_builder.tools import register_tool

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
from src.agent_builder.observability import setup_logger

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
from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.observability import ExecutionTracer, trace_stream

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

## 🛣️ Roadmap

- [x] Architecture design & project spec
- [x] Pydantic tool registry with JSON Schema export (Phase 1)
- [x] Unified LLM Gateway & Fake test provider (Phase 2)
- [x] Core ReAct execution loop & event stream (Phase 3)
- [x] Declarative YAML parser & validator (Phase 4)
- [x] FastAPI service with REST & SSE endpoints (Phase 5)
- [x] Observability, execution tracing & structured logging (Phase 6)
- [ ] Multi-provider expansions (Anthropic, Gemini native SDKs)
- [ ] Persistent agent memory & conversation session store
- [ ] Interactive Web UI / Playground

---

## 📜 License

MIT License. Feel free to use, modify, and distribute.
