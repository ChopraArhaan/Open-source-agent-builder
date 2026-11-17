# 🤖 Open Source Agent Builder

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.0+-red.svg)](https://docs.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Build, run, and deploy autonomous AI agents from scratch.**
> Built with pure Python and FastAPI — featuring a raw ReAct execution loop, dynamic Pydantic tool registry, and declarative YAML configs. Zero framework bloat, maximum transparency.

---

## 💡 Overview

Many contemporary AI agent frameworks hide core mechanics behind thick abstractions, complex inheritance trees, and brittle dependencies. **Open Source Agent Builder** is built from first principles to demystify agentic systems and deliver a clean, production-grade API service for custom agents.

It provides everything you need to build, test, and serve autonomous agents without relying on heavyweight orchestration libraries.

---

## ✨ Key Features

- **⚡ Zero Framework Bloat**: Direct LLM calls and explicit state handling. No LangChain or LlamaIndex black boxes.
- **🔄 Raw ReAct Execution Loop**: Fully observable **Thought ➔ Action ➔ Observation** runtime loop with configurable iteration limits, loop detection, and early exits.
- **🛠️ Pydantic-Powered Tool Registry**: Declare tools using type-safe Pydantic models. Automatically generates OpenAI/Anthropic/Gemini-compatible tool schemas.
- **📄 Declarative Agent YAMLs**: Define personas, instructions, tools, system prompts, and model hyper-parameters in human-readable YAML specs.
- **🚀 Production-Ready FastAPI Runtime**: Expose your agents instantly via REST endpoints, background tasks, and real-time Server-Sent Events (SSE) streaming.
- **🔌 Multi-Provider LLM Gateway**: Standardized client layer supporting OpenAI, Anthropic Claude, Google Gemini, and local Ollama models.
- **🛡️ Execution Guardrails & Sandboxing**: Timeout controls, tool execution policies, and validation gates before invoking destructive tools.

---

## 🏛️ Architecture

```
                          ┌────────────────────────┐
                          │   declarative YAML     │
                          │     (agent.yaml)       │
                          └───────────┬────────────┘
                                      │ loads
                                      ▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Client / UI    │ ◄───► │    FastAPI Server      │ ◄───► │   Tool Registry      │
│ (REST / SSE)    │       │ (Sessions & Endpoints) │       │ (Pydantic Validated) │
└─────────────────┘       └───────────┬────────────┘       └──────────┬───────────┘
                                      │ runs                          │
                                      ▼                               │ executes
                          ┌────────────────────────┐                  │
                          │   ReAct Execution Loop │ ◄────────────────┘
                          │ (Thought/Act/Observe)  │
                          └───────────┬────────────┘
                                      │ calls
                                      ▼
                          ┌────────────────────────┐
                          │  Unified LLM Gateway   │
                          │(OpenAI/Claude/Gemini)  │
                          └────────────────────────┘
```

---

## 📂 Project Structure

```text
open-source-agent-builder/
├── agents/                       # Declarative agent configurations (YAML)
│   ├── web_researcher.yaml
│   └── code_assistant.yaml
├── src/
│   ├── api/                      # FastAPI application layer
│   │   ├── routes/               # API endpoints (run, stream, tools, agents)
│   │   ├── dependencies.py       # Dependency injection
│   │   └── server.py             # FastAPI entrypoint
│   ├── core/                     # Agent execution core
│   │   ├── engine.py             # ReAct loop implementation
│   │   ├── memory.py             # Working memory & conversation buffer
│   │   ├── parser.py             # Thought/Action parsing & tool-call handling
│   │   └── state.py              # Execution state models
│   ├── llm/                      # Unified LLM provider interface
│   │   ├── base.py               # Abstract provider interface
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   └── gemini_provider.py
│   ├── tools/                    # Tool registry and built-in tools
│   │   ├── registry.py           # Dynamic tool decorator & registry
│   │   ├── builtins/             # Default tools (web search, calculator, etc.)
│   │   └── sandbox.py            # Execution wrapper & safe sandbox
│   └── utils/
│       ├── config.py             # Global application configuration
│       └── logger.py             # Structured telemetry & logging
├── tests/                        # Comprehensive unit & integration tests
├── .env.example                  # Environment variable template
├── pyproject.toml                # Project metadata & dependencies
├── README.md                     # Project documentation
└── SUMMARY.md                    # Architecture & engineering breakdown
```

---

## ⚡ Quickstart

### 1. Define an Agent in YAML (`agents/analyst.yaml`)

```yaml
name: "market-analyst"
version: "1.0.0"
description: "Autonomous agent for researching market trends and stock data"

llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.2
  max_tokens: 1500

system_prompt: |
  You are an expert market analyst. Answer the user request using the tools provided.
  Always explain your thinking before taking an action.
  Validate your data before formulating your final response.

tools:
  - web_search
  - financial_calculator
  - fetch_ticker_data

execution:
  max_iterations: 8
  timeout_seconds: 60
  early_stop_strategy: "force_finalize"
```

### 2. Define a Tool with Pydantic

```python
from pydantic import BaseModel, Field
from src.tools.registry import register_tool

class SearchInput(BaseModel):
    query: str = Field(description="Search keyword or expression")
    max_results: int = Field(default=5, description="Number of results to retrieve")

@register_tool(
    name="web_search",
    description="Searches the web for up-to-date facts and documents.",
    args_schema=SearchInput
)
async def web_search(query: str, max_results: int = 5) -> str:
    # Tool execution logic here
    return f"Retrieved {max_results} results for '{query}'"
```

### 3. Run via FastAPI

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.api.server:app --reload --port 8000
```

#### Run an Agent Request:
```bash
curl -X POST "http://localhost:8000/api/v1/agents/market-analyst/run" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Analyze the revenue growth of NVIDIA over the last 3 quarters"}'
```

---

## 🛣️ Roadmap

- [x] Architecture design & project spec
- [ ] Core ReAct execution loop & state engine
- [ ] Pydantic tool registry with JSON Schema export
- [ ] Unified LLM Gateway (OpenAI, Gemini, Anthropic)
- [ ] Declarative YAML parser & validator
- [ ] FastAPI service with REST & SSE endpoints
- [ ] Agent memory store (in-memory + persistent SQLite)
- [ ] Interactive Web UI / Playground

---

## 📜 License

MIT License. Feel free to use, modify, and distribute.
