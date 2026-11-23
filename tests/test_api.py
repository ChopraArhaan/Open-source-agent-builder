"""Test suite for FastAPI endpoints: health, agents, tools, run, and SSE streaming."""

import json

import pytest
from fastapi.testclient import TestClient

from src.agent_builder.api.app import create_app
from src.agent_builder.config.schema import AgentConfig, ExecutionConfig, LLMConfig
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.models import ToolCall
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


@pytest.fixture
def test_setup():
    """Setup an isolated registry, sample configs, and fake LLM."""
    registry = ToolRegistry()
    register_builtin_tools(registry)

    configs = {
        "math-agent": AgentConfig(
            name="math-agent",
            version="1.0.0",
            description="Agent for mathematical calculations",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="Solve math problems accurately.",
            tools=["calculator"],
            execution=ExecutionConfig(max_iterations=5),
        ),
        "research-agent": AgentConfig(
            name="research-agent",
            version="2.0.0",
            description="Agent for searching info",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="Search facts accurately.",
            tools=["mock_search", "current_time"],
        ),
    }

    fake_llm = FakeLLMProvider()

    app = create_app(
        registry=registry,
        llm_override=fake_llm,
        configs=configs,
    )
    client = TestClient(app)
    return client, fake_llm, registry, configs


def test_health_endpoint(test_setup):
    """Verify /health returns 200 ok status."""
    client, _, _, _ = test_setup
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_and_get_agents(test_setup):
    """Verify /agents lists all agents and /agents/{name} returns config."""
    client, _, _, _ = test_setup

    # 1. List agents
    resp = client.get("/agents")
    assert resp.status_code == 200
    agents = resp.json()
    assert len(agents) == 2
    names = {a["name"] for a in agents}
    assert names == {"math-agent", "research-agent"}

    # 2. Get specific agent
    resp_math = client.get("/agents/math-agent")
    assert resp_math.status_code == 200
    math_cfg = resp_math.json()
    assert math_cfg["name"] == "math-agent"
    assert math_cfg["tools"] == ["calculator"]
    assert math_cfg["system_prompt"] == "Solve math problems accurately."

    # 3. 404 for unknown agent
    resp_unknown = client.get("/agents/ghost-agent")
    assert resp_unknown.status_code == 404
    assert "not found" in resp_unknown.json()["detail"].lower()


def test_list_tools(test_setup):
    """Verify /tools lists registered tools with parameters and timeout."""
    client, _, _, _ = test_setup
    resp = client.get("/tools")
    assert resp.status_code == 200
    tools = resp.json()
    tool_names = {t["name"] for t in tools}
    assert "calculator" in tool_names
    assert "current_time" in tool_names
    assert "mock_search" in tool_names

    calc_tool = next(t for t in tools if t["name"] == "calculator")
    assert "parameters" in calc_tool
    assert calc_tool["timeout_seconds"] > 0


def test_run_agent_single_turn(test_setup):
    """Verify POST /agents/{name}/run with single turn answer."""
    client, fake_llm, _, _ = test_setup
    fake_llm.set_responses(["42 is the answer."])

    resp = client.post(
        "/agents/math-agent/run",
        json={"prompt": "What is the meaning of life?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_name"] == "math-agent"
    assert data["final_answer"] == "42 is the answer."
    assert len(data["events"]) == 1
    assert data["events"][0]["type"] == "final"
    assert data["events"][0]["content"] == "42 is the answer."


def test_run_agent_with_tool_call(test_setup):
    """Verify POST /agents/{name}/run executes tool call cycle and returns events."""
    client, fake_llm, _, _ = test_setup
    fake_llm.set_responses([
        [ToolCall(name="calculator", arguments={"expression": "12 * 12"})],
        "12 * 12 is 144.",
    ])

    resp = client.post(
        "/agents/math-agent/run",
        json={"prompt": "Compute 12 * 12"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_name"] == "math-agent"
    assert data["final_answer"] == "12 * 12 is 144."
    assert len(data["events"]) == 3
    assert data["events"][0]["type"] == "tool_call"
    assert data["events"][1]["type"] == "observation"
    assert data["events"][1]["tool_result"]["output"] == "144"
    assert data["events"][2]["type"] == "final"


def test_run_unknown_agent_returns_404(test_setup):
    """Verify running a non-existent agent returns 404."""
    client, _, _, _ = test_setup
    resp = client.post(
        "/agents/not-a-real-agent/run",
        json={"prompt": "Hello"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_run_validation_error(test_setup):
    """Verify empty prompt returns 422 validation error."""
    client, _, _, _ = test_setup
    resp = client.post(
        "/agents/math-agent/run",
        json={"prompt": ""},
    )
    assert resp.status_code == 422


def test_stream_agent_sse(test_setup):
    """Verify POST /agents/{name}/stream yields real-time Server-Sent Events."""
    client, fake_llm, _, _ = test_setup
    fake_llm.set_responses([
        [ToolCall(name="calculator", arguments={"expression": "5 + 5"})],
        "Result is 10.",
    ])

    resp = client.post(
        "/agents/math-agent/stream",
        json={"prompt": "What is 5 + 5?"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    # Parse SSE text blocks
    text = resp.text
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    assert len(blocks) == 3

    # Block 1: tool_call
    assert "event: tool_call" in blocks[0]
    data1_line = next(line for line in blocks[0].split("\n") if line.startswith("data: "))
    payload1 = json.loads(data1_line[6:])
    assert payload1["type"] == "tool_call"
    assert payload1["tool_call"]["name"] == "calculator"

    # Block 2: observation
    assert "event: observation" in blocks[1]
    data2_line = next(line for line in blocks[1].split("\n") if line.startswith("data: "))
    payload2 = json.loads(data2_line[6:])
    assert payload2["type"] == "observation"
    assert payload2["tool_result"]["output"] == "10"

    # Block 3: final
    assert "event: final" in blocks[2]
    data3_line = next(line for line in blocks[2].split("\n") if line.startswith("data: "))
    payload3 = json.loads(data3_line[6:])
    assert payload3["type"] == "final"
    assert payload3["content"] == "Result is 10."


def test_stream_unknown_agent_returns_404(test_setup):
    """Verify streaming a non-existent agent returns 404."""
    client, _, _, _ = test_setup
    resp = client.post(
        "/agents/unknown/stream",
        json={"prompt": "Hello"},
    )
    assert resp.status_code == 404


def test_api_v1_prefix_endpoints(test_setup):
    """Verify /api/v1 prefix routes serve agents, tools, run, and stream."""
    client, fake_llm, _, _ = test_setup
    fake_llm.set_responses(["v1 response"])

    # 1. /api/v1/health
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 2. /api/v1/agents
    r_agents = client.get("/api/v1/agents")
    assert r_agents.status_code == 200
    assert len(r_agents.json()) == 2

    # 3. /api/v1/agents/{name}
    r_agent = client.get("/api/v1/agents/math-agent")
    assert r_agent.status_code == 200
    assert r_agent.json()["name"] == "math-agent"

    # 4. /api/v1/tools
    r_tools = client.get("/api/v1/tools")
    assert r_tools.status_code == 200
    assert any(t["name"] == "calculator" for t in r_tools.json())

    # 5. /api/v1/agents/{name}/run
    r_run = client.post("/api/v1/agents/math-agent/run", json={"prompt": "Calculate 1+1"})
    assert r_run.status_code == 200
    assert r_run.json()["final_answer"] == "v1 response"


def test_auth_disabled_by_default(test_setup):
    """Verify all endpoints are open when api_key is None / unset."""
    client, _, _, _ = test_setup
    assert client.get("/agents").status_code == 200
    assert client.get("/tools").status_code == 200
    assert client.get("/health").status_code == 200


def test_auth_enforcement_and_credentials():
    """Verify API-key authentication behavior on protected endpoints and health exemption."""
    configs = {
        "dummy": AgentConfig(
            name="dummy",
            llm=LLMConfig(provider="fake", model="fake-model"),
            system_prompt="Dummy",
        )
    }
    app = create_app(
        configs=configs,
        api_key="my-secret-key-123",
        llm_override=FakeLLMProvider(responses=["Done"]),
    )
    client = TestClient(app)

    # 1. /health and /api/v1/health stay open without any key
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 200

    # 2. Missing key returns 401
    r_missing = client.get("/agents")
    assert r_missing.status_code == 401
    assert r_missing.json()["detail"] == "Invalid or missing API key"

    assert client.get("/tools").status_code == 401
    assert client.post("/agents/dummy/run", json={"prompt": "hi"}).status_code == 401

    # 3. Wrong key returns 401
    assert client.get("/agents", headers={"X-API-Key": "wrong-key"}).status_code == 401
    assert client.get("/agents", headers={"Authorization": "Bearer wrong-key"}).status_code == 401

    # 4. Query string is NEVER accepted (must be 401)
    r_query = client.get("/agents?api_key=my-secret-key-123")
    assert r_query.status_code == 401

    r_query2 = client.get("/agents?token=my-secret-key-123")
    assert r_query2.status_code == 401

    # 5. Valid X-API-Key header returns 200
    r_header = client.get("/agents", headers={"X-API-Key": "my-secret-key-123"})
    assert r_header.status_code == 200
    assert len(r_header.json()) == 1

    # 6. Valid Authorization Bearer header returns 200
    r_bearer = client.get("/agents", headers={"Authorization": "Bearer my-secret-key-123"})
    assert r_bearer.status_code == 200
    assert len(r_bearer.json()) == 1

    # 7. Check /api/v1 routes with valid auth
    assert client.get("/api/v1/agents", headers={"X-API-Key": "my-secret-key-123"}).status_code == 200
    assert client.get("/api/v1/tools", headers={"X-API-Key": "my-secret-key-123"}).status_code == 200


def test_docs_and_openapi_stay_open_with_auth():
    """Verify OpenAPI schema and docs (/openapi.json, /docs, /redoc) remain open when auth is enabled."""
    app = create_app(api_key="secret-token-xyz")
    client = TestClient(app)

    # Core protected endpoints are blocked
    assert client.get("/agents").status_code == 401
    assert client.get("/tools").status_code == 401

    # Documentation and schema endpoints remain accessible without auth
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200



def test_cors_behavior_and_wildcard_credentials():
    """Verify CORS configuration defaults to no allowed origins and never pairs '*' with credentials."""
    # 1. Default: no allowed origins
    app_default = create_app()
    client_default = TestClient(app_default)
    resp_def = client_default.options(
        "/health",
        headers={
            "Origin": "https://random.site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_def.headers.get("access-control-allow-origin") is None

    # 2. Configured origins: specific allowed origin allows credentials
    app_custom = create_app(cors_origins="https://dashboard.example.com, https://admin.example.com")
    client_custom = TestClient(app_custom)
    resp_custom = client_custom.options(
        "/health",
        headers={
            "Origin": "https://dashboard.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_custom.headers.get("access-control-allow-origin") == "https://dashboard.example.com"
    assert resp_custom.headers.get("access-control-allow-credentials") == "true"

    # 3. Wildcard origin: NEVER allow credentials
    app_wildcard = create_app(cors_origins="*")
    client_wildcard = TestClient(app_wildcard)
    resp_wild = client_wildcard.options(
        "/health",
        headers={
            "Origin": "https://any.site.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp_wild.headers.get("access-control-allow-origin") == "*"
    assert resp_wild.headers.get("access-control-allow-credentials") is None

