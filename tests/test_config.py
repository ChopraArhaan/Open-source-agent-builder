"""Test suite for AgentConfig schema validation, YAML loading, directory scanning, and engine assembly."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent_builder.config.loader import (
    AgentConfigError,
    build_engine_from_config,
    load_agent_config,
    scan_agents_dir,
)
from src.agent_builder.config.schema import (
    AgentConfig,
    ExecutionConfig,
    LLMConfig,
)
from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.models import ToolCall
from src.agent_builder.tools import default_registry
from src.agent_builder.tools.builtins import register_builtin_tools
from src.agent_builder.tools.registry import ToolRegistry


def test_valid_yaml_parsing(tmp_path: Path):
    """Verify valid YAML parses into AgentConfig with exact expected values."""
    yaml_content = """
name: "finance-bot"
version: "2.1.0"
description: "Handles financial calculations"
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 500
system_prompt: "You are a financial advisor."
tools:
  - "calculator"
execution:
  max_iterations: 5
  timeout_seconds: 45.0
  max_repeats: 4
"""
    config_file = tmp_path / "finance.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = load_agent_config(config_file)
    assert config.name == "finance-bot"
    assert config.version == "2.1.0"
    assert config.description == "Handles financial calculations"
    assert config.llm.provider == "openai"
    assert config.llm.model == "gpt-4o"
    assert config.llm.temperature == 0.3
    assert config.llm.max_tokens == 500
    assert config.system_prompt == "You are a financial advisor."
    assert config.tools == ["calculator"]
    assert config.execution.max_iterations == 5
    assert config.execution.timeout_seconds == 45.0
    assert config.execution.max_repeats == 4


def test_omitted_execution_gets_defaults(tmp_path: Path):
    """Verify omitting execution section uses default ExecutionConfig."""
    yaml_content = """
name: "minimal-bot"
llm:
  provider: "openai"
  model: "gpt-4o-mini"
system_prompt: "Minimal system prompt"
"""
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")

    config = load_agent_config(config_file)
    assert config.name == "minimal-bot"
    assert config.version == "1.0.0"
    assert config.description == ""
    assert config.tools == []
    assert config.execution.max_iterations == 10
    assert config.execution.timeout_seconds == 120.0
    assert config.execution.max_repeats == 3


def test_schema_validation_errors():
    """Verify strict validation rules: extra keys, name pattern, ranges, duplicate tools."""
    # 1. Unknown / extra key forbidden (extra='forbid')
    with pytest.raises(ValidationError, match="extra_forbidden"):
        AgentConfig(
            name="test-bot",
            system_prompt="Prompt",
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            unknown_field="not allowed",  # type: ignore
        )

    # 1b. api_key in llm config forbidden
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LLMConfig(provider="openai", model="gpt-4o", api_key="secret-key")  # type: ignore

    # 2. Bad name format (must match ^[a-z0-9][a-z0-9_-]*$)
    for bad_name in ["TestBot", "-bad", "with spaces", "name!", ""]:
        with pytest.raises(ValidationError):
            AgentConfig(
                name=bad_name,
                system_prompt="Prompt",
                llm=LLMConfig(provider="openai", model="gpt-4o"),
            )

    # 3. Temperature out of range [0.0, 2.0]
    with pytest.raises(ValidationError):
        LLMConfig(provider="openai", model="gpt-4o", temperature=-0.1)

    with pytest.raises(ValidationError):
        LLMConfig(provider="openai", model="gpt-4o", temperature=2.1)

    # 4. max_iterations < 1
    with pytest.raises(ValidationError):
        ExecutionConfig(max_iterations=0)

    # 5. Non-positive timeout
    with pytest.raises(ValidationError):
        ExecutionConfig(timeout_seconds=0.0)

    # 6. max_repeats < 2
    with pytest.raises(ValidationError):
        ExecutionConfig(max_repeats=1)

    # 7. Duplicate tools in list rejected
    with pytest.raises(ValidationError, match="Duplicate tools found"):
        AgentConfig(
            name="dup-tool-bot",
            system_prompt="Prompt",
            llm=LLMConfig(provider="openai", model="gpt-4o"),
            tools=["calculator", "current_time", "calculator"],
        )


def test_loader_file_errors(tmp_path: Path):
    """Verify empty file, non-mapping YAML, and syntax errors raise AgentConfigError with path."""
    # 1. Missing file
    missing = tmp_path / "ghost.yaml"
    with pytest.raises(AgentConfigError) as exc_info:
        load_agent_config(missing)
    assert str(missing) in str(exc_info.value)

    # 2. Empty file
    empty = tmp_path / "empty.yaml"
    empty.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(AgentConfigError) as exc_info:
        load_agent_config(empty)
    assert str(empty) in str(exc_info.value)
    assert "empty" in str(exc_info.value).lower()

    # 3. Non-mapping YAML (top level list or scalar)
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(AgentConfigError) as exc_info:
        load_agent_config(list_yaml)
    assert str(list_yaml) in str(exc_info.value)
    assert "mapping" in str(exc_info.value).lower()

    # 4. YAML syntax error
    syntax_err = tmp_path / "syntax.yaml"
    syntax_err.write_text("name: [unclosed list", encoding="utf-8")
    with pytest.raises(AgentConfigError) as exc_info:
        load_agent_config(syntax_err)
    assert str(syntax_err) in str(exc_info.value)
    assert "syntax" in str(exc_info.value).lower()


def test_scan_agents_dir(tmp_path: Path):
    """Verify scan_agents_dir finds .yaml/.yml, ignores others/subdirs, sorts, and checks duplicates."""
    # Create directory with .yaml, .yml, .txt, and a subdirectory
    (tmp_path / "b_agent.yaml").write_text(
        "name: b-bot\nllm:\n  provider: openai\n  model: gpt-4o\nsystem_prompt: prompt",
        encoding="utf-8",
    )
    (tmp_path / "a_agent.yml").write_text(
        "name: a-bot\nllm:\n  provider: openai\n  model: gpt-4o\nsystem_prompt: prompt",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "c_agent.yaml").write_text(
        "name: c-bot\nllm:\n  provider: openai\n  model: gpt-4o\nsystem_prompt: prompt",
        encoding="utf-8",
    )

    # Strict scan
    configs = scan_agents_dir(tmp_path, strict=True)
    assert len(configs) == 2
    assert "a-bot" in configs
    assert "b-bot" in configs
    # Non-recursive: c-bot in subdir is ignored
    assert "c-bot" not in configs

    # Deterministic alphabetical order by filename: a_agent.yml comes before b_agent.yaml
    names_in_order = list(configs.keys())
    assert names_in_order == ["a-bot", "b-bot"]

    # Duplicate agent name test
    (tmp_path / "duplicate.yaml").write_text(
        "name: a-bot\nllm:\n  provider: openai\n  model: gpt-4o\nsystem_prompt: prompt",
        encoding="utf-8",
    )
    with pytest.raises(AgentConfigError) as exc_info:
        scan_agents_dir(tmp_path, strict=True)
    assert "Duplicate agent name 'a-bot'" in str(exc_info.value)
    assert "a_agent.yml" in str(exc_info.value)
    assert "duplicate.yaml" in str(exc_info.value)

    # Missing directory check
    with pytest.raises(AgentConfigError, match="Directory not found"):
        scan_agents_dir(tmp_path / "non_existent")


def test_scan_agents_dir_strict_vs_non_strict(tmp_path: Path):
    """Verify strict=True raises one combined error while strict=False returns (configs, errors)."""
    (tmp_path / "valid.yaml").write_text(
        "name: valid-agent\nllm:\n  provider: openai\n  model: gpt-4o\nsystem_prompt: ok",
        encoding="utf-8",
    )
    (tmp_path / "broken1.yaml").write_text("name: 123broken", encoding="utf-8")
    (tmp_path / "broken2.yaml").write_text("not even yaml: [", encoding="utf-8")

    # strict=True: raises AgentConfigError listing all errors
    with pytest.raises(AgentConfigError) as exc_info:
        scan_agents_dir(tmp_path, strict=True)
    err_msg = str(exc_info.value)
    assert "broken1.yaml" in err_msg
    assert "broken2.yaml" in err_msg

    # strict=False: returns tuple of (configs, errors)
    configs, errors = scan_agents_dir(tmp_path, strict=False)
    assert len(configs) == 1
    assert "valid-agent" in configs
    assert len(errors) == 2


def test_build_engine_from_config():
    """Verify engine assembly, mapped attributes, tool filtering, and immutability of shared registry."""
    shared_registry = ToolRegistry()
    register_builtin_tools(shared_registry)
    initial_tool_count = len(shared_registry.list_specs())

    config = AgentConfig(
        name="math-bot",
        system_prompt="You are a math assistant.",
        llm=LLMConfig(provider="fake", model="fake-model", temperature=0.5, max_tokens=256),
        tools=["calculator"],
        execution=ExecutionConfig(max_iterations=7, timeout_seconds=42.0, max_repeats=5),
    )

    fake_llm = FakeLLMProvider()
    engine = build_engine_from_config(config, llm=fake_llm, registry=shared_registry)

    # 1. Assert on engine attributes
    assert isinstance(engine, ReActEngine)
    assert engine.system_prompt == "You are a math assistant."
    assert engine.max_iterations == 7
    assert engine.timeout_seconds == 42.0
    assert engine.max_consecutive_identical_calls == 5
    assert engine.llm is fake_llm

    # 2. Only whitelisted tools are visible in engine's registry
    assert engine.tools is not None
    assert engine.tools.has("calculator")
    assert not engine.tools.has("current_time")
    assert not engine.tools.has("mock_search")
    assert len(engine.tools.list_specs()) == 1

    # 3. Shared registry was not mutated
    assert len(shared_registry.list_specs()) == initial_tool_count
    assert shared_registry.has("current_time")
    assert shared_registry.has("mock_search")

    # 4. Unknown tool raises AgentConfigError with unknown and available names
    bad_config = AgentConfig(
        name="bad-tool-bot",
        system_prompt="Prompt",
        llm=LLMConfig(provider="fake", model="fake-model"),
        tools=["non_existent_tool"],
    )
    with pytest.raises(AgentConfigError) as exc_info:
        build_engine_from_config(bad_config, registry=shared_registry)
    err_msg = str(exc_info.value)
    assert "non_existent_tool" in err_msg
    assert "calculator" in err_msg


def test_end_to_end_engine_with_fake_llm():
    """Build an engine from config and run a multi-step query through to final answer."""
    shared_registry = ToolRegistry()
    register_builtin_tools(shared_registry)

    config = AgentConfig(
        name="math-solver",
        system_prompt="Solve math problems accurately.",
        llm=LLMConfig(provider="fake", model="fake-model"),
        tools=["calculator"],
        execution=ExecutionConfig(max_iterations=5),
    )

    fake_llm = FakeLLMProvider(
        responses=[
            [ToolCall(name="calculator", arguments={"expression": "15 * 6"})],
            "The calculation 15 * 6 equals 90.",
        ]
    )

    engine = build_engine_from_config(config, llm=fake_llm, registry=shared_registry)

    async def _run():
        final_answer, events = await engine.run("What is 15 * 6?")
        assert final_answer == "The calculation 15 * 6 equals 90."
        assert len(events) == 3
        assert events[0].type == "tool_call"
        assert events[1].type == "observation"
        assert events[1].tool_result is not None
        assert events[1].tool_result.output == "90"
        assert events[2].type == "final"

    asyncio.run(_run())


def test_examples_cannot_go_stale():
    """Verify all real YAML files in agents/ directory parse and build clean engines with default_registry."""
    agents_dir = Path(__file__).resolve().parent.parent / "agents"
    assert agents_dir.is_dir(), f"agents directory not found at {agents_dir}"

    configs = scan_agents_dir(agents_dir, strict=True)
    assert len(configs) >= 2, f"Expected at least 2 example agents in {agents_dir}, found {len(configs)}"

    assert "researcher" in configs
    assert "math-tutor" in configs

    fake_llm = FakeLLMProvider()

    for name, config in configs.items():
        assert isinstance(config, AgentConfig)
        assert config.name == name
        assert len(config.system_prompt) > 0
        # Build engine using the real default_registry
        engine = build_engine_from_config(config, llm=fake_llm, registry=default_registry)
        assert isinstance(engine, ReActEngine)
        assert engine.tools is not None
        # All tools declared in YAML must exist in the assembled engine
        for tool_name in config.tools:
            assert engine.tools.has(tool_name)
