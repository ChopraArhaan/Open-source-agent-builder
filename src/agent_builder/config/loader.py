"""YAML configuration loader and ReActEngine builder."""

from pathlib import Path
from typing import Any, Literal, overload

import yaml
from pydantic import ValidationError

from src.agent_builder.config.schema import AgentConfig
from src.agent_builder.engine.react import ReActEngine
from src.agent_builder.llm import create_llm_provider
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.tools import default_registry
from src.agent_builder.tools.registry import ToolRegistry


class AgentConfigError(Exception):
    """Raised when an agent configuration file is invalid, missing, or malformed."""

    def __init__(self, message: str, path: str | Path | None = None) -> None:
        self.path = str(path) if path is not None else None
        if self.path and self.path not in message:
            full_message = f"[{self.path}] {message}"
        else:
            full_message = message
        super().__init__(full_message)


def load_agent_config(path: str | Path) -> AgentConfig:
    """Load and validate an AgentConfig from a YAML file.

    Raises:
        AgentConfigError: For missing files, empty documents, non-mapping root,
                          YAML syntax errors, or schema validation failures.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise AgentConfigError(f"Configuration file not found: '{file_path}'", path=file_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise AgentConfigError(f"Failed to read file: {e}", path=file_path) from e

    if not content.strip():
        raise AgentConfigError("Configuration file is empty", path=file_path)

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise AgentConfigError(f"YAML syntax error: {e}", path=file_path) from e

    if not isinstance(data, dict):
        raise AgentConfigError(
            f"Expected top-level YAML mapping, got {type(data).__name__}",
            path=file_path,
        )

    try:
        return AgentConfig.model_validate(data)
    except ValidationError as e:
        field_errors: list[str] = []
        for err in e.errors():
            loc = " -> ".join(str(item) for item in err["loc"])
            field_errors.append(f"{loc}: {err['msg']}")
        summary = "; ".join(field_errors)
        raise AgentConfigError(
            f"Schema validation failed: {summary}",
            path=file_path,
        ) from e


@overload
def scan_agents_dir(
    directory: str | Path,
    *,
    strict: Literal[True] = ...,
) -> dict[str, AgentConfig]: ...


@overload
def scan_agents_dir(
    directory: str | Path,
    *,
    strict: Literal[False],
) -> tuple[dict[str, AgentConfig], list[AgentConfigError]]: ...


@overload
def scan_agents_dir(
    directory: str | Path,
    *,
    strict: bool,
) -> dict[str, AgentConfig] | tuple[dict[str, AgentConfig], list[AgentConfigError]]: ...


def scan_agents_dir(
    directory: str | Path,
    *,
    strict: bool = True,
) -> dict[str, AgentConfig] | tuple[dict[str, AgentConfig], list[AgentConfigError]]:
    """Scan a directory for agent YAML configs (*.yaml, *.yml).

    Non-recursive, sorted by filename for deterministic ordering.

    Args:
        directory: Path to directory containing agent YAML files.
        strict: If True, raises AgentConfigError listing all failures if any file
                is invalid or duplicate names are found.
                If False, returns a tuple of (valid_configs, error_list) where
                valid_configs is dict[str, AgentConfig] and error_list is list[AgentConfigError].

    Returns:
        If strict=True: dict[str, AgentConfig] mapping agent name to config.
        If strict=False: tuple[dict[str, AgentConfig], list[AgentConfigError]].

    Raises:
        AgentConfigError: If directory does not exist, or if strict=True and any
                          validation / duplicate errors occur.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise AgentConfigError(f"Directory not found: '{dir_path}'", path=dir_path)

    # Collect non-recursive *.yaml and *.yml files, sorted by filename
    yaml_files = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in (".yaml", ".yml")],
        key=lambda p: p.name,
    )

    configs: dict[str, AgentConfig] = {}
    name_to_file: dict[str, Path] = {}
    errors: list[AgentConfigError] = []

    for file_path in yaml_files:
        try:
            config = load_agent_config(file_path)
            # Check for duplicate agent names
            if config.name in configs:
                first_file = name_to_file[config.name]
                dup_err = AgentConfigError(
                    f"Duplicate agent name '{config.name}' found in '{first_file}' and '{file_path}'",
                    path=file_path,
                )
                errors.append(dup_err)
            else:
                configs[config.name] = config
                name_to_file[config.name] = file_path
        except AgentConfigError as err:
            errors.append(err)

    if strict and errors:
        error_lines = "\n".join(f"  - {err}" for err in errors)
        raise AgentConfigError(
            f"Failed to scan agent directory '{dir_path}' with {len(errors)} error(s):\n{error_lines}",
            path=dir_path,
        )

    if not strict:
        return configs, errors

    return configs


def build_engine_from_config(
    config: AgentConfig,
    *,
    llm: BaseLLMProvider | None = None,
    registry: ToolRegistry | None = None,
) -> ReActEngine:
    """Build a configured ReActEngine from an AgentConfig specification.

    Args:
        config: Validated AgentConfig.
        llm: Optional pre-built LLM provider. If omitted, built via factory from config.llm.
        registry: Optional shared ToolRegistry. If omitted, default_registry is used.

    Returns:
        A ready-to-run ReActEngine instance.

    Raises:
        AgentConfigError: If any configured tools are not present in the registry.
    """
    effective_registry = registry if registry is not None else default_registry

    # 1. Validate tools against registry
    available_tools = sorted(effective_registry._tools.keys())
    unknown_tools = [tool for tool in config.tools if not effective_registry.has(tool)]
    if unknown_tools:
        raise AgentConfigError(
            f"Unknown tool(s) {unknown_tools} configured for agent '{config.name}'. "
            f"Available tools: {available_tools}"
        )

    # 2. Build filtered registry containing only whitelisted tools (without mutating original)
    filtered_registry = ToolRegistry()
    for tool_name in config.tools:
        filtered_registry._tools[tool_name] = effective_registry._tools[tool_name]
        filtered_registry._specs[tool_name] = effective_registry._specs[tool_name]
        filtered_registry._schemas[tool_name] = effective_registry._schemas[tool_name]
        filtered_registry._raw_funcs[tool_name] = effective_registry._raw_funcs[tool_name]

    # 3. Build or use provided LLM provider
    if llm is not None:
        effective_llm = llm
    else:
        llm_kwargs: dict[str, Any] = {
            "temperature": config.llm.temperature,
        }
        if config.llm.max_tokens is not None:
            llm_kwargs["max_tokens"] = config.llm.max_tokens

        effective_llm = create_llm_provider(
            provider=config.llm.provider,
            model=config.llm.model,
            **llm_kwargs,
        )

    # 4. Assemble ReActEngine with mapped execution settings
    return ReActEngine(
        llm=effective_llm,
        tools=filtered_registry,
        system_prompt=config.system_prompt,
        max_iterations=config.execution.max_iterations,
        timeout_seconds=config.execution.timeout_seconds,
        max_consecutive_identical_calls=config.execution.max_repeats,
    )
