"""Built-in tools: hardened safe calculator, current time, and mock search."""

import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.agent_builder.tools.registry import ToolRegistry

# Safe operators allowed in the AST calculator
_SAFE_OPERATORS: dict[type[ast.AST], Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Whitelist of safe math functions
_SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}

MAX_EXPRESSION_LENGTH = 200
MAX_BASE_VALUE = 10_000
MAX_EXPONENT_VALUE = 100


def _eval_ast_node(node: ast.AST) -> int | float:
    """Recursively evaluates an AST node with strict type and limit checks."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    if isinstance(node, ast.BinOp):
        bin_op_type = type(node.op)
        if bin_op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {bin_op_type.__name__}")

        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        # Division by zero check
        if bin_op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero")

        # Exponentiation hardening against DoS / memory exhaustion
        if bin_op_type is ast.Pow:
            if abs(left) > MAX_BASE_VALUE:
                raise ValueError(
                    f"Base {left} exceeds maximum allowed base limit of {MAX_BASE_VALUE}"
                )
            if abs(right) > MAX_EXPONENT_VALUE:
                raise ValueError(
                    f"Exponent {right} exceeds maximum allowed exponent limit of {MAX_EXPONENT_VALUE}"
                )
            result = _SAFE_OPERATORS[bin_op_type](left, right)
            if isinstance(result, complex):
                raise ValueError("Complex numbers are not supported")
            return result

        return _SAFE_OPERATORS[bin_op_type](left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            func_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else type(node.func).__name__
            )
            raise ValueError(f"Function '{func_name}' is not permitted")

        fn = _SAFE_FUNCTIONS[node.func.id]
        args = [_eval_ast_node(arg) for arg in node.args]

        if node.func.id == "sqrt":
            if len(args) != 1:
                raise ValueError("sqrt requires exactly 1 argument")
            if args[0] < 0:
                raise ValueError("Cannot calculate square root of a negative number")

        return fn(*args)

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def evaluate_expression(expression: str) -> str:
    """Safely evaluate an arithmetic expression using AST parsing. Never uses eval()."""
    cleaned = expression.strip()
    if not cleaned:
        raise ValueError("Expression cannot be empty")

    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        parsed_ast = ast.parse(cleaned, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid mathematical syntax: {e.msg}") from e

    result = _eval_ast_node(parsed_ast)
    # Format integers cleanly without trailing .0
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


# --- Tool Schemas & Definitions ---


class CalculatorInput(BaseModel):
    """Input parameters for the calculator tool."""
    expression: str = Field(
        description="Mathematical expression to evaluate (e.g. '2 + 2 * 3', 'sqrt(16) + 10')"
    )


def calculator(expression: str) -> str:
    """Perform safe mathematical calculations without eval."""
    return evaluate_expression(expression)


class CurrentTimeInput(BaseModel):
    """Input parameters for the current time tool."""
    timezone_name: str = Field(
        default="UTC",
        description="Timezone name (defaults to UTC)"
    )


def current_time(timezone_name: str = "UTC") -> str:
    """Return the current date and time in ISO 8601 format."""
    now_utc = datetime.now(timezone.utc)
    return f"{now_utc.isoformat()} ({timezone_name})"


class MockSearchInput(BaseModel):
    """Input parameters for the mock search tool."""
    query: str = Field(description="Search terms or query string")
    max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of results to return"
    )


async def mock_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Deterministic mock search engine for testing and local exploration."""
    lowered = query.lower()
    canned_database: list[dict[str, Any]] = [
        {
            "title": "Open Source Agent Builder Guide",
            "snippet": "A lightweight, transparent engine for autonomous AI agents built with Python and FastAPI.",
            "url": "https://github.com/agent-builder/guide",
            "keywords": ["agent", "builder", "fastapi", "react", "python"],
        },
        {
            "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
            "snippet": "ReAct prompt paradigm interleaves reasoning traces and action steps for LLM problem solving.",
            "url": "https://arxiv.org/abs/2210.03629",
            "keywords": ["react", "reasoning", "acting", "prompt", "paper"],
        },
        {
            "title": "Pydantic V2 Documentation",
            "snippet": "Data validation and settings management using Python type annotations.",
            "url": "https://docs.pydantic.dev",
            "keywords": ["pydantic", "schema", "validation", "types"],
        },
        {
            "title": "FastAPI Web Framework",
            "snippet": "High performance modern Python web framework for building APIs with automatic OpenAPI docs.",
            "url": "https://fastapi.tiangolo.com",
            "keywords": ["fastapi", "rest", "api", "server", "sse"],
        },
    ]

    matched: list[dict[str, str]] = []
    for entry in canned_database:
        entry_title = str(entry["title"])
        entry_keywords: list[str] = entry.get("keywords", [])
        if any(kw in lowered for kw in entry_keywords) or any(word in entry_title.lower() for word in lowered.split()):
            matched.append({
                "title": entry_title,
                "snippet": str(entry["snippet"]),
                "url": str(entry["url"]),
            })

    if not matched:
        matched.append({
            "title": f"No direct matches found for '{query}'",
            "snippet": f"Simulated search results for query: {query}",
            "url": f"https://mocksearch.local/search?q={query}",
        })

    return matched[:max_results]


def register_builtin_tools(registry: ToolRegistry) -> None:
    """Register all built-in tools to the given registry."""
    registry.register(
        name="calculator",
        description="Safely evaluate mathematical expressions. Supports +, -, *, /, //, %, **, sqrt, abs, round, min, max.",
        args_schema=CalculatorInput,
        func=calculator,
        timeout_seconds=5.0,
    )
    registry.register(
        name="current_time",
        description="Get current ISO-8601 timestamp in UTC.",
        args_schema=CurrentTimeInput,
        func=current_time,
        timeout_seconds=5.0,
    )
    registry.register(
        name="mock_search",
        description="Search documentation and web data with mock results.",
        args_schema=MockSearchInput,
        func=mock_search,
        timeout_seconds=10.0,
    )
