"""LLM provider interfaces and implementations."""

from typing import Any

from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.llm.openai import OpenAIProvider

__all__ = [
    "BaseLLMProvider",
    "FakeLLMProvider",
    "OpenAIProvider",
    "create_llm_provider",
]


def create_llm_provider(
    provider: str,
    model: str,
    **kwargs: Any,
) -> BaseLLMProvider:
    """Factory function to instantiate an LLM provider."""
    normalized = provider.strip().lower()

    if normalized in ("openai", "azure_openai"):
        return OpenAIProvider(model=model, **kwargs)

    if normalized == "fake":
        return FakeLLMProvider(model=model, **kwargs)

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported providers: 'openai', 'fake'."
    )
