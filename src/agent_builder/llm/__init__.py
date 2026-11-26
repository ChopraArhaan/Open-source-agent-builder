"""LLM provider interfaces and implementations."""

from typing import Any

from src.agent_builder.llm.anthropic_provider import AnthropicProvider
from src.agent_builder.llm.base import BaseLLMProvider
from src.agent_builder.llm.fake import FakeLLMProvider
from src.agent_builder.llm.gemini_provider import GeminiProvider
from src.agent_builder.llm.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "FakeLLMProvider",
    "GeminiProvider",
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

    if normalized == "anthropic":
        return AnthropicProvider(model=model, **kwargs)

    if normalized in ("gemini", "google"):
        return GeminiProvider(model=model, **kwargs)

    if normalized == "fake":
        return FakeLLMProvider(model=model, **kwargs)

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. Supported providers: 'openai', 'anthropic', 'gemini', 'fake'."
    )
