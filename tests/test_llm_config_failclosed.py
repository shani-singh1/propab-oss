"""LLM client must fail loud on misconfiguration (CFG2/CFG3).

Previously an unsupported provider (e.g. "anthropic", a typo) or a supported
provider with an empty API key silently returned a hardcoded placeholder
hypothesis, which downstream treated as a real model answer. A misconfigured
deployment would then "research" one canned claim across every domain while
looking healthy. These tests pin the fail-closed contract: misconfiguration
raises LLMConfigError and no placeholder text is ever fabricated.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from propab.llm import (
    LLMClient,
    LLMConfigError,
    SUPPORTED_LLM_PROVIDERS,
    _is_transient_llm_error,
    _validate_llm_config,
)

# The exact fabricated string that used to leak out of a misconfigured client.
_PLACEHOLDER = "A controlled intervention can improve measurable outcome quality."


def _construct(provider: str, api_key: str) -> LLMClient:
    """Construct a real LLMClient (exercising __init__ validation)."""
    return LLMClient(
        provider=provider,
        model="test-model",
        api_key=api_key,
        emitter=object(),          # unused before any call
        session_factory=object(),  # unused before any call
    )


def test_unknown_provider_raises_on_construction() -> None:
    with pytest.raises(LLMConfigError):
        _construct("anthropic", "sk-whatever")


def test_typo_provider_raises_on_construction() -> None:
    with pytest.raises(LLMConfigError):
        _construct("opnai", "sk-whatever")


def test_openai_empty_key_raises_on_construction() -> None:
    with pytest.raises(LLMConfigError):
        _construct("openai", "")


def test_gemini_empty_key_raises_on_construction() -> None:
    with pytest.raises(LLMConfigError):
        _construct("gemini", "   ")  # whitespace-only counts as empty


def test_ollama_needs_no_key() -> None:
    # Ollama is a local server; an empty key is valid and must not raise.
    client = _construct("ollama", "")
    assert client.provider == "ollama"


def test_supported_providers_with_keys_construct() -> None:
    assert _construct("openai", "sk-x").provider == "openai"
    assert _construct("gemini", "AIza-x").provider == "gemini"


def test_validate_helper_contract() -> None:
    assert _validate_llm_config("OpenAI", "sk-x") == "openai"  # normalized
    assert _validate_llm_config(None, "sk-x") == "openai"      # default
    assert _validate_llm_config("ollama", "") == "ollama"
    with pytest.raises(LLMConfigError):
        _validate_llm_config("claude", "sk-x")
    with pytest.raises(LLMConfigError):
        _validate_llm_config("openai", "")


def test_config_error_is_not_transient() -> None:
    # Must never be retried/hidden by the backoff wrapper.
    assert _is_transient_llm_error(LLMConfigError("bad")) is False


def test_config_error_propagates_through_retry_wrapper() -> None:
    # A config error surfaced from _call_provider_once must propagate immediately,
    # not be swallowed as a transient retry.
    client = _construct("openai", "sk-x")

    async def boom(_prompt: str) -> str:
        raise LLMConfigError("missing API key for provider 'openai'")

    client._call_provider_once = boom  # type: ignore[assignment]
    with pytest.raises(LLMConfigError):
        asyncio.run(client._call_provider("hi"))


def test_supported_set_is_exactly_the_three_dispatchable_providers() -> None:
    assert SUPPORTED_LLM_PROVIDERS == frozenset({"openai", "gemini", "ollama"})


def test_placeholder_string_absent_from_core_package() -> None:
    # The fabricated hypothesis must exist nowhere in the shippable code path.
    core = Path(__file__).resolve().parents[1] / "packages" / "propab-core" / "propab"
    assert core.is_dir(), core
    for py in core.rglob("*.py"):
        assert _PLACEHOLDER not in py.read_text(encoding="utf-8"), py
