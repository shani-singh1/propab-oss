"""CFG5: embedding-model default must be provider-appropriate.

The shipped ``embed_model`` default (``text-embedding-3-small``) is an OpenAI-only
id. A deployment that sets ``EMBED_PROVIDER=gemini`` but leaves ``EMBED_MODEL`` at
its default would otherwise send the OpenAI id to Google's embeddings endpoint,
400, and silently fall back to non-embedding ranking. Resolution fixes that.
"""

import asyncio

import pytest

from propab.config import Settings
from propab.embeddings import (
    _GOOGLE_DEFAULT_EMBED_MODEL,
    _OPENAI_DEFAULT_EMBED_MODEL,
    embed_texts,
    resolve_embed_model,
)


# --- resolve_embed_model ---------------------------------------------------

def test_gemini_default_resolves_to_google_model() -> None:
    resolved = resolve_embed_model("gemini", _OPENAI_DEFAULT_EMBED_MODEL)
    assert resolved == _GOOGLE_DEFAULT_EMBED_MODEL
    assert resolved != _OPENAI_DEFAULT_EMBED_MODEL


def test_google_default_resolves_to_google_model() -> None:
    assert resolve_embed_model("google", _OPENAI_DEFAULT_EMBED_MODEL) == _GOOGLE_DEFAULT_EMBED_MODEL


def test_openai_default_is_unchanged() -> None:
    assert resolve_embed_model("openai", _OPENAI_DEFAULT_EMBED_MODEL) == _OPENAI_DEFAULT_EMBED_MODEL


def test_explicit_google_override_is_respected() -> None:
    # An operator who explicitly sets a Google embed id must not be overridden.
    assert resolve_embed_model("gemini", "text-embedding-004") == "text-embedding-004"


def test_explicit_openai_override_is_respected() -> None:
    assert resolve_embed_model("openai", "text-embedding-3-large") == "text-embedding-3-large"


# --- Settings.effective_embed_model ---------------------------------------

def test_settings_effective_embed_model_gemini() -> None:
    # Pass the OpenAI cross-provider default EXPLICITLY (not via the ambient .env,
    # which a real deployment may set) so this exercises the resolution: gemini
    # provider + OpenAI default model -> Google default.
    s = Settings(embed_provider="gemini", embed_model=_OPENAI_DEFAULT_EMBED_MODEL)
    assert s.effective_embed_model == _GOOGLE_DEFAULT_EMBED_MODEL


def test_settings_effective_embed_model_openai_unchanged() -> None:
    # Explicit OpenAI default so the assertion is independent of any ambient
    # EMBED_MODEL in a deployment .env.
    s = Settings(embed_provider="openai", embed_model=_OPENAI_DEFAULT_EMBED_MODEL)
    assert s.effective_embed_model == _OPENAI_DEFAULT_EMBED_MODEL


# --- embed_texts backstop (resolution takes effect at the call layer) ------

def test_embed_texts_routes_google_default_to_google_endpoint(monkeypatch) -> None:
    """embed_texts, called with the raw settings default under a Google provider,
    must resolve the model before dispatching to the Google embedder."""
    seen: dict[str, str] = {}

    async def fake_google(*, texts, api_key, model):
        seen["model"] = model
        return [[0.0]] * len(texts)

    async def fake_openai(*, texts, api_key, model):  # pragma: no cover - must not run
        raise AssertionError("Google provider must not hit the OpenAI embedder")

    monkeypatch.setattr("propab.embeddings._google_embed", fake_google)
    monkeypatch.setattr("propab.embeddings._openai_embed", fake_openai)

    out = asyncio.run(
        embed_texts(
            texts=["hello"],
            api_key="AIzaKey",
            model=_OPENAI_DEFAULT_EMBED_MODEL,  # raw settings default
            provider="gemini",
        )
    )
    assert out == [[0.0]]
    assert seen["model"] == _GOOGLE_DEFAULT_EMBED_MODEL


def test_embed_texts_openai_happy_path_unchanged(monkeypatch) -> None:
    seen: dict[str, str] = {}

    async def fake_openai(*, texts, api_key, model):
        seen["model"] = model
        return [[1.0]] * len(texts)

    monkeypatch.setattr("propab.embeddings._openai_embed", fake_openai)

    out = asyncio.run(
        embed_texts(
            texts=["hi"],
            api_key="sk-x",
            model=_OPENAI_DEFAULT_EMBED_MODEL,
            provider="openai",
        )
    )
    assert out == [[1.0]]
    assert seen["model"] == _OPENAI_DEFAULT_EMBED_MODEL
