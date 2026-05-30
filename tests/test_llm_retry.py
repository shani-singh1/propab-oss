"""LLM transient-error classification and retry behavior.

A single timeout used to crash whole campaigns; transient errors must be retried
with backoff, while non-transient errors propagate immediately.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from propab.llm import LLMClient, _is_transient_llm_error


def _make_client() -> LLMClient:
    # Bypass __init__ side effects — we only exercise _call_provider's retry loop.
    c = LLMClient.__new__(LLMClient)
    c.provider = "openai"
    c.model = "test"
    c.api_key = "k"
    return c


def test_is_transient_classification() -> None:
    assert _is_transient_llm_error(httpx.ReadTimeout("slow"))
    assert _is_transient_llm_error(httpx.ConnectError("down"))

    resp429 = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    assert _is_transient_llm_error(httpx.HTTPStatusError("rate", request=resp429.request, response=resp429))

    resp400 = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    assert not _is_transient_llm_error(httpx.HTTPStatusError("bad", request=resp400.request, response=resp400))

    assert not _is_transient_llm_error(ValueError("logic bug"))


def test_retry_recovers_after_transient(monkeypatch) -> None:
    monkeypatch.setattr("propab.llm.settings.llm_max_retries", 3, raising=False)
    monkeypatch.setattr("propab.llm.settings.llm_retry_base_delay_sec", 0.0, raising=False)

    calls = {"n": 0}

    async def flaky(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("slow")
        return "ok"

    c = _make_client()
    monkeypatch.setattr(c, "_call_provider_once", flaky)

    out = asyncio.run(c._call_provider("hi"))
    assert out == "ok"
    assert calls["n"] == 3


def test_retry_exhausts_then_raises(monkeypatch) -> None:
    monkeypatch.setattr("propab.llm.settings.llm_max_retries", 2, raising=False)
    monkeypatch.setattr("propab.llm.settings.llm_retry_base_delay_sec", 0.0, raising=False)

    calls = {"n": 0}

    async def always_timeout(_prompt: str) -> str:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    c = _make_client()
    monkeypatch.setattr(c, "_call_provider_once", always_timeout)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(c._call_provider("hi"))
    assert calls["n"] == 3  # initial + 2 retries


def test_non_transient_not_retried(monkeypatch) -> None:
    monkeypatch.setattr("propab.llm.settings.llm_max_retries", 5, raising=False)
    monkeypatch.setattr("propab.llm.settings.llm_retry_base_delay_sec", 0.0, raising=False)

    calls = {"n": 0}

    async def logic_error(_prompt: str) -> str:
        calls["n"] += 1
        raise ValueError("not transient")

    c = _make_client()
    monkeypatch.setattr(c, "_call_provider_once", logic_error)

    with pytest.raises(ValueError):
        asyncio.run(c._call_provider("hi"))
    assert calls["n"] == 1  # no retries for non-transient
