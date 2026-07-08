from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.events import EventEmitter
from propab.types import EventType

logger = logging.getLogger(__name__)

# Status codes worth retrying: rate limits and transient server-side failures.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# Providers the client can actually dispatch to. Anything else (a typo, "anthropic",
# "claude", …) is a misconfiguration, not a runnable request.
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "gemini", "ollama"})
# Providers that require a non-empty API key. Ollama is a local server needing none.
_KEY_REQUIRED_PROVIDERS = frozenset({"openai", "gemini"})


class LLMConfigError(RuntimeError):
    """Raised when the LLM client is misconfigured (unknown provider or missing key).

    A config error must FAIL LOUD: never fall back to a fabricated placeholder
    response, which downstream would treat as a real model answer and quietly
    "research" one canned claim across every domain while looking healthy.
    """


def _normalize_provider(provider: str | None) -> str:
    """Lower-cased, stripped provider name; empty defaults to 'openai'."""
    return (provider or "openai").strip().lower()


def _validate_llm_config(provider: str | None, api_key: str | None) -> str:
    """Validate provider + key, returning the normalized provider or raising.

    Raises LLMConfigError for an unsupported provider or a supported provider
    that requires a key but was given an empty one.
    """
    prov = _normalize_provider(provider)
    if prov not in SUPPORTED_LLM_PROVIDERS:
        raise LLMConfigError(
            f"unsupported LLM provider {provider!r} "
            f"(supported: {', '.join(sorted(SUPPORTED_LLM_PROVIDERS))})"
        )
    if prov in _KEY_REQUIRED_PROVIDERS and not (api_key or "").strip():
        raise LLMConfigError(f"missing API key for provider {prov!r}")
    return prov


def _is_transient_llm_error(exc: Exception) -> bool:
    """True for network/timeout/rate-limit errors that a retry can plausibly fix."""
    # Config errors are never transient — do not waste retries hiding a misconfig.
    if isinstance(exc, LLMConfigError):
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        emitter: EventEmitter,
        session_factory: async_sessionmaker,
    ) -> None:
        # Fail loud on misconfiguration at construction time rather than silently
        # fabricating a placeholder hypothesis on first call.
        self.provider = _validate_llm_config(provider, api_key)
        self.model = model
        self.api_key = api_key
        self.emitter = emitter
        self.session_factory = session_factory

    async def call(
        self,
        *,
        prompt: str,
        purpose: str,
        session_id: str,
        hypothesis_id: str | None = None,
    ) -> str:
        # A per-call uuid lets the UI pair each llm.prompt with its exact
        # llm.response instead of relying on fragile FIFO ordering.
        call_id = str(uuid4())
        started = time.perf_counter()
        await self.emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_PROMPT,
            step=f"llm.{purpose}",
            payload={"prompt": prompt, "purpose": purpose, "model": self.model, "call_id": call_id},
            hypothesis_id=hypothesis_id,
        )

        # Providers fill ``usage`` in-place with tokens_in / tokens_out when they
        # report them; it stays a per-call dict so concurrent calls never race.
        usage: dict[str, Any] = {}
        response_text = await self._call_provider(prompt, usage_out=usage)

        duration_ms = int((time.perf_counter() - started) * 1000)
        tokens_in = usage.get("tokens_in")
        tokens_out = usage.get("tokens_out")
        await self.emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_RESPONSE,
            step=f"llm.{purpose}",
            payload={
                "response": response_text,
                "purpose": purpose,
                "model": self.model,
                "call_id": call_id,
                "duration_ms": duration_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
            hypothesis_id=hypothesis_id,
        )

        await self._persist_call(
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            purpose=purpose,
            prompt=prompt,
            response=response_text,
            duration_ms=duration_ms,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
        )
        return response_text

    async def _call_provider(self, prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        """Dispatch to the provider with bounded exponential backoff on transient errors.

        A single LLM timeout used to crash entire campaigns; retrying transient
        failures (timeouts, connection resets, 429/5xx) keeps long runs alive.
        """
        max_retries = max(0, int(getattr(settings, "llm_max_retries", 3)))
        base_delay = float(getattr(settings, "llm_retry_base_delay_sec", 2.0))
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                # Only thread usage_out through when a caller asked for it, so a
                # test that monkeypatches _call_provider_once(prompt) still works.
                if usage_out is not None:
                    return await self._call_provider_once(prompt, usage_out=usage_out)
                return await self._call_provider_once(prompt)
            except Exception as exc:  # noqa: BLE001 — classify then re-raise non-transient
                if not _is_transient_llm_error(exc) or attempt >= max_retries:
                    raise
                last_exc = exc
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Transient LLM error (%s: %s); retry %d/%d in %.1fs",
                    type(exc).__name__, exc, attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
        # Unreachable: loop either returns or raises, but keep type-checkers happy.
        raise last_exc if last_exc else RuntimeError("LLM call failed without an exception")

    @staticmethod
    def _record_usage(usage_out: dict[str, Any] | None, tokens_in: Any, tokens_out: Any) -> None:
        """Stash token counts into the caller's per-call usage dict (best-effort)."""
        if usage_out is None:
            return
        usage_out["tokens_in"] = int(tokens_in) if isinstance(tokens_in, (int, float)) and not isinstance(tokens_in, bool) else None
        usage_out["tokens_out"] = int(tokens_out) if isinstance(tokens_out, (int, float)) and not isinstance(tokens_out, bool) else None

    async def _call_provider_once(self, prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        # Provider/key are validated in __init__; re-validate defensively so a
        # mutated client (or a subclass) can never fall through to a placeholder.
        prov = _validate_llm_config(self.provider, self.api_key)
        if prov == "ollama":
            return await self._ollama_chat(prompt, usage_out=usage_out)
        if prov == "gemini":
            return await self._gemini_generate_content(prompt, usage_out=usage_out)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        tout = float(getattr(settings, "llm_http_timeout_sec", 180.0))
        timeout = httpx.Timeout(tout, connect=min(60.0, tout))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
        u = body.get("usage") or {}
        self._record_usage(usage_out, u.get("prompt_tokens"), u.get("completion_tokens"))
        return body["choices"][0]["message"]["content"]

    async def _ollama_chat(self, prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        base = (getattr(settings, "ollama_base_url", None) or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        tout = max(120.0, float(getattr(settings, "llm_http_timeout_sec", 180.0)))
        async with httpx.AsyncClient(timeout=httpx.Timeout(tout, connect=min(60.0, tout))) as client:
            response = await client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()
        self._record_usage(usage_out, body.get("prompt_eval_count"), body.get("eval_count"))
        msg = body.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return json.dumps(body)[:8000]

    async def _gemini_generate_content(self, prompt: str, *, usage_out: dict[str, Any] | None = None) -> str:
        if not (self.api_key or "").strip():
            raise LLMConfigError("missing API key for provider 'gemini'")
        mid = (self.model or "gemini-2.0-flash").strip()
        if mid.startswith("models/"):
            mid = mid[len("models/") :]
        key = (self.api_key or "").strip()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent"
        params = {"key": key}
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }
        tout = max(120.0, float(getattr(settings, "llm_http_timeout_sec", 180.0)))
        async with httpx.AsyncClient(timeout=httpx.Timeout(tout, connect=min(60.0, tout))) as client:
            response = await client.post(url, params=params, json=payload)
        response.raise_for_status()
        body = response.json()
        um = body.get("usageMetadata") or {}
        self._record_usage(usage_out, um.get("promptTokenCount"), um.get("candidatesTokenCount"))
        cands = body.get("candidates") or []
        if not cands:
            return json.dumps(body)[:8000]
        parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
        texts: list[str] = []
        for part in parts:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        out = "".join(texts).strip()
        return out if out else json.dumps(body)[:8000]

    async def _persist_call(
        self,
        *,
        session_id: str,
        hypothesis_id: str | None,
        purpose: str,
        prompt: str,
        response: str,
        duration_ms: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO llm_calls (
                        id, session_id, hypothesis_id, call_purpose, model, prompt_text, response_text, input_tokens, output_tokens, duration_ms
                    )
                    VALUES (
                        :id, :session_id, :hypothesis_id, :call_purpose, :model, :prompt_text, :response_text, :input_tokens, :output_tokens, :duration_ms
                    )
                    """
                ),
                {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "hypothesis_id": hypothesis_id,
                    "call_purpose": purpose,
                    "model": self.model,
                    "prompt_text": prompt,
                    "response_text": response,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_ms": duration_ms,
                },
            )
            await session.commit()
