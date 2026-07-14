"""Evolve — the mutation client (the model seam).

Deliberately NOT `propab.llm.LLMClient`. That one is coupled to an EventEmitter and a DB
session_factory and persists every call — correct for a campaign that makes tens of calls, fatal for
a search loop that makes millions. This client does one thing: prompt in, text out.

It is **sync and thread-safe** on purpose. `ParallelExecutor` runs N workers, each of which spends
almost all of its time blocked on the model (measured: the sandbox does ~430 evals/sec while a
mutation costs ~1s — a ~400x gap, so the LLM is the bottleneck and the sandbox has headroom to
spare). Blocking calls in threads therefore parallelize exactly as well as async here, with none of
the plumbing. Scale concurrency, not sandboxes: `workers=64, pool_size=8`.

The model is a COMMODITY in this design — a mutation operator over code, nothing more. FunSearch beat
state of the art on cap sets with a non-frontier model because the *verifier* carries the result. So
the default here is the cheap fast model, and a frontier model is a ceiling-raiser you can swap in,
never a prerequisite.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.5-flash"
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The model could not be reached. The engine treats this as a dead mutation, not a crash."""


class GeminiMutationClient:
    """Minimal sync Gemini client satisfying `mutator.LLMClient` (`complete(prompt) -> str`).

    Thread-safe: `httpx.Client` is safe to share across threads, and the only mutable state is a
    counter behind a lock. One connection pool is shared by all workers, which is what keeps 64
    concurrent mutations from opening 64 TLS handshakes.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.9,
        timeout_s: float = 120.0,
        max_retries: int = 4,
        max_output_tokens: int | None = 8192,
        thinking_budget: int | None = 0,
    ) -> None:
        key = (api_key or os.getenv("GOOGLE_API_KEY") or _settings_value("google_api_key")).strip()
        if not key:
            raise LLMError(
                "no API key: set GOOGLE_API_KEY in the environment or .env "
                "(the mutation client has no model to call)"
            )
        self._key = key
        self.model = (
            model or os.getenv("LLM_MODEL") or _settings_value("llm_model") or DEFAULT_MODEL
        ).strip()
        if self.model.startswith("models/"):
            self.model = self.model[len("models/") :]
        # High temperature ON PURPOSE. This is a mutation operator: a deterministic one collapses the
        # population onto a single lineage, and the whole point of islands + families is to keep
        # exploring. Diversity is the product here, not a defect.
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        # THINKING OFF, and this is load-bearing. Gemini 3.x flash reasons by default and bills those
        # thought tokens against maxOutputTokens. On our ~9.6k-char mutation prompt it spent 3,928 of
        # 4,096 tokens thinking, left 164 for code, and returned a function truncated mid-body
        # (finishReason=MAX_TOKENS). extract_program() then found nothing runnable and fell back to
        # the no-op — so nearly every child emitted ZERO candidates and the search was mostly empty.
        # Measured on the real prompt: thinking on -> 0 candidates; thinking off -> 350.
        # A mutation operator does not need to reason. It needs to write code.
        self.thinking_budget = thinking_budget

        # Auth goes in a HEADER, never a query param. httpx logs the request URL at INFO, so
        # `?key=...` would print the secret into every campaign log (it did — caught in the first
        # live run). A header keeps it out of logs, tracebacks, and proxy access logs.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(30.0, timeout_s)),
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=64),
            headers={"x-goog-api-key": self._key},
        )
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.retries = 0

    # ------------------------------------------------------------------ api
    def complete(self, prompt: str) -> str:
        """Prompt in, text out. Raises LLMError only when every retry is exhausted — the Mutator
        catches that and returns a no-op program, so a model outage degrades the search rather than
        killing the run."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        gen: dict[str, Any] = {"temperature": self.temperature}
        if self.max_output_tokens:
            gen["maxOutputTokens"] = int(self.max_output_tokens)
        if self.thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": int(self.thinking_budget)}
        payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen}

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)   # key travels in the header
                if resp.status_code in _TRANSIENT_STATUS:
                    raise httpx.HTTPStatusError(
                        f"transient {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                with self._lock:
                    self.calls += 1
                return _extract_text(resp.json())
            except Exception as exc:  # noqa: BLE001 — retry policy is decided below
                last = exc
                if attempt >= self.max_retries or not _is_transient(exc):
                    break
                with self._lock:
                    self.retries += 1
                # Exponential backoff with jitter: 64 workers that all retry in lockstep after a 429
                # simply reproduce the 429.
                time.sleep(min(30.0, (2**attempt) * 0.5 * (1.0 + random.random())))

        with self._lock:
            self.failures += 1
        raise LLMError(f"model call failed after {self.max_retries + 1} attempts: {last}") from last

    def stats(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "model": self.model,
                "calls": self.calls,
                "retries": self.retries,
                "failures": self.failures,
            }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GeminiMutationClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------ helpers
def _settings_value(name: str) -> str:
    """Fall back to propab's pydantic settings, which load `.env`. Imported lazily so this module
    stays usable (and testable) without the rest of propab's config stack."""
    try:
        from propab.config import settings

        return str(getattr(settings, name, "") or "").strip()
    except Exception:  # noqa: BLE001 — no config is a normal standalone case, not an error
        return ""


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError))


def _extract_text(body: dict[str, Any]) -> str:
    """Pull the text out of a Gemini response. Returns "" rather than raising on a shape we don't
    recognise — an empty completion is a dead mutation, which the Mutator already handles."""
    for cand in body.get("candidates") or []:
        parts = ((cand or {}).get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str))
        if text.strip():
            return text.strip()
    return ""
