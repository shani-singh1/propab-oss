"""
Shared Gemini text-generation client.

Used by exactly two call sites, both of which are explicitly *not* part of
the core citation-extraction contract:
- ``evaluator/litqa2_live.py`` — answers a multiple-choice question from
  retrieved evidence (an eval, not a claim).
- ``extractors/llm_claim_locator.py`` — identifies *which* sentence in a
  document states a claim; it never supplies the claim's text itself (see
  that module's docstring for why this distinction is load-bearing).

The core `/prior` claim extraction (``extractors/claims.py``,
``sources/_latex.py``) has no LLM in it at all — verbatim text there is
always a direct slice of the source. This client exists for the two places
where an LLM's *judgment* (which sentence, what answer) is useful without
ever letting the LLM originate the text a citation is graded against.
"""
from __future__ import annotations

import asyncio

import httpx


async def gemini_generate(
    prompt: str, *, api_key: str, model: str, http_timeout: float = 45.0, max_retries: int = 1
) -> str:
    """Mirrors ``packages/propab-core/propab/llm.py``'s ``_gemini_generate_content``
    exactly (same endpoint, same ``?key=`` auth, same request shape) so this
    behaves identically to the rest of Propab's Gemini usage.

    Retries on timeout/5xx: measured live on a 25-question LitQA2 run, 10/25
    questions (40%) hit ``httpx.ReadTimeout`` with zero retry — each one
    silently became a scored "wrong, not sure" despite having real evidence,
    dragging both accuracy and coverage down for reasons that had nothing to
    do with retrieval or answer quality. A transient API hiccup should not
    count the same as a genuinely wrong answer.

    Timeout/retry tuning is deliberately tight (45s per attempt, 1 retry):
    a first attempt at ``max_retries=2, http_timeout=120.0`` measured *worse*
    on a rerun of the same 25 questions (more, not fewer, outer-timeout
    failures) — each slow attempt was eating the entire per-question budget
    by itself, so a retry never got a chance to run before the caller's
    overall timeout fired. Failing fast and retrying quickly beats one very
    patient attempt when the caller itself has a hard deadline."""
    mid = (model or "gemini-2.0-flash").strip()
    if mid.startswith("models/"):
        mid = mid[len("models/"):]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0}}
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(http_timeout, connect=min(60.0, http_timeout))) as client:
                resp = await client.post(url, params={"key": api_key}, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                await asyncio.sleep(2.0 * (2**attempt))
                continue
            resp.raise_for_status()
            body = resp.json()
            cands = body.get("candidates") or []
            if not cands:
                return ""
            parts = (((cands[0] or {}).get("content") or {}).get("parts")) or []
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(2.0 * (2**attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    return ""
