from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Cross-provider default embed model id. Ships as the ``embed_model`` default in
# config.py and is OpenAI-only, so when the provider is Google we treat this exact
# value as "operator left the default" and resolve it to the Google default below.
_OPENAI_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# Google's default embedding model (see ``_normalize_google_model`` and the
# literature service, which defaults ``embed_model`` to the same id).
_GOOGLE_DEFAULT_EMBED_MODEL = "gemini-embedding-2"


def resolve_embed_model(provider: str, model: str) -> str:
    """Return a provider-appropriate embedding model id.

    When ``provider`` is google/gemini but ``model`` is still the OpenAI-only
    cross-provider default (``text-embedding-3-small``), resolve it to the Google
    default so a deployment that only sets ``EMBED_PROVIDER=gemini`` still gets a
    working embed model. Any explicit ``model`` override is returned unchanged.
    The OpenAI happy path (openai + text-embedding-3-small) is untouched.
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip()
    if p in {"google", "gemini"} and m == _OPENAI_DEFAULT_EMBED_MODEL:
        logger.warning(
            "embed_provider=%s but embed_model is the OpenAI default %r; "
            "resolving to Google default %r. Set EMBED_MODEL explicitly to override.",
            p,
            _OPENAI_DEFAULT_EMBED_MODEL,
            _GOOGLE_DEFAULT_EMBED_MODEL,
        )
        return _GOOGLE_DEFAULT_EMBED_MODEL
    return m


async def _openai_embed(*, texts: list[str], api_key: str, model: str) -> list[list[float]]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    return [row["embedding"] for row in sorted(data, key=lambda x: x["index"])]


def _normalize_google_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        m = "gemini-embedding-2"
    return m[len("models/") :] if m.startswith("models/") else m


async def _google_embed(*, texts: list[str], api_key: str, model: str) -> list[list[float]]:
    # Official docs: POST /v1beta/models/{model}:embedContent, x-goog-api-key header.
    # For retrieval we need one embedding per text, so we issue one call per input.
    model_id = _normalize_google_model(model)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:embedContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for t in texts:
            payload: dict[str, Any] = {
                "model": f"models/{model_id}",
                "content": {"parts": [{"text": t}]},
            }
            response: httpx.Response | None = None
            for attempt in range(3):
                response = await client.post(url, headers=headers, json=payload)
                # Gemini embedding endpoint can intermittently return 429/503 under burst load.
                if response.status_code not in (429, 503):
                    break
                if attempt < 2:
                    await asyncio.sleep(1.5 * (2**attempt))
            if response is None:
                raise RuntimeError("Google embedding call did not produce a response.")
            response.raise_for_status()
            body = response.json()
            emb = body.get("embedding") or {}
            vals = emb.get("values")
            if not isinstance(vals, list):
                raise ValueError("Google embeddings response missing embedding.values")
            out.append([float(x) for x in vals])
    return out


async def embed_texts(*, texts: list[str], api_key: str, model: str, provider: str = "openai") -> list[list[float]]:
    if not api_key or not texts:
        return []
    p = (provider or "openai").strip().lower()
    # Resolve a cross-provider default (e.g. the OpenAI id under a Google provider)
    # to a provider-appropriate model so callers that pass the raw settings default
    # do not silently 400 and fall back to non-embedding ranking.
    model = resolve_embed_model(p, model)
    if p in {"google", "gemini"}:
        return await _google_embed(texts=texts, api_key=api_key, model=model)
    return await _openai_embed(texts=texts, api_key=api_key, model=model)
