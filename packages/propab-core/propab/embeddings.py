from __future__ import annotations

from typing import Any

import httpx


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
            response = await client.post(url, headers=headers, json=payload)
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
    if p in {"google", "gemini"}:
        return await _google_embed(texts=texts, api_key=api_key, model=model)
    return await _openai_embed(texts=texts, api_key=api_key, model=model)
