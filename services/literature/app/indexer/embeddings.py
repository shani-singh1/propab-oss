"""
Embeddings for claim retrieval and dedup.

Three providers behind one interface:
- ``google``/``gemini``: Gemini embeddings via the Generative Language API
  (``embedContent``) — the default, matching ``EMBED_PROVIDER=google`` /
  ``EMBED_MODEL=gemini-embedding-2`` already used elsewhere in Propab
  (``packages/propab-core/propab/embeddings.py``), so this service shares the
  same embedding quality rather than silently defaulting to something weaker.
- ``openai``: real OpenAI embeddings, used when explicitly configured.
- ``offline``: a deterministic hashing embedding (no network, no API key)
  used whenever no matching API key is configured, so the service still runs
  and its semantic-similarity logic (novelty check, dedup, clustering) is
  exercisable in dev/CI without any external dependency. It is not as good as
  a real embedding model, but it is consistent (same text -> same vector) and
  captures token overlap, which is enough for cosine similarity to behave
  sanely in tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Sequence

import httpx

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _offline_embed(text: str, dim: int) -> list[float]:
    """Hashed bag-of-tokens embedding: each token deterministically hashes to
    a dimension and sign, then the vector is L2-normalized. Two texts sharing
    tokens get non-trivial cosine similarity; unrelated texts get near-zero."""
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return vec
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _normalize_google_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        m = "gemini-embedding-2"
    return m[len("models/"):] if m.startswith("models/") else m


class EmbeddingClient:
    def __init__(
        self,
        *,
        provider: str = "offline",
        model: str = "text-embedding-3-small",
        api_key: str = "",
        google_api_key: str = "",
        dim: int = 256,
        http_timeout: float = 30.0,
        max_concurrency: int = 8,
    ) -> None:
        p = (provider or "offline").strip().lower()
        if p in ("google", "gemini") and google_api_key:
            self.provider = "google"
        elif p == "openai" and api_key:
            self.provider = "openai"
        else:
            self.provider = "offline"
        self.model = model
        self._api_key = api_key
        self._google_api_key = google_api_key
        self._max_concurrency = max_concurrency
        self.dim = dim
        self._timeout = http_timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self.provider == "google":
                return await self._embed_google(texts)
            if self.provider == "openai":
                return await self._embed_openai(texts)
        except Exception:
            # A flaky embedding API must never take the whole service
            # down — degrade to offline embeddings for this batch.
            pass
        return [_offline_embed(t, self.dim) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0] if result else []

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    async def _embed_google(self, texts: list[str]) -> list[list[float]]:
        # Official docs: POST /v1beta/models/{model}:embedContent, x-goog-api-key
        # header. No batch endpoint for retrieval-quality embeddings, so one
        # call per input. Requests run concurrently (bounded by a semaphore —
        # a /prior call can easily produce 50+ claims, and issuing those
        # fully sequentially at ~0.3-0.8s each would make embedding the
        # slowest part of the pipeline by far) but each result is placed back
        # at its original index, so callers can still zip(texts, embeddings).
        # Mirrors packages/propab-core/propab/embeddings.py's request shape
        # exactly so both call sites behave identically against the same API.
        model_id = _normalize_google_model(self.model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:embedContent"
        headers = {"x-goog-api-key": self._google_api_key, "Content-Type": "application/json"}
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _one(client: httpx.AsyncClient, text: str) -> list[float]:
            payload = {"model": f"models/{model_id}", "content": {"parts": [{"text": text}]}}
            async with semaphore:
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
            vals = (body.get("embedding") or {}).get("values")
            if not isinstance(vals, list):
                raise ValueError("Google embeddings response missing embedding.values")
            return [float(x) for x in vals]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return list(await asyncio.gather(*(_one(client, t) for t in texts)))
