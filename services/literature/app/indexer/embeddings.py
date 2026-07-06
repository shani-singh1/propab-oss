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
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Per-instance text→vector cache. The QA/eval deep-read path re-ranks the
        # same chunks several times per question (once per candidate doc, then a
        # pooled final rank), so without caching dense ranking re-embeds every
        # chunk ~18× and every question times out (measured, 0.10.x). Caching
        # embeds each unique text ONCE; subsequent ranks are local cosine only.
        cache = self._cache
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            try:
                if self.provider == "google":
                    vecs = await self._embed_google(missing)
                elif self.provider == "openai":
                    vecs = await self._embed_openai(missing)
                else:
                    vecs = [_offline_embed(t, self.dim) for t in missing]
            except Exception:
                # A flaky embedding API must never take the whole service down —
                # degrade to offline embeddings for the missing texts.
                vecs = [_offline_embed(t, self.dim) for t in missing]
            for t, v in zip(missing, vecs):
                cache[t] = v
            # Bound memory for the long-running /prior service: drop the oldest
            # entries (dict preserves insertion order) once the cache is large.
            # The eval process is short-lived so it never trips this; a server
            # embedding endlessly would otherwise grow the cache without limit.
            if len(cache) > 50_000:
                for old in list(cache)[: len(cache) - 50_000]:
                    del cache[old]
        return [cache[t] for t in texts]

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
        # POST /v1beta/models/{model}:batchEmbedContents — the BATCH endpoint,
        # up to 100 texts in ONE request. The per-text ``embedContent`` path
        # (one HTTP call per input, fired concurrently) was the root cause of
        # the ~28% ReadTimeout rate that capped every LitQA2 eval run
        # (CHANGELOG.md 0.5.0) — dense chunk ranking could fire ~40 embed calls
        # per question × 6 concurrent questions = ~240 in flight. Batching
        # collapses a whole paper's chunks into a single call, which is what
        # makes dense passage retrieval (0.10.0) viable at all. Order is
        # preserved (the API returns embeddings in request order), so callers
        # can still zip(texts, embeddings). Mirrors the request shape in
        # packages/propab-core/propab/embeddings.py.
        model_id = _normalize_google_model(self.model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:batchEmbedContents"
        headers = {"x-goog-api-key": self._google_api_key, "Content-Type": "application/json"}
        semaphore = asyncio.Semaphore(self._max_concurrency)
        batches = [texts[i:i + 100] for i in range(0, len(texts), 100)]

        async def _batch(client: httpx.AsyncClient, batch: list[str]) -> list[list[float]]:
            payload = {
                "requests": [
                    {"model": f"models/{model_id}", "content": {"parts": [{"text": t}]}} for t in batch
                ]
            }
            async with semaphore:
                response: httpx.Response | None = None
                for attempt in range(3):
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code not in (429, 503):
                        break
                    if attempt < 2:
                        await asyncio.sleep(1.5 * (2**attempt))
            if response is None:
                raise RuntimeError("Google embedding call did not produce a response.")
            response.raise_for_status()
            body = response.json()
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ValueError("Google batchEmbedContents response missing/mismatched embeddings")
            out: list[list[float]] = []
            for item in embeddings:
                vals = (item or {}).get("values")
                if not isinstance(vals, list):
                    raise ValueError("Google embeddings response missing embedding.values")
                out.append([float(x) for x in vals])
            return out

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            batch_results = await asyncio.gather(*(_batch(client, b) for b in batches))
        return [emb for batch in batch_results for emb in batch]
