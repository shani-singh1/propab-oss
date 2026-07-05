"""
Vector storage for semantic claim retrieval.

Real backend: Qdrant, via ``qdrant-client``. Fallback backend: a small
in-memory brute-force cosine index — used when ``qdrant_backend != "qdrant"``
or when the configured Qdrant instance is unreachable at startup, so the
service degrades to "slower but correct" rather than failing to start. Both
backends implement the same three operations the retriever needs: upsert,
semantic search, and a full dump (used by novelty_check.py's top-K scan).
"""
from __future__ import annotations

import uuid
from typing import Any

from services.literature.app.indexer.embeddings import cosine_similarity
from services.literature.app.models import ExtractedClaim


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._claims: dict[str, ExtractedClaim] = {}

    async def upsert(self, claims: list[ExtractedClaim]) -> None:
        for c in claims:
            if c.embedding:
                self._claims[c.claim_id or str(uuid.uuid4())] = c

    async def search(self, embedding: list[float], top_k: int = 30, domain_id: str = "") -> list[tuple[ExtractedClaim, float]]:
        scored = [
            (c, cosine_similarity(embedding, c.embedding))
            for c in self._claims.values()
            if c.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def all(self) -> list[ExtractedClaim]:
        return list(self._claims.values())

    async def count(self) -> int:
        return len(self._claims)


class QdrantVectorStore:
    def __init__(self, *, url: str, collection: str, dim: int) -> None:
        self._url = url
        self._collection = collection
        self._dim = dim
        self._client = None

    async def _ensure_client(self):
        if self._client is not None:
            return self._client
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = AsyncQdrantClient(url=self._url)
        collections = await client.get_collections()
        if self._collection not in {c.name for c in collections.collections}:
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
        self._client = client
        return client

    async def upsert(self, claims: list[ExtractedClaim]) -> None:
        from qdrant_client.models import PointStruct

        client = await self._ensure_client()
        points = [
            PointStruct(
                id=_claim_uuid(c.claim_id),
                vector=c.embedding,
                payload=c.model_dump(exclude={"embedding"}),
            )
            for c in claims
            if c.embedding
        ]
        if points:
            await client.upsert(collection_name=self._collection, points=points)

    async def search(self, embedding: list[float], top_k: int = 30, domain_id: str = "") -> list[tuple[ExtractedClaim, float]]:
        client = await self._ensure_client()
        hits = await client.search(collection_name=self._collection, query_vector=embedding, limit=top_k)
        out = []
        for h in hits:
            payload = dict(h.payload or {})
            payload["embedding"] = []
            out.append((ExtractedClaim(**payload), float(h.score)))
        return out

    async def all(self) -> list[ExtractedClaim]:
        client = await self._ensure_client()
        points, _ = await client.scroll(collection_name=self._collection, limit=10_000, with_payload=True)
        return [ExtractedClaim(**{**dict(p.payload or {}), "embedding": []}) for p in points]

    async def count(self) -> int:
        client = await self._ensure_client()
        info = await client.count(collection_name=self._collection)
        return info.count


def _claim_uuid(claim_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, claim_id or str(uuid.uuid4())))


async def build_vector_store(*, backend: str, url: str, collection: str, dim: int):
    if backend == "qdrant" and url:
        store = QdrantVectorStore(url=url, collection=collection, dim=dim)
        try:
            await store._ensure_client()
            return store
        except Exception:
            # Qdrant configured but unreachable — degrade rather than crash.
            return InMemoryVectorStore()
    return InMemoryVectorStore()
