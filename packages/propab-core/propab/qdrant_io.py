from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _point_id(session_id: str, paper_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{paper_id}:{chunk_index}"))


def upsert_chunks(
    *,
    url: str,
    collection: str,
    session_id: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
) -> None:
    if not url or not chunks or len(chunks) != len(vectors):
        return
    try:
        client = QdrantClient(url=url, prefer_grpc=False)
        size = len(vectors[0])
        try:
            client.get_collection(collection)
        except Exception:
            client.create_collection(
                collection_name=collection,
                vectors_config=qm.VectorParams(size=size, distance=qm.Distance.COSINE),
            )
        points = []
        for ch, vec in zip(chunks, vectors, strict=True):
            pid = _point_id(session_id, ch["paper_id"], ch["chunk_index"])
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "session_id": session_id,
                        "paper_id": ch["paper_id"],
                        "chunk_index": ch["chunk_index"],
                        "text": ch["text"][:8000],
                    },
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
    except Exception:
        return


def search_chunks(*, url: str, collection: str, session_id: str, vector: list[float], limit: int = 40) -> list[dict[str, Any]]:
    if not url:
        return []
    try:
        client = QdrantClient(url=url, prefer_grpc=False)
        hits = client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            query_filter=qm.Filter(must=[qm.FieldCondition(key="session_id", match=qm.MatchValue(value=session_id))]),
        )
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        pl = hit.payload or {}
        out.append(
            {
                "paper_id": pl.get("paper_id"),
                "chunk_index": pl.get("chunk_index"),
                "text": pl.get("text", ""),
                "score": float(hit.score or 0.0),
                "rank": rank,
                "source": "dense",
            }
        )
    return out
