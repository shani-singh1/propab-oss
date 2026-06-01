"""Query-scoped literature retrieval cache.

Reuses paper IDs only when the research question is similar to a prior query —
never the global "most recently indexed papers" anti-pattern.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from propab.config import settings
from propab.embeddings import embed_texts
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def query_hash(query: str) -> str:
    normalized = _normalize_query(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


async def _embed_query(query: str) -> list[float]:
    api_key = (settings.embed_api_secret or "").strip()
    if not api_key:
        return []
    vecs = await embed_texts(
        texts=[query[:8000]],
        api_key=api_key,
        model=settings.embed_model,
        provider=settings.embed_provider,
    )
    return vecs[0] if vecs else []


async def lookup_cached_paper_ids(
    session_factory: async_sessionmaker,
    question: str,
) -> tuple[list[str], str | None]:
    """Return (paper_ids, match_kind) where match_kind is 'exact'|'similar'|None."""
    ttl = timedelta(days=settings.literature_query_cache_ttl_days)
    cutoff = datetime.now(timezone.utc) - ttl
    q_hash = query_hash(question)

    async with session_factory() as session:
        try:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT paper_ids, query_text
                        FROM literature_query_cache
                        WHERE query_hash = :qh AND created_at >= :cutoff
                        ORDER BY created_at DESC
                        LIMIT 1
                        """
                    ),
                    {"qh": q_hash, "cutoff": cutoff},
                )
            ).first()
        except Exception:
            logger.warning("literature_cache: lookup failed (table missing?)", exc_info=True)
            return [], None
        if row and row.paper_ids:
            ids = _coerce_paper_ids(row.paper_ids)
            if ids:
                return ids, "exact"

        try:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT query_text, paper_ids, query_embedding
                        FROM literature_query_cache
                        WHERE created_at >= :cutoff AND query_embedding IS NOT NULL
                        ORDER BY created_at DESC
                        LIMIT 50
                        """
                    ),
                    {"cutoff": cutoff},
                )
            ).all()
        except Exception:
            logger.warning("literature_cache: similarity lookup failed", exc_info=True)
            return [], None

    if not rows:
        return [], None

    q_vec = await _embed_query(question)
    if not q_vec:
        return [], None

    threshold = settings.literature_query_cache_similarity
    best_ids: list[str] = []
    best_sim = -1.0
    for row in rows:
        emb = row.query_embedding
        if isinstance(emb, str):
            emb = json.loads(emb)
        if not emb:
            continue
        sim = _cosine(q_vec, emb)
        if sim >= threshold and sim > best_sim:
            best_sim = sim
            best_ids = _coerce_paper_ids(row.paper_ids)

    if best_ids:
        return best_ids, "similar"
    return [], None


async def store_query_cache(
    session_factory: async_sessionmaker,
    question: str,
    paper_ids: list[str],
) -> None:
    if not paper_ids:
        return
    q_hash = query_hash(question)
    try:
        q_vec = await _embed_query(question)
        emb_json = json.dumps(q_vec) if q_vec else None
    except Exception:
        logger.warning("literature_cache: could not embed query for cache", exc_info=True)
        emb_json = None

    async with session_factory() as session:
        try:
            await session.execute(
                text(
                    """
                    INSERT INTO literature_query_cache
                        (query_hash, query_text, paper_ids, query_embedding)
                    VALUES (:qh, :qt, CAST(:pids AS jsonb), CAST(:emb AS jsonb))
                    """
                ),
                {
                    "qh": q_hash,
                    "qt": question[:4000],
                    "pids": json.dumps(paper_ids),
                    "emb": emb_json,
                },
            )
            await session.commit()
        except Exception:
            logger.warning("literature_cache: store failed (table missing?)", exc_info=True)


async def load_papers_by_ids(
    session_factory: async_sessionmaker,
    paper_ids: list[str],
) -> list[dict[str, Any]]:
    if not paper_ids:
        return []
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, title, abstract, authors, pdf_url, sections_json, ingested_at, status
                    FROM papers
                    WHERE id = ANY(:ids)
                    """
                ),
                {"ids": paper_ids},
            )
        ).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        paper = dict(row)
        if not paper.get("pdf_url") and paper.get("id"):
            paper["pdf_url"] = f"https://arxiv.org/pdf/{paper['id']}.pdf"
        out.append(paper)
    return out


def _coerce_paper_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [str(x) for x in raw if x]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
