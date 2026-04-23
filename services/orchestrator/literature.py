from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from typing import Any
from xml.etree import ElementTree

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.types import EventType
from services.orchestrator.intake import ParsedQuestion


@dataclass(slots=True)
class Prior:
    established_facts: list[dict]
    contested_claims: list[dict]
    open_gaps: list[dict]
    dead_ends: list[dict]
    key_papers: list[dict]

    def to_dict(self) -> dict:
        return {
            "established_facts": self.established_facts,
            "contested_claims": self.contested_claims,
            "open_gaps": self.open_gaps,
            "dead_ends": self.dead_ends,
            "key_papers": self.key_papers,
        }


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(xml_text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", namespace):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=namespace).split("/")[-1]
        title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
        authors = [a.findtext("atom:name", default="", namespaces=namespace) for a in entry.findall("atom:author", namespace)]
        papers.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": summary,
                "authors": [author for author in authors if author],
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
            }
        )
    return papers


async def _fetch_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, params=params)
    response.raise_for_status()
    return _parse_arxiv_feed(response.text)


async def _load_cached_papers(session_factory: async_sessionmaker, ttl_days: int) -> list[dict[str, Any]]:
    min_time = datetime.now(tz=UTC) - timedelta(days=ttl_days)
    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT id, title, abstract, authors, pdf_url, ingested_at
                    FROM papers
                    WHERE status = 'indexed' AND ingested_at >= :min_time
                    ORDER BY ingested_at DESC
                    LIMIT 5
                    """
                ),
                {"min_time": min_time},
            )
        ).mappings().all()
    return [dict(row) for row in rows]


async def _upsert_papers(session_factory: async_sessionmaker, papers: list[dict[str, Any]]) -> None:
    async with session_factory() as session:
        for paper in papers:
            await session.execute(
                text(
                    """
                    INSERT INTO papers (id, title, authors, abstract, pdf_url, status)
                    VALUES (:id, :title, CAST(:authors AS jsonb), :abstract, :pdf_url, 'indexed')
                    ON CONFLICT (id) DO UPDATE
                    SET title = EXCLUDED.title,
                        authors = EXCLUDED.authors,
                        abstract = EXCLUDED.abstract,
                        pdf_url = EXCLUDED.pdf_url,
                        status = 'indexed',
                        ingested_at = NOW()
                    """
                ),
                {
                    "id": paper["id"],
                    "title": paper["title"],
                    "authors": json.dumps(paper["authors"]),
                    "abstract": paper["abstract"],
                    "pdf_url": paper["pdf_url"],
                },
            )
        await session.commit()


async def build_prior(
    parsed: ParsedQuestion,
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    paper_ttl_days: int,
) -> Prior:
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.fetch",
        payload={"query": parsed.text},
    )

    cached = await _load_cached_papers(session_factory, ttl_days=paper_ttl_days)
    papers = cached
    if cached:
        for paper in cached:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.LIT_PAPER_CACHED,
                step="literature.cache_hit",
                payload={"arxiv_id": paper["id"], "title": paper["title"]},
            )
    else:
        papers = await _fetch_arxiv(parsed.text)
        await _upsert_papers(session_factory, papers)
        for paper in papers:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.LIT_PAPER_FOUND,
                step="literature.fetch",
                payload={"arxiv_id": paper["id"], "title": paper["title"]},
            )
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.LIT_PAPER_INDEXED,
                step="literature.index",
                payload={"arxiv_id": paper["id"]},
            )

    return Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[
            {
                "text": f"No indexed literature yet for: {parsed.text}",
                "source_paper": "bootstrap",
                "gap_type": "missing_data",
            }
        ],
        dead_ends=[],
        key_papers=[
            {"paper_id": p["id"], "summary": p.get("abstract", "")[:220], "title": p.get("title", "")}
            for p in papers
            if p.get("id")
        ],
    )
