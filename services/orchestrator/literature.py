from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from typing import Any
from xml.etree import ElementTree

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.prior_builder import synthesize_prior_from_papers
from services.orchestrator.schemas import Prior


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
                "sections_json": None,
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
                    SELECT id, title, abstract, authors, pdf_url, ingested_at, sections_json
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
            sections = paper.get("sections_json")
            params = {
                "id": paper["id"],
                "title": paper["title"],
                "authors": json.dumps(paper["authors"]),
                "abstract": paper["abstract"],
                "pdf_url": paper["pdf_url"],
            }
            if isinstance(sections, dict):
                params["sections_json"] = json.dumps(sections)
                await session.execute(
                    text(
                        """
                        INSERT INTO papers (id, title, authors, abstract, pdf_url, sections_json, status)
                        VALUES (:id, :title, CAST(:authors AS jsonb), :abstract, :pdf_url, CAST(:sections_json AS jsonb), 'indexed')
                        ON CONFLICT (id) DO UPDATE
                        SET title = EXCLUDED.title,
                            authors = EXCLUDED.authors,
                            abstract = EXCLUDED.abstract,
                            pdf_url = EXCLUDED.pdf_url,
                            sections_json = COALESCE(EXCLUDED.sections_json, papers.sections_json),
                            status = 'indexed',
                            ingested_at = NOW()
                        """
                    ),
                    params,
                )
            else:
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
                    params,
                )
        await session.commit()


async def _update_paper_sections(
    session_factory: async_sessionmaker,
    paper_id: str,
    sections_json: dict[str, Any],
) -> None:
    async with session_factory() as session:
        await session.execute(
            text(
                """
                UPDATE papers
                SET sections_json = CAST(:sections_json AS jsonb)
                WHERE id = :id
                """
            ),
            {"id": paper_id, "sections_json": json.dumps(sections_json)},
        )
        await session.commit()


def _parse_pdf_bytes(content: bytes) -> dict[str, Any]:
    import fitz

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page_count = doc.page_count
        texts: list[str] = []
        for i in range(min(page_count, 50)):
            page = doc.load_page(i)
            texts.append(page.get_text("text") or "")
        body = "\n\n".join(texts).strip()[:200_000]
        sections_json = {
            "page_count": page_count,
            "body": body,
            "pages_indexed": min(page_count, 50),
        }
        section_count = max(1, min(page_count, 50))
        return {"sections_json": sections_json, "section_count": section_count}
    finally:
        doc.close()


async def _download_pdf_bytes(client: httpx.AsyncClient, pdf_url: str) -> bytes:
    response = await client.get(pdf_url, follow_redirects=True)
    response.raise_for_status()
    return response.content


async def _enrich_papers_with_pdf(
    papers: list[dict[str, Any]],
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> None:
    async with httpx.AsyncClient(timeout=90.0) as client:
        for paper in papers:
            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                continue
            sections = paper.get("sections_json")
            if isinstance(sections, dict) and sections.get("body"):
                continue
            try:
                content = await _download_pdf_bytes(client, pdf_url)
                parsed = await asyncio.to_thread(_parse_pdf_bytes, content)
                sections_json = parsed["sections_json"]
                await _update_paper_sections(session_factory, paper["id"], sections_json)
                paper["sections_json"] = sections_json
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_PARSED,
                    step="literature.parse",
                    payload={
                        "arxiv_id": paper["id"],
                        "section_count": parsed["section_count"],
                        "pages_indexed": sections_json.get("pages_indexed"),
                    },
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_INDEXED,
                    step="literature.index",
                    payload={"arxiv_id": paper["id"], "source": "pdf_text"},
                )
            except Exception as exc:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_PARSED,
                    step="literature.parse",
                    payload={"arxiv_id": paper.get("id"), "error": str(exc), "skipped": True},
                )


async def build_prior(
    parsed: ParsedQuestion,
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    paper_ttl_days: int,
    llm: LLMClient,
) -> Prior:
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.fetch",
        payload={"query": parsed.text},
    )

    cached = await _load_cached_papers(session_factory, ttl_days=paper_ttl_days)
    papers = list(cached)
    if papers:
        for paper in papers:
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
                payload={"arxiv_id": paper["id"], "source": "metadata"},
            )

    await _enrich_papers_with_pdf(papers, session_id=session_id, emitter=emitter, session_factory=session_factory)

    return await synthesize_prior_from_papers(
        llm=llm,
        session_id=session_id,
        question=parsed.text,
        papers=papers,
        emitter=emitter,
    )
