from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any
from xml.etree import ElementTree

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.embeddings import embed_texts
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.literature_cache import (
    load_papers_by_ids,
    lookup_cached_paper_ids,
    store_query_cache,
)
from services.orchestrator.literature_quality import (
    build_retrieval_diagnostics,
    build_search_intents,
    classify_evidence_status,
    compute_evidence_coverage,
    gate_corpus_quality,
    insufficient_prior,
)
from services.orchestrator.prior_builder import synthesize_prior_from_papers
from services.orchestrator.retrieval import expand_query, run_hybrid_retrieval
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


def _extract_arxiv_ids_from_text(text: str, limit: int = 80) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in re.finditer(r"arxiv:?\s*v?(\d{4})\.(\d{4,5})(v\d+)?", text, re.IGNORECASE):
        aid = f"{m.group(1)}.{m.group(2)}{m.group(3) or ''}"
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
        if len(out) >= limit:
            break
    for m in re.finditer(r"\b(\d{4})\.(\d{4,5})(v\d+)?\b", text):
        aid = f"{m.group(1)}.{m.group(2)}{m.group(3) or ''}"
        if aid not in seen and len(m.group(1)) == 4:
            seen.add(aid)
            out.append(aid)
        if len(out) >= limit:
            break
    return out


async def _persist_paper_citations(
    session_factory: async_sessionmaker,
    source_paper_id: str,
    cited_ids: list[str],
) -> None:
    cited_ids = [c for c in cited_ids if c and c != source_paper_id]
    if not cited_ids:
        return
    async with session_factory() as session:
        for cid in cited_ids:
            await session.execute(
                text(
                    """
                    INSERT INTO paper_citations (source_paper_id, cited_paper_id)
                    VALUES (:source, :cited)
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"source": source_paper_id, "cited": cid},
            )
        await session.commit()


async def _fetch_semantic_scholar(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": max_results, "fields": "paperId,title,abstract,externalIds,authors"}
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    papers: list[dict[str, Any]] = []
    for item in payload.get("data") or []:
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        pid = str(item.get("paperId", ""))
        if arxiv_id:
            paper_id = str(arxiv_id)
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        else:
            paper_id = f"semantic_scholar:{pid}" if pid else ""
            pdf_url = None
        authors_raw = item.get("authors") or []
        names = [a.get("name", "") for a in authors_raw if isinstance(a, dict)]
        papers.append(
            {
                "id": paper_id,
                "title": item.get("title") or "",
                "abstract": item.get("abstract") or "",
                "authors": [n for n in names if n],
                "pdf_url": pdf_url,
                "sections_json": None,
            }
        )
    return [p for p in papers if p.get("id")]


async def _fetch_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, params=params)
    response.raise_for_status()
    return _parse_arxiv_feed(response.text)


async def _fetch_arxiv_by_ids(arxiv_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch arXiv metadata for known IDs (citation graph expansion)."""
    ids = [i for i in arxiv_ids if i and not i.startswith("semantic_scholar:")][:20]
    if not ids:
        return []
    url = "https://export.arxiv.org/api/query"
    params = {"id_list": ",".join(ids), "max_results": len(ids)}
    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.get(url, params=params)
    response.raise_for_status()
    return _parse_arxiv_feed(response.text)


async def _expand_papers_via_citations(
    session_factory: async_sessionmaker,
    papers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add 1-hop cited papers from the citation graph (domain-agnostic)."""
    seed_ids = [str(p["id"]) for p in papers if p.get("id")]
    if not seed_ids:
        return papers
    cap = max(0, settings.literature_citation_expand_max)
    if cap == 0:
        return papers

    async with session_factory() as session:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT DISTINCT cited_paper_id
                    FROM paper_citations
                    WHERE source_paper_id = ANY(:ids)
                    LIMIT :lim
                    """
                ),
                {"ids": seed_ids, "lim": cap * 2},
            )
        ).fetchall()

    existing = {str(p.get("id")) for p in papers if p.get("id")}
    cited_ids = [str(r[0]) for r in rows if r and r[0] and str(r[0]) not in existing][:cap]
    if not cited_ids:
        return papers

    extra = await load_papers_by_ids(session_factory, cited_ids)
    found_ids = {str(p.get("id")) for p in extra if p.get("id")}
    missing = [cid for cid in cited_ids if cid not in found_ids]
    if missing:
        fetched = await _fetch_arxiv_by_ids(missing)
        if fetched:
            await _upsert_papers(session_factory, fetched)
            extra.extend(fetched)

    if not extra:
        return papers
    return papers + extra


async def _filter_with_adaptive_threshold(
    question: str,
    papers: list[dict[str, Any]],
    *,
    session_id: str,
    emitter: EventEmitter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Relevance filter; relax threshold once when too few papers survive (general-purpose)."""
    threshold = settings.literature_relevance_threshold
    kept, dropped = await _filter_papers_by_question_relevance(
        question, papers, session_id=session_id, emitter=emitter, threshold=threshold,
    )
    floor = settings.literature_relevance_threshold_floor
    min_kept = settings.literature_min_papers_kept
    if len(kept) >= min_kept or threshold <= floor:
        return kept, dropped, threshold

    relaxed = max(floor, threshold - 0.08)
    if relaxed >= threshold:
        return kept, dropped, threshold

    kept2, dropped2 = await _filter_papers_by_question_relevance(
        question, papers, session_id=session_id, emitter=emitter, threshold=relaxed,
    )
    if len(kept2) > len(kept):
        return kept2, dropped2, relaxed
    return kept, dropped, threshold


async def _fetch_papers_multi_intent(
    intents: list[str],
    *,
    per_intent: int,
    max_total: int,
) -> list[dict[str, Any]]:
    """Fetch from arXiv (fallback Semantic Scholar) per search intent; merge and dedupe."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for intent in intents:
        if len(merged) >= max_total:
            break
        batch = await _fetch_arxiv(intent, max_results=per_intent)
        if not batch:
            batch = await _fetch_semantic_scholar(intent, max_results=per_intent)
        for paper in batch:
            pid = str(paper.get("id") or "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            merged.append(paper)
            if len(merged) >= max_total:
                break
    return merged


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
            raw = page.get_text("text", sort=True) or ""
            if "â" in raw:
                try:
                    raw = raw.encode("latin-1").decode("utf-8", errors="replace")
                except UnicodeError:
                    pass
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            # Drop isolated page numbers / very short header/footer noise.
            cleaned = [ln for ln in lines if not (len(ln) <= 3 and ln.isdigit())]
            texts.append("\n".join(cleaned))
        body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(texts)).replace("\x00", "").strip()[:200_000]
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


def _cosine_question_paper(qv: list[float], pv: list[float]) -> float:
    dot = sum(x * y for x, y in zip(qv, pv))
    nq = math.sqrt(sum(x * x for x in qv))
    np_ = math.sqrt(sum(x * x for x in pv))
    if nq == 0.0 or np_ == 0.0:
        return 0.0
    return dot / (nq * np_)


async def _filter_papers_by_question_relevance(
    question: str,
    papers: list[dict[str, Any]],
    *,
    session_id: str,
    emitter: EventEmitter,
    threshold: float = 0.4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop papers whose title+abstract embedding similarity to the question is below *threshold*."""
    if not papers:
        return [], []
    api_key = (settings.embed_api_secret or "").strip()
    if not api_key:
        tagged = [dict(p, _relevance_score=None) for p in papers]
        return tagged, []

    texts = [question[:8000]] + [
        f"{p.get('title') or ''}\n{p.get('abstract') or ''}"[:8000] for p in papers
    ]
    try:
        vectors = await asyncio.wait_for(
            embed_texts(
                texts=texts,
                api_key=api_key,
                model=settings.embed_model,
                provider=settings.embed_provider,
            ),
            timeout=90.0,
        )
    except Exception:
        tagged = [dict(p, _relevance_score=None) for p in papers]
        return tagged, []
    if not vectors or len(vectors) != len(texts):
        tagged = [dict(p, _relevance_score=None) for p in papers]
        return tagged, []

    qvec = vectors[0]
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    dropped_meta: list[dict[str, Any]] = []
    kept_meta: list[dict[str, Any]] = []
    for paper, pvec in zip(papers, vectors[1:], strict=True):
        sim = _cosine_question_paper(qvec, pvec)
        tagged = dict(paper)
        tagged["_relevance_score"] = round(sim, 4)
        if sim >= threshold:
            kept.append(tagged)
            kept_meta.append(
                {"paper_id": paper.get("id"), "title": paper.get("title"), "similarity": round(sim, 4)}
            )
        else:
            dropped.append(tagged)
            dropped_meta.append(
                {"paper_id": paper.get("id"), "title": paper.get("title"), "similarity": round(sim, 4)}
            )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_PAPERS_RELEVANCE_FILTER,
        step="literature.relevance_filter",
        payload={
            "threshold": threshold,
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "kept": kept_meta[:24],
            "dropped": dropped_meta[:24],
        },
    )
    return kept, dropped


async def _enrich_papers_with_pdf(
    papers: list[dict[str, Any]],
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
) -> None:
    parallelism = max(1, settings.literature_pdf_parallelism)
    sem = asyncio.Semaphore(parallelism)

    async def _parse_one(paper: dict[str, Any], client: httpx.AsyncClient) -> None:
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            return
        sections = paper.get("sections_json")
        if isinstance(sections, dict) and sections.get("body"):
            return
        async with sem:
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
                body_text = sections_json.get("body") or ""
                cited = _extract_arxiv_ids_from_text(body_text)
                await _persist_paper_citations(session_factory, paper["id"], cited)
            except Exception as exc:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_PARSED,
                    step="literature.parse",
                    payload={"arxiv_id": paper.get("id"), "error": str(exc), "skipped": True},
                )

    async with httpx.AsyncClient(timeout=90.0) as client:
        await asyncio.gather(*(_parse_one(p, client) for p in papers))


async def build_prior(
    parsed: ParsedQuestion,
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    paper_ttl_days: int,
    llm: LLMClient,
) -> Prior:
    del paper_ttl_days  # legacy param; query cache TTL is settings-driven
    question = parsed.text
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.fetch",
        payload={"query": question},
    )

    expansion = await expand_query(llm, session_id, question)
    search_intents = build_search_intents(question, expansion)

    cached_ids, cache_match = await lookup_cached_paper_ids(session_factory, question)
    papers_fetched: list[dict[str, Any]] = []
    did_live_fetch = False

    if cached_ids:
        papers_fetched = await load_papers_by_ids(session_factory, cached_ids)
        for paper in papers_fetched:
            await emitter.emit(
                session_id=session_id,
                event_type=EventType.LIT_PAPER_CACHED,
                step="literature.cache_hit",
                payload={
                    "arxiv_id": paper["id"],
                    "title": paper.get("title"),
                    "cache_match": cache_match,
                    "query_scoped": True,
                },
            )

    max_rounds = settings.literature_expansion_rounds + 1
    papers_kept: list[dict[str, Any]] = []
    papers_dropped: list[dict[str, Any]] = []
    retrieval_chunks: list[dict[str, Any]] = []
    evidence_coverage = 0.0
    gate_passed = False
    gate_reasons: list[str] = []
    expansion_round = 0
    relevance_threshold_used = settings.literature_relevance_threshold

    for expansion_round in range(max_rounds):
        if expansion_round > 0 or not papers_fetched:
            multiplier = expansion_round + 1
            per_intent = min(settings.literature_fetch_per_intent * multiplier, 25)
            max_candidates = min(settings.literature_max_candidates * multiplier, 50)
            papers_fetched = await _fetch_papers_multi_intent(
                search_intents,
                per_intent=per_intent,
                max_total=max_candidates,
            )
            did_live_fetch = True
            cache_match = None
            await _upsert_papers(session_factory, papers_fetched)
            for paper in papers_fetched:
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_FOUND,
                    step="literature.fetch",
                    payload={
                        "arxiv_id": paper["id"],
                        "title": paper.get("title"),
                        "expansion_round": expansion_round,
                    },
                )
                await emitter.emit(
                    session_id=session_id,
                    event_type=EventType.LIT_PAPER_INDEXED,
                    step="literature.index",
                    payload={"arxiv_id": paper["id"], "source": "metadata"},
                )

        await _enrich_papers_with_pdf(
            papers_fetched,
            session_id=session_id,
            emitter=emitter,
            session_factory=session_factory,
        )
        papers_fetched = await _expand_papers_via_citations(session_factory, papers_fetched)

        papers_kept, papers_dropped, relevance_threshold_used = await _filter_with_adaptive_threshold(
            question,
            papers_fetched,
            session_id=session_id,
            emitter=emitter,
        )

        retrieval_chunks = await run_hybrid_retrieval(
            session_id=session_id,
            question=question,
            papers=papers_kept,
            llm=llm,
            emitter=emitter,
            session_factory=session_factory,
        )
        kept_ids = {str(p.get("id")) for p in papers_kept if p.get("id")}
        retrieval_chunks = [c for c in retrieval_chunks if str(c.get("paper_id")) in kept_ids]

        chunk_texts = [str(c.get("text") or "") for c in retrieval_chunks if c.get("text")]
        evidence_coverage = await compute_evidence_coverage(question, chunk_texts)
        gate_passed, gate_reasons = gate_corpus_quality(
            papers_kept=papers_kept,
            chunk_count=len(retrieval_chunks),
            evidence_coverage=evidence_coverage,
        )
        if gate_passed:
            break

    diagnostics = build_retrieval_diagnostics(
        question=question,
        search_intents=search_intents,
        cache_match=cache_match,
        papers_fetched=papers_fetched,
        papers_kept=papers_kept,
        papers_dropped=papers_dropped,
        chunk_count=len(retrieval_chunks),
        evidence_coverage=evidence_coverage,
        expansion_round=expansion_round,
        gate_reasons=gate_reasons,
    )
    diagnostics["relevance_threshold_used"] = relevance_threshold_used
    diagnostics["retrieved_chunks"] = [
        {
            "paper_id": c.get("paper_id"),
            "chunk_index": c.get("chunk_index"),
            "text_preview": (c.get("text") or "")[:240],
        }
        for c in retrieval_chunks[:30]
    ]

    if did_live_fetch and papers_fetched:
        await store_query_cache(
            session_factory,
            question,
            [str(p["id"]) for p in papers_fetched if p.get("id")],
        )

    if not gate_passed:
        prior = insufficient_prior(question, diagnostics)
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_PRIOR_BUILT,
            step="literature.prior_build",
            payload={"prior": prior.to_dict(), "skipped_llm": True},
        )
        return prior

    prior = await synthesize_prior_from_papers(
        llm=llm,
        session_id=session_id,
        question=question,
        papers=papers_kept,
        emitter=emitter,
        retrieval_chunks=retrieval_chunks,
    )
    prior.evidence_coverage = evidence_coverage
    prior.retrieval_diagnostics = diagnostics
    prior.evidence_status = classify_evidence_status(
        prior,
        evidence_coverage=evidence_coverage,
        gate_passed=gate_passed,
    )

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_PRIOR_BUILT,
        step="literature.prior_build",
        payload={"prior": prior.to_dict()},
    )
    return prior
