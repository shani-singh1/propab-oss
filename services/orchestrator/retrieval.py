from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from typing import Any

from rank_bm25 import RankBM25Okapi

from propab.config import settings
from propab.embeddings import embed_texts
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.qdrant_io import search_chunks, upsert_chunks
from propab.types import EventType
from services.orchestrator.bm25_sqlite import read_session_chunks, write_session_chunks


def _chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", s.lower())


def _paper_full_text(paper: dict[str, Any]) -> str:
    parts = [paper.get("title") or "", paper.get("abstract") or ""]
    sec = paper.get("sections_json")
    if isinstance(sec, dict) and sec.get("body"):
        parts.append(str(sec["body"])[:120_000])
    return "\n\n".join(p for p in parts if p).strip()


def _build_chunks_for_session(papers: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for paper in papers:
        pid = paper.get("id") or ""
        if not pid:
            continue
        body = _paper_full_text(paper)
        for idx, ch in enumerate(_chunk_text(body)):
            rows.append((pid, idx, ch))
    return rows


def _extract_expansion_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


async def expand_query(llm: LLMClient, session_id: str, question: str) -> dict[str, Any]:
    prompt = f"""You expand a research retrieval query.

Question:
{question}

Return JSON only:
{{
  "original": "<verbatim question>",
  "rephrasings": ["...", "...", "..."],
  "concepts": ["...", "..."]
}}
Use exactly 3 rephrasings and up to 5 concepts.
"""
    raw = await llm.call(prompt=prompt, purpose="query_expansion", session_id=session_id)
    data = _extract_expansion_json(raw) or {}
    original = str(data.get("original") or question)
    reps = data.get("rephrasings")
    if not isinstance(reps, list):
        reps = []
    concepts = data.get("concepts")
    if not isinstance(concepts, list):
        concepts = []
    rephrasings = [str(x) for x in reps[:3] if str(x).strip()]
    concepts_s = [str(x) for x in concepts[:5] if str(x).strip()]
    return {"original": original, "rephrasings": rephrasings, "concepts": concepts_s}


def _rrf_merge(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rankings:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _bm25_rank(bm25: RankBM25Okapi, corpus_ids: list[str], query: str, limit: int = 40) -> list[str]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[str] = []
    for i in ranked[:limit]:
        out.append(corpus_ids[i])
    return out


async def run_hybrid_retrieval(
    *,
    session_id: str,
    question: str,
    papers: list[dict[str, Any]],
    llm: LLMClient,
    emitter: EventEmitter,
) -> list[dict[str, Any]]:
    expansion = await expand_query(llm, session_id, question)
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_RETRIEVAL_QUERY,
        step="literature.retrieval",
        payload=expansion,
    )

    rows = _build_chunks_for_session(papers)
    await asyncio.to_thread(write_session_chunks, session_id, settings.propab_data_dir, rows)
    loaded = await asyncio.to_thread(read_session_chunks, session_id, settings.propab_data_dir)
    if not loaded:
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_RETRIEVAL_RESULTS,
            step="literature.retrieval",
            payload={"chunks": [], "note": "No chunks indexed for retrieval."},
        )
        return []

    corpus_texts = [t for _, _, t in loaded]
    corpus_ids = [f"{pid}|{idx}" for pid, idx, _ in loaded]
    tokenized = [_tokenize(t) or ["_"] for t in corpus_texts]
    bm25 = RankBM25Okapi(tokenized)

    queries = [expansion["original"], *expansion["rephrasings"]]
    bm25_rankings: list[list[str]] = []
    for q in queries:
        bm25_rankings.append(_bm25_rank(bm25, corpus_ids, q, limit=40))

    dense_hits: list[dict[str, Any]] = []
    if settings.qdrant_url and settings.openai_api_key:
        chunk_records = [
            {"paper_id": pid, "chunk_index": idx, "text": txt} for pid, idx, txt in loaded
        ]
        if all((c.get("text") or "").strip() for c in chunk_records):
            texts_for_embed = [c["text"][:8000] for c in chunk_records]
            vectors = await embed_texts(texts=texts_for_embed, api_key=settings.openai_api_key, model=settings.embed_model)
            if vectors and len(vectors) == len(chunk_records):
                await asyncio.to_thread(
                    lambda: upsert_chunks(
                        url=settings.qdrant_url,
                        collection=settings.qdrant_collection,
                        session_id=session_id,
                        chunks=chunk_records,
                        vectors=vectors,
                    )
                )
                qvec = (await embed_texts(texts=[question], api_key=settings.openai_api_key, model=settings.embed_model))[0]
                dense_hits = await asyncio.to_thread(
                    lambda: search_chunks(
                        url=settings.qdrant_url,
                        collection=settings.qdrant_collection,
                        session_id=session_id,
                        vector=qvec,
                        limit=40,
                    )
                )

    dense_ranking = [f"{h['paper_id']}|{h['chunk_index']}" for h in dense_hits if h.get("paper_id") is not None]
    all_rankings = [*bm25_rankings]
    if dense_ranking:
        all_rankings.append(dense_ranking)

    merged = _rrf_merge(all_rankings)
    top_ids = [cid for cid, _ in merged[:20]]
    id_to_chunk = {f"{pid}|{idx}": (pid, idx, txt) for pid, idx, txt in loaded}
    chunks_out: list[dict[str, Any]] = []
    for cid in top_ids:
        row = id_to_chunk.get(cid)
        if not row:
            continue
        pid, idx, txt = row
        chunks_out.append({"paper_id": pid, "chunk_index": idx, "text": txt[:4000]})

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_RETRIEVAL_RESULTS,
        step="literature.retrieval",
        payload={"chunk_count": len(chunks_out), "chunks": [{"paper_id": c["paper_id"], "chunk_index": c["chunk_index"]} for c in chunks_out]},
    )
    return chunks_out
