"""
Chunk-level embedding retrieval (RAG).

Replaces "keyword-search, then hand the LLM whatever chunk of text comes
back first" with the PaperQA2-style pattern: search → fetch full text →
split into paragraph-sized chunks → embed every chunk → rank chunks by
cosine similarity to the *actual question*, not the search query used to
find the document. This is strictly more informative than either of this
service's two earlier evidence paths for QA:

- Raw truncated document text (``evaluator/litqa2_live.retrieve_evidence_documents``)
  always hands over the first ~700 characters of a document regardless of
  where in that document the actual answer lives — the wrong choice for a
  paper whose relevant result is in Results/Methods/a table, not the first
  paragraph.
- Structured claims (``established_facts`` via ``extractors/llm_claim_locator.py``)
  are individual sentences stripped of surrounding context — measured live
  (CHANGELOG.md 0.4.0), this reduced answer precision relative to raw text,
  because QA benefits from the paragraph a fact sits in, not just the fact.

Chunking keeps paragraph-level context intact (unlike single-sentence
claims) while ranking by relevance to the question (unlike "first N chars
of a document"), so it should not have either failure mode.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from services.literature.app.indexer.embeddings import EmbeddingClient, cosine_similarity
from services.literature.app.models import FullTextDocument
from services.literature.app.sources.base import BaseSource

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class Chunk:
    text: str
    source: str
    title: str
    year: int
    url: str

    def format(self) -> str:
        return f"{self.title} ({self.source}, {self.year or 'n.d.'}): {self.text}"


def chunk_document(doc: FullTextDocument, *, max_chars: int = 900, min_chars: int = 80) -> list[Chunk]:
    """Paragraph-based chunking: split on ``\\n\\n``, then greedily merge
    consecutive paragraphs up to ``max_chars`` so short paragraphs (common in
    PMC's per-<p> splitting — see ``sources/pubmed.py``) don't each become
    their own near-empty chunk. A lone paragraph longer than ``max_chars``
    becomes its own chunk unsplit rather than being cut mid-sentence."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.body_text or "") if p.strip()]
    if not paragraphs:
        return []

    raw_chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}" if buf else p
        if len(candidate) <= max_chars or not buf:
            buf = candidate
        else:
            raw_chunks.append(buf)
            buf = p
    if buf:
        raw_chunks.append(buf)

    return [
        Chunk(text=c, source=doc.source, title=doc.title, year=doc.year, url=doc.url)
        for c in raw_chunks
        if len(c) >= min_chars
    ]


def rank_chunks_bm25(question: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Local lexical (BM25) ranking — no network, instant, no contention.

    This is the default ranker for the QA/eval path, and deliberately so:
    measured live, dense (Gemini) chunk embedding fired up to ~90 concurrent
    ``embedContent`` HTTP calls per question, and that burst was the root
    cause of the ~28% ``ReadTimeout`` rate that dominated every eval run
    (each timeout is scored as wrong, so 28% timeouts caps accuracy at 0.72
    before answer quality even matters — see CHANGELOG.md 0.5.0). BM25 needs
    zero network calls, so it removes that contention entirely. It is also a
    genuinely strong ranker for this task specifically: LitQA2 questions turn
    on specific named entities, genes, and numbers, and lexical overlap finds
    the chunk that literally contains them — exactly what dense similarity can
    smear over. Dense embedding ranking stays available via
    ``rerank_chunks`` for callers that want it (e.g. semantic dedup in the
    ``/prior`` pipeline, where contention is not the constraint)."""
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(c.text) for c in chunks]
    # A chunk with no alphanumeric tokens would break BM25's IDF math; keep
    # only rankable chunks, but never return empty if any chunk had text.
    rankable = [(c, toks) for c, toks in zip(chunks, tokenized) if toks]
    if not rankable:
        return chunks[:top_k]
    bm25 = BM25Okapi([toks for _, toks in rankable])
    scores = bm25.get_scores(_tokenize(question))
    ordered = sorted(zip((c for c, _ in rankable), scores), key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in ordered[:top_k]]


def _doc_relevance_scores(question: str, docs: list[Any]) -> list[float]:
    """BM25 relevance of each candidate document's (title + abstract) to the
    question — used to pick which documents are worth fetching full text for,
    before any full-text fetch happens."""
    if not docs:
        return []
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(f"{getattr(d, 'title', '')} {getattr(d, 'abstract', '')}") or ["_"] for d in docs]
    bm25 = BM25Okapi(corpus)
    return list(bm25.get_scores(_tokenize(question)))


async def retrieve_relevant_chunks(
    sources: dict[str, BaseSource],
    embedder: EmbeddingClient,
    *,
    question: str,
    search_terms: list[str],
    profile: dict[str, Any],
    max_docs_per_source: int = 12,
    max_docs_to_fetch: int = 12,
    max_chunks_per_doc: int = 4,
    top_k: int = 6,
    ranker: str = "bm25",
) -> tuple[list[Chunk], list[str]]:
    """Issue several targeted searches → rank candidate documents → fetch full
    text of the best few → chunk → rank chunks (per-document cap) by relevance
    to ``question``.

    ``search_terms`` is a list of *complete targeted query strings* (see
    ``evaluator/litqa2_live.reformulate_query_for_search``), each issued as
    its OWN separate search and the results unioned — the way a real
    retrieval agent runs several precise searches. This replaced (CHANGELOG
    0.7.0) a query[0]+OR-the-rest design that broadened every search into
    500K-result recency dumps that buried the specific target paper.

    Two-stage ranking (from a live 0.6.0 failure): LitQA2 questions are about
    ONE specific paper, so (1) rank candidate *documents* by title+abstract
    relevance and only fetch full text for the top ``max_docs_to_fetch``, and
    (2) cap chunks per document so no single paper dominates the evidence set.

    ``ranker`` selects local BM25 (default, no network) or dense embedding
    similarity for the chunk stage. Never raises — a single source's search/
    fetch failure is swallowed so one flaky API doesn't blank out others."""
    # Cap the number of distinct queries actually issued: each query fans out
    # to every source, so latency scales with len(queries) × n_sources. Three
    # targeted queries give strong recall without a 25-search-per-question
    # latency blowup on the n=100 eval.
    queries = (list(search_terms) or [question])[:3]
    # Each query is issued directly (relevance-sorted by the source) with NO
    # extra OR terms — the query string is already the targeted, complete
    # search. Base profile terms are dropped for the eval path so they can't
    # re-broaden a precise query.
    per_query_profile = dict(profile)
    per_query_profile["search_terms"] = []
    relevant = {name: src for name, src in sources.items() if src.is_relevant(dict(profile))}

    async def _search(name: str, src: BaseSource, q: str) -> tuple[str, list[Any]]:
        try:
            docs = await src.search(q, per_query_profile)
        except Exception:
            docs = []
        return name, docs[: max(2, max_docs_per_source // max(1, len(queries)))]

    tasks = [_search(n, s, q) for n, s in relevant.items() for q in queries]
    search_results = await asyncio.gather(*tasks)
    sources_with_hits = sorted({name for name, docs in search_results if docs})

    # Stage 1: union candidate documents across all queries+sources, dedup by
    # (source, external_id) and by doi so the same paper found via multiple
    # queries isn't fetched repeatedly; rank by title+abstract relevance.
    candidates: list[tuple[str, Any]] = []
    seen_docs: set[str] = set()
    for name, docs in search_results:
        for raw in docs:
            key = getattr(raw, "doi", "") or f"{name}:{getattr(raw, 'external_id', '')}"
            if key and key not in seen_docs:
                seen_docs.add(key)
                candidates.append((name, raw))
    if not candidates:
        return [], sources_with_hits
    scores = _doc_relevance_scores(question, [raw for _, raw in candidates])
    top_candidates = [
        (name, raw)
        for (name, raw), _ in sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    ][:max_docs_to_fetch]

    async def _fetch(name: str, raw_doc: Any) -> FullTextDocument | None:
        try:
            return await sources[name].fetch_full_text(raw_doc)
        except Exception:
            return None

    full_docs = [
        d for d in await asyncio.gather(*(_fetch(name, raw) for name, raw in top_candidates))
        if d is not None and (d.body_text or "").strip()
    ]

    # Stage 2: chunk each fetched doc, rank chunks *within each document*, and
    # keep only the best ``max_chunks_per_doc`` from each — so the final pool
    # is diverse across papers before the global rank/top_k.
    pooled: list[Chunk] = []
    for doc in full_docs:
        doc_chunks = chunk_document(doc)
        best = await rerank_chunks(embedder, question, doc_chunks, max_chunks_per_doc, ranker=ranker)
        pooled.extend(best)

    ranked = await rerank_chunks(embedder, question, pooled, top_k, ranker=ranker)
    return ranked, sources_with_hits


async def rerank_chunks(
    embedder: EmbeddingClient, question: str, chunks: list[Chunk], top_k: int, *, ranker: str = "bm25"
) -> list[Chunk]:
    """Standalone re-rank, used directly by multi-round retrieval (accumulate
    chunks across rounds first, then rank the combined pool once) as well as
    internally by ``retrieve_relevant_chunks`` above. ``ranker="bm25"`` (the
    default) is local and network-free; ``"embedding"`` uses dense Gemini
    similarity (higher-latency, contention-prone — see ``rank_chunks_bm25``)."""
    if not chunks:
        return []
    if ranker == "bm25":
        return rank_chunks_bm25(question, chunks, top_k)
    question_embedding = await embedder.embed_one(question)
    chunk_embeddings = await embedder.embed([c.text for c in chunks])
    scored = list(zip(chunks, chunk_embeddings))
    scored.sort(key=lambda pair: cosine_similarity(question_embedding, pair[1]), reverse=True)
    return [c for c, _ in scored[:top_k]]
