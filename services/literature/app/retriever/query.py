"""
Query pipeline — given a research question + domain profile, retrieve
established knowledge, contradictions, gaps, and tabulated values, and
structure them into the ``/prior`` output contract.

This is the component the campaign-launch integration calls first. Every
step is domain-agnostic: search terms, classification codes, and which
sources matter all come from the domain profile — this module only knows
how to run the pipeline generically.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from services.literature.app.context import PipelineContext
from services.literature.app.extractors._bounds import parse_bounds
from services.literature.app.extractors.claims import extract_bibliography_annotations
from services.literature.app.extractors.llm_claim_locator import locate_claims
from services.literature.app.models import (
    ExtractedClaim,
    FullTextDocument,
    KnowledgeGap,
    NoveltyBar,
    OpenProblem,
    PriorResponse,
    RawDocument,
    TabulatedSequence,
)

_CLAIM_TYPE_KEYWORDS = {
    "existence": ("exist", "there is", "there exists", "construction of"),
    "density": ("density", "asymptotic", "growth rate", "fraction of"),
    "threshold": ("threshold", "phase transition", "critical value", "at least", "at most"),
    "comparison": ("compare", "versus", "better than", "outperform", "larger than", "smaller than"),
    "structural": ("structure", "invariant", "property", "characteriz"),
}
_SCOPE_RE = re.compile(r"\b(?:n|q|dimension)\s*(?:∈|in|up to|<=|≤)\s*\[?(\d+(?:,\s*\d+)?)\]?", re.I)


def decompose_question(question: str) -> dict[str, Any]:
    """Structured components: subject, property, claim_type, scope. A
    lightweight heuristic — good enough to drive query generation and to be
    inspectable/debuggable, not an NLP model that can't be evaluated."""
    lower = question.lower()
    claim_type = "structural"
    for ctype, keywords in _CLAIM_TYPE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            claim_type = ctype
            break
    scope_m = _SCOPE_RE.search(question)
    scope = scope_m.group(1) if scope_m else ""
    return {
        "subject": question.strip(),
        "property": claim_type,
        "claim_type": claim_type,
        "scope": scope,
    }


def dedup_claims(claims: list[ExtractedClaim], threshold: float) -> list[ExtractedClaim]:
    """Claims with cosine similarity > threshold are the same claim from
    different sources — keep the one with the more precise source (has a
    DOI/arxiv id, longer verbatim as a proxy for specificity)."""
    from services.literature.app.indexer.embeddings import cosine_similarity

    kept: list[ExtractedClaim] = []
    for claim in claims:
        replaced = False
        for i, existing in enumerate(kept):
            if not claim.embedding or not existing.embedding:
                continue
            if cosine_similarity(claim.embedding, existing.embedding) > threshold:
                if _specificity(claim) > _specificity(existing):
                    kept[i] = claim
                replaced = True
                break
        if not replaced:
            kept.append(claim)
    return kept


def _specificity(c: ExtractedClaim) -> tuple:
    return (bool(c.source_doi or c.source_url), len(c.verbatim))


async def process_document(ctx: PipelineContext, doc: FullTextDocument) -> dict[str, list]:
    claims = await ctx.claims_extractor.extract(doc)
    claims += await extract_bibliography_annotations(doc)
    if not claims and ctx.llm_api_key:
        # Regex/signal-phrase extraction found nothing — measured live, this
        # is the normal case for abstract-only sources (PubMed, bioRxiv):
        # their prose rarely contains "we show"/"it is known that"-style
        # hedges. The LLM locator only ever supplies a sentence *index*;
        # verbatim text is always the code-side lookup at that index (see
        # extractors/llm_claim_locator.py docstring) — it cannot paraphrase
        # its way into a fabricated citation. Gated on llm_api_key so the
        # rest of the pipeline is unaffected when it's unset.
        claims += await locate_claims(doc, api_key=ctx.llm_api_key, model=ctx.llm_model)
    tables = await ctx.tables_extractor.extract(doc)
    open_problems = await ctx.open_problems_extractor.extract(doc)
    return {"claims": claims, "tables": tables, "open_problems": open_problems}


async def _fetch_and_process(ctx: PipelineContext, source_name: str, raw_doc: RawDocument) -> dict[str, list] | None:
    source = ctx.sources.get(source_name)
    if source is None:
        return None
    try:
        full_doc = await source.fetch_full_text(raw_doc)
    except Exception:
        return None
    result = await process_document(ctx, full_doc)
    result["_doc"] = full_doc  # type: ignore[assignment]
    return result


async def run_query_pipeline(
    ctx: PipelineContext,
    *,
    research_question: str,
    domain_id: str,
    profile: dict[str, Any],
    depth: str = "standard",
) -> PriorResponse:
    decomposed = decompose_question(research_question)
    max_docs = ctx.depth_docs.get(depth, ctx.depth_docs["standard"])

    relevant_sources = {
        name: src for name, src in ctx.sources.items()
        if src.is_relevant(profile) and name not in ("crossref",)  # crossref is enrichment, not primary search
    }

    async def _search(name: str, src) -> tuple[str, list[RawDocument]]:
        try:
            docs = await src.search(research_question, profile)
        except Exception:
            docs = []
        return name, docs[:max_docs]

    search_results = await asyncio.gather(*(_search(n, s) for n, s in relevant_sources.items()))

    # Two-level citation-depth crawl from seed papers (deep/exhaustive only).
    if depth in ("deep", "exhaustive") and "semantic_scholar" in ctx.sources:
        s2 = ctx.sources["semantic_scholar"]
        seed_docs: list[RawDocument] = []
        for seed in profile.get("seed_papers", []) or []:
            paper_id = f"arXiv:{seed['arxiv_id']}" if seed.get("arxiv_id") else seed.get("doi", "")
            if not paper_id:
                continue
            try:
                citing = await s2.citations_of(paper_id)
                cited = await s2.references_of(paper_id)
            except Exception:
                citing, cited = [], []
            seed_docs.extend(citing[: max_docs] + cited[: max_docs])
        if seed_docs:
            search_results.append(("semantic_scholar", seed_docs))

    fetch_tasks = []
    for name, docs in search_results:
        for raw_doc in docs:
            fetch_tasks.append(_fetch_and_process(ctx, name, raw_doc))
    processed = [r for r in await asyncio.gather(*fetch_tasks) if r]

    all_claims: list[ExtractedClaim] = []
    all_tables: list[TabulatedSequence] = []
    all_open_problems: list[OpenProblem] = []
    docs_indexed: list[FullTextDocument] = []
    for r in processed:
        all_claims.extend(r["claims"])
        all_tables.extend(r["tables"])
        all_open_problems.extend(r["open_problems"])
        docs_indexed.append(r["_doc"])

    # Embed all claims (Step 5), then dedup (Step 6).
    if all_claims:
        embeddings = await ctx.embedder.embed([c.verbatim for c in all_claims])
        for c, emb in zip(all_claims, embeddings):
            c.embedding = emb
    all_claims = dedup_claims(all_claims, ctx.dedup_similarity_threshold)

    contradictions = await ctx.contradictions_extractor.find_contradictions(all_claims)
    gaps = await ctx.gaps_extractor.find_gaps(all_claims, all_open_problems)

    # OEIS tabulation overlay for domains that declare OEIS sequences.
    all_tables.extend(await _oeis_tabulations(ctx, profile))

    established_facts = [c for c in all_claims if c.status == "proven"]
    dead_ends = [c for c in all_claims if c.status == "disproven"]

    novelty_bar = _build_novelty_bar(profile, all_tables, established_facts)

    # Persist (best-effort — a storage hiccup should not fail a /prior call).
    try:
        await ctx.structured_store.save_claims(domain_id, all_claims)
        await ctx.structured_store.save_tables(domain_id, all_tables)
        await ctx.structured_store.save_open_problems(domain_id, all_open_problems)
        for doc in docs_indexed:
            await ctx.structured_store.mark_paper_indexed(domain_id, doc.source, doc.external_id)
        await ctx.vector_store.upsert(all_claims)
    except Exception:
        pass

    return PriorResponse(
        established_facts=established_facts,
        open_gaps=gaps,
        contradictions=contradictions,
        dead_ends=dead_ends,
        tabulated_values=all_tables,
        novelty_bar=novelty_bar,
        sources_consulted=sorted(relevant_sources.keys()),
        papers_indexed=len(docs_indexed),
        citation_verification_rate=None,
    )


async def _oeis_tabulations(ctx: PipelineContext, profile: dict[str, Any]) -> list[TabulatedSequence]:
    oeis = ctx.sources.get("oeis")
    if oeis is None:
        return []
    tab_sources = profile.get("tabulation_sources", []) or []
    out: list[TabulatedSequence] = []
    for tab in tab_sources:
        if tab.get("name", "").lower() != "oeis":
            continue
        ids = tab.get("identifiers", []) or []
        await oeis.warm_cache(ids)
        for seq_id in ids:
            seq = oeis._sequence_cache.get(seq_id)
            if not seq:
                continue
            data = [int(v) for v in str(seq.get("data", "")).split(",") if v.strip().lstrip("-").isdigit()]
            if not data:
                continue
            offset = int(str(seq.get("offset", "0,1")).split(",")[0])
            values = {str(offset + i): v for i, v in enumerate(data)}
            out.append(
                TabulatedSequence(
                    description=seq.get("name", seq_id),
                    index_variable="n",
                    value_variable=seq_id,
                    values=values,
                    max_index=float(offset + len(data) - 1),
                    min_index=float(offset),
                    location=f"OEIS {seq_id}",
                    oeis_match=seq_id,
                    extraction_confidence=1.0,
                )
            )
    return out


def _build_novelty_bar(
    profile: dict[str, Any], tables: list[TabulatedSequence], established_facts: list[ExtractedClaim]
) -> NoveltyBar:
    ceiling = {t.value_variable or t.description: t.max_index for t in tables if t.max_index is not None}
    bounds = []
    for c in established_facts:
        if parse_bounds(c.verbatim):
            bounds.append(c.verbatim)
    criteria = profile.get("novelty_criteria") or (
        "A finding is novel if it is not present in any tabulated source and not "
        "directly implied by an established bound listed above."
    )
    return NoveltyBar(criteria=criteria, tabulated_ceiling=ceiling, established_bounds=bounds[:20])
