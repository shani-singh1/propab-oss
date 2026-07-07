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
import time
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

# Standard-depth (cold, in-budget) caps. The standard path never downloads a
# full-text PDF (see run_query_pipeline / _process_abstract): it builds claims
# from the search hit's own title + abstract, so the only cost is search +
# lightweight extraction. Cap the fan-out so a cold /prior stays well under the
# 60s standard budget even when several sources are slow: consult only the
# first few sources in the domain's declared priority order, and only the top
# few hits per source.
_STANDARD_MAX_SOURCES = 4
_STANDARD_MAX_DOCS_PER_SOURCE = 5


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


def raw_to_abstract_document(raw_doc: RawDocument) -> FullTextDocument:
    """Turn a search hit into an ``abstract_only`` FullTextDocument WITHOUT any
    network call — no PDF/LaTeX download. Every field is already on the
    ``RawDocument`` the source's ``search`` returned (title, authors, year,
    doi, url, abstract). The abstract becomes ``body_text`` so the existing
    extractors — the regex/signal-phrase ClaimsExtractor and, when no regex
    claim is found and an LLM key is set, the code-verbatim locate_claims path
    — run over it exactly as they do for the arXiv/PubMed abstract_only
    fallback. Title is prepended so a claim stated only in the title is still
    reachable by the linguistic scan."""
    title = (raw_doc.title or "").strip()
    abstract = (raw_doc.abstract or "").strip()
    if title and abstract:
        body = f"{title}.\n\n{abstract}"
    else:
        body = title or abstract
    return FullTextDocument(
        source=raw_doc.source,
        external_id=raw_doc.external_id,
        title=raw_doc.title,
        authors=raw_doc.authors,
        year=raw_doc.year,
        doi=raw_doc.doi,
        url=raw_doc.url,
        body_text=body,
        extraction_method="abstract_only",
        extraction_quality=0.0,
    )


async def _process_abstract(ctx: PipelineContext, raw_doc: RawDocument) -> dict[str, list] | None:
    """Standard-depth per-doc work: extract claims from the search hit's
    abstract/title only. Never calls ``source.fetch_full_text`` — that is the
    slow (PDF download + parse) path that blows the standard budget. Docs whose
    abstract yields no usable text still cost nothing beyond the extractors."""
    doc = raw_to_abstract_document(raw_doc)
    if not doc.body_text.strip():
        return None
    result = await process_document(ctx, doc)
    result["_doc"] = doc  # type: ignore[assignment]
    return result


def _order_sources_by_priority(
    relevant_sources: dict[str, Any], profile: dict[str, Any]
) -> list[str]:
    """Source names in the domain's declared ``source_priorities`` order first
    (so the standard path spends its small budget on the sources the domain
    trusts most), with any relevant-but-unlisted sources appended in a stable
    order after them."""
    priorities = [p.lower() for p in profile.get("source_priorities", []) or []]
    ordered = [p for p in priorities if p in relevant_sources]
    ordered += [n for n in sorted(relevant_sources) if n not in ordered]
    return ordered


async def run_query_pipeline(
    ctx: PipelineContext,
    *,
    research_question: str,
    domain_id: str,
    profile: dict[str, Any],
    depth: str = "standard",
    deadline_sec: float | None = None,
) -> PriorResponse:
    """Retrieve knowledge and structure it into a ``/prior`` response.

    ``standard`` depth is the cold, in-budget path a campaign launch calls: it
    NEVER downloads a full-text PDF. It searches the top few priority sources,
    then builds claims from each hit's own title + abstract (see
    ``_process_abstract``) — fast enough to return a real, if lighter, prior
    well inside the 60s budget. ``deep``/``exhaustive`` keep the full-text +
    citation-crawl behavior unchanged (offline/seed use, larger budgets).

    ``deadline_sec`` is a soft wall-clock budget: whatever docs finished
    processing when it is hit are synthesized into a (partial) prior rather than
    discarded. ``sources_consulted``/``papers_indexed`` always reflect what was
    actually consulted, never an aspirational list.
    """
    _decomposed = decompose_question(research_question)  # inspectable side-channel; kept for parity
    max_docs = ctx.depth_docs.get(depth, ctx.depth_docs["standard"])
    is_standard = depth == "standard"
    start = time.monotonic()

    def _time_left() -> float | None:
        if deadline_sec is None:
            return None
        return deadline_sec - (time.monotonic() - start)

    relevant_sources = {
        name: src for name, src in ctx.sources.items()
        if src.is_relevant(profile) and name not in ("crossref",)  # crossref is enrichment, not primary search
    }

    # Standard: cap fan-out to the top priority sources so a cold /prior stays
    # in-budget. Deep/exhaustive consult everything relevant.
    if is_standard:
        ordered = _order_sources_by_priority(relevant_sources, profile)[:_STANDARD_MAX_SOURCES]
        search_source_names = ordered
        per_source_docs = min(max_docs, _STANDARD_MAX_DOCS_PER_SOURCE)
    else:
        search_source_names = list(relevant_sources)
        per_source_docs = max_docs

    async def _search(name: str, src) -> tuple[str, list[RawDocument]]:
        try:
            docs = await src.search(research_question, profile)
        except Exception:
            docs = []
        return name, docs[:per_source_docs]

    # Search fan-out honors the deadline too: a source whose ``search`` hangs
    # must not block the whole /prior. Searches that don't return in time are
    # cancelled and simply contribute nothing (they are NOT reported as
    # consulted, keeping ``sources_consulted`` honest).
    search_tasks = [
        asyncio.ensure_future(_search(n, relevant_sources[n])) for n in search_source_names
    ]
    search_results: list[tuple[str, list[RawDocument]]] = [
        r for r in await _gather_with_deadline(search_tasks, _time_left) if r
    ]

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

    # Which sources we actually searched — the honest ``sources_consulted``.
    # (A source with a search error still counts as consulted; it just yielded
    # nothing. Sources dropped by the standard-path cap are NOT claimed.)
    consulted = sorted({name for name, _ in search_results})

    # Per-doc work. Standard: abstract-only, no full-text download. Deep/
    # exhaustive: the full-text fetch + extraction path, unchanged.
    per_doc_tasks: list[asyncio.Task] = []
    for name, docs in search_results:
        for raw_doc in docs:
            if is_standard:
                coro = _process_abstract(ctx, raw_doc)
            else:
                coro = _fetch_and_process(ctx, name, raw_doc)
            per_doc_tasks.append(asyncio.ensure_future(coro))

    # Give the per-doc phase a small floor so fast abstract extraction still
    # runs to completion even if the search phase consumed most of the budget;
    # a hung per-doc task (e.g. a stalled LLM locate) is still cut off shortly
    # after. Deep/exhaustive full-text fetches have no floor (their budgets are
    # large) so a slow download can't drag past the deadline.
    per_doc_floor = 3.0 if is_standard else 0.0
    processed = await _gather_with_deadline(per_doc_tasks, _time_left, floor_sec=per_doc_floor)

    all_claims: list[ExtractedClaim] = []
    all_tables: list[TabulatedSequence] = []
    all_open_problems: list[OpenProblem] = []
    docs_indexed: list[FullTextDocument] = []
    for r in processed:
        if not r:
            continue
        all_claims.extend(r["claims"])
        all_tables.extend(r["tables"])
        all_open_problems.extend(r["open_problems"])
        docs_indexed.append(r["_doc"])

    return await _synthesize_prior(
        ctx,
        domain_id=domain_id,
        profile=profile,
        all_claims=all_claims,
        all_tables=all_tables,
        all_open_problems=all_open_problems,
        docs_indexed=docs_indexed,
        consulted=consulted,
    )


async def _gather_with_deadline(
    tasks: list[asyncio.Task], time_left: Any, *, floor_sec: float = 0.0
) -> list[Any]:
    """Await ``tasks``, returning results of whichever completed. If a soft
    deadline (``time_left()`` returns a number of seconds) elapses first, stop
    waiting and cancel the rest — the caller synthesizes a prior from the
    partial set rather than getting nothing. With no deadline this is a plain
    ``gather``. A task that raised contributes ``None`` (dropped by callers).

    ``floor_sec`` guarantees a minimum window even when the budget is already
    spent, so a phase of fast CPU-bound tasks (abstract extraction) still gets
    scheduled and completes instead of being cancelled before it runs. A truly
    hung task is still cut off within ``floor_sec`` of the deadline."""
    if not tasks:
        return []
    remaining = time_left()
    if remaining is None:
        results: list[Any] = []
        for r in await asyncio.gather(*tasks, return_exceptions=True):
            results.append(None if isinstance(r, BaseException) else r)
        return results
    wait_for = max(remaining, floor_sec)
    done, pending = await asyncio.wait(tasks, timeout=wait_for)
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    results = []
    for t in done:
        try:
            results.append(t.result())
        except Exception:
            results.append(None)
    return results


async def _synthesize_prior(
    ctx: PipelineContext,
    *,
    domain_id: str,
    profile: dict[str, Any],
    all_claims: list[ExtractedClaim],
    all_tables: list[TabulatedSequence],
    all_open_problems: list[OpenProblem],
    docs_indexed: list[FullTextDocument],
    consulted: list[str],
) -> PriorResponse:
    """Build the PriorResponse from whatever was gathered — used by both the
    normal completion path and the partial-on-deadline path, so a timeout still
    returns a real prior with the claims/tables collected so far."""
    # Embed all claims (Step 5), then dedup (Step 6).
    if all_claims:
        embeddings = await ctx.embedder.embed([c.verbatim for c in all_claims])
        for c, emb in zip(all_claims, embeddings):
            c.embedding = emb
    all_claims = dedup_claims(all_claims, ctx.dedup_similarity_threshold)

    contradictions = await ctx.contradictions_extractor.find_contradictions(all_claims)
    gaps = await ctx.gaps_extractor.find_gaps(all_claims, all_open_problems)

    # OEIS tabulation overlay for domains that declare OEIS sequences.
    all_tables = list(all_tables)
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
        sources_consulted=consulted,
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
