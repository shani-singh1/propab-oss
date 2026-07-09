"""
Top-level orchestration: wires sources, extractors, and storage together and
exposes the five operations the FastAPI routes need. This is the only module
that constructs concrete source/store instances — everything downstream
(``retriever/*``) only ever sees the abstract ``PipelineContext``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from services.literature.app.config import Settings
from services.literature.app.context import PipelineContext
from services.literature.app.extractors.claims import extract_bibliography_annotations
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.postgres_store import build_structured_store
from services.literature.app.indexer.qdrant_store import build_vector_store
from services.literature.app.models import (
    CoverageDomain,
    CoverageResponse,
    Finding,
    GapsResponse,
    HealthResponse,
    IngestResponse,
    NoveltyResponse,
    PriorResponse,
    RawDocument,
)
from services.literature.app.retriever.gap_mapper import map_gaps
from services.literature.app.retriever.novelty_check import check_novelty as _check_novelty
from services.literature.app.retriever.query import (
    augment_open_gaps_with_search,
    process_document,
    run_query_pipeline,
)
from services.literature.app.sources.arxiv import ArxivSource, normalize_arxiv_id
from services.literature.app.sources.biorxiv import BiorxivSource
from services.literature.app.sources.crossref import CrossrefSource
from services.literature.app.sources.europepmc import EuropePmcSource
from services.literature.app.sources.mathoverflow import MathOverflowSource
from services.literature.app.sources.oeis import OeisSource
from services.literature.app.sources.pubmed import PubmedSource
from services.literature.app.sources.semantic_scholar import SemanticScholarSource
from services.literature.app.sources.zbmath import ZbmathSource

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def build_sources(settings: Settings) -> dict[str, Any]:
    return {
        "arxiv": ArxivSource(
            cache_dir=settings.cache_dir,
            max_results=settings.arxiv_max_results_per_query,
            min_interval_sec=settings.arxiv_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "oeis": OeisSource(
            min_interval_sec=settings.oeis_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "semantic_scholar": SemanticScholarSource(
            api_key=settings.semantic_scholar_api_key,
            min_interval_sec=settings.semantic_scholar_min_interval_sec,
            max_citations_per_seed=settings.semantic_scholar_max_citations_per_seed,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "mathoverflow": MathOverflowSource(
            min_interval_sec=settings.stackexchange_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "zbmath": ZbmathSource(
            min_interval_sec=settings.zbmath_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "pubmed": PubmedSource(
            api_key=settings.ncbi_api_key,
            min_interval_sec=settings.pubmed_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "biorxiv": BiorxivSource(
            min_interval_sec=settings.biorxiv_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "europepmc": EuropePmcSource(
            min_interval_sec=settings.europepmc_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
        "crossref": CrossrefSource(
            min_interval_sec=settings.crossref_min_interval_sec,
            http_timeout=settings.http_timeout_sec,
            user_agent=settings.user_agent,
        ),
    }


class LiteraturePipeline:
    def __init__(self, settings: Settings, ctx: PipelineContext) -> None:
        self.settings = settings
        self.ctx = ctx
        # In-process /prior result cache (see Settings.prior_cache_*). Instance-
        # scoped: there is exactly one pipeline per running service, so this
        # serves "repeated identical builds within a run" while a fresh pipeline
        # (tests, restart) always starts empty. Values are (stored_at, prior).
        self._prior_cache: dict[str, tuple[float, PriorResponse]] = {}

    @classmethod
    async def create(cls, settings: Settings) -> "LiteraturePipeline":
        sources = build_sources(settings)
        embedder = EmbeddingClient(
            provider=settings.embed_provider,
            model=settings.embed_model,
            api_key=settings.openai_api_key,
            google_api_key=settings.google_api_key,
            dim=settings.embed_dim,
            http_timeout=settings.http_timeout_sec,
        )
        vector_store = await build_vector_store(
            backend=settings.qdrant_backend,
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dim=settings.embed_dim,
        )
        structured_store = build_structured_store(
            backend=settings.postgres_backend, database_url=settings.database_url
        )
        await structured_store.init()
        ctx = PipelineContext(
            sources=sources,
            embedder=embedder,
            vector_store=vector_store,
            structured_store=structured_store,
            novelty_similarity_floor=settings.novelty_similarity_floor,
            novelty_top_k=settings.novelty_top_k,
            novelty_confidence_verdict_floor=settings.novelty_confidence_verdict_floor,
            dedup_similarity_threshold=settings.dedup_similarity_threshold,
            llm_api_key=settings.google_api_key,
            llm_model=settings.llm_model,
        )
        return cls(settings, ctx)

    async def aclose(self) -> None:
        for source in self.ctx.sources.values():
            await source.aclose()

    async def build_prior(
        self, *, research_question: str, domain_id: str, profile: dict[str, Any], depth: str = "standard"
    ) -> PriorResponse:
        # Serve a recent identical build straight from cache (campaign restart,
        # orchestrator retry, two campaigns on the same question) — a repeated
        # /prior then costs ~0 instead of re-hitting every upstream.
        cache_key = self._prior_cache_key(research_question, domain_id, profile, depth)
        cached = self._prior_cache_get(cache_key)
        if cached is not None:
            return cached.model_copy(deep=True)

        timeout = self.settings.depth_timeout_sec.get(depth, 40)
        # The budget is passed INTO the pipeline as a soft deadline: when it is
        # hit, the pipeline synthesizes a prior from whatever docs/claims it has
        # gathered so far instead of throwing everything away. A slightly-slow
        # build now yields partial real results, not an empty response.
        # ``asyncio.wait_for`` remains as an outer hard safety net (with a small
        # grace margin over the soft deadline, so the soft path — which returns
        # partial results — normally fires first); on the rare hard timeout we
        # still degrade to an empty prior so the caller can fall back.
        try:
            prior = await asyncio.wait_for(
                run_query_pipeline(
                    self.ctx,
                    research_question=research_question,
                    domain_id=domain_id,
                    profile=profile,
                    depth=depth,
                    deadline_sec=timeout,
                ),
                timeout=timeout + 5,
            )
        except asyncio.TimeoutError:
            # Hard net tripped (synthesis itself stalled past the grace margin):
            # never fail the whole request — return an empty-but-valid prior and
            # let the caller fall back to the domain plugin's built-in prior.
            logger.warning(
                "literature build_prior hard-timed out after %.0fs (depth=%s, domain=%s); "
                "returning empty prior so the caller can fall back",
                timeout + 5,
                depth,
                domain_id,
            )
            return self._empty_prior(profile)
        except Exception:
            # Any unexpected failure inside the core build must degrade to an
            # empty prior too — a /prior call must never raise into the caller
            # and stall campaign start; the caller then falls back honestly.
            logger.exception(
                "literature build_prior failed unexpectedly (depth=%s, domain=%s); "
                "returning empty prior so the caller can fall back",
                depth,
                domain_id,
            )
            return self._empty_prior(profile)

        # Frontier-gap augmentation: a bounded "<term> open problem" live search
        # runs *after* the core prior has already been built, under its own
        # separate deadline (never the core-prior deadline). This is what makes
        # open_gaps non-empty for real questions whose fetched abstracts are all
        # proven-only. If it overruns or finds nothing, the core prior is
        # returned unchanged — the augmentation is strictly additive.
        try:
            extra = await augment_open_gaps_with_search(
                self.ctx,
                profile,
                existing_gaps=prior.open_gaps,
                timeout=self.settings.gap_augment_timeout_sec,
            )
            if extra:
                prior.open_gaps = list(prior.open_gaps) + extra
        except Exception:
            pass

        self._prior_cache_put(cache_key, prior)
        return prior

    # -- /prior result cache --------------------------------------------------

    def _empty_prior(self, profile: dict[str, Any]) -> PriorResponse:
        from services.literature.app.models import NoveltyBar

        return PriorResponse(
            novelty_bar=NoveltyBar(criteria=profile.get("novelty_criteria", "")),
            sources_consulted=[],
            papers_indexed=0,
        )

    def _prior_cache_key(
        self, research_question: str, domain_id: str, profile: dict[str, Any], depth: str
    ) -> str | None:
        try:
            profile_repr = json.dumps(profile, sort_keys=True, default=str)
        except Exception:
            return None  # unserializable profile → skip caching, build fresh
        return "\x1f".join((depth, domain_id, research_question.strip(), profile_repr))

    def _prior_cache_get(self, key: str | None) -> PriorResponse | None:
        if key is None or self.settings.prior_cache_ttl_sec <= 0:
            return None
        entry = self._prior_cache.get(key)
        if entry is None:
            return None
        stored_at, prior = entry
        if (time.monotonic() - stored_at) > self.settings.prior_cache_ttl_sec:
            self._prior_cache.pop(key, None)  # expired
            return None
        return prior

    def _prior_cache_put(self, key: str | None, prior: PriorResponse) -> None:
        if key is None or self.settings.prior_cache_ttl_sec <= 0:
            return
        # Only cache priors that actually carry content: a degraded/empty result
        # (upstream hiccup) must never be pinned for the TTL, so a later retry
        # can still recover real evidence.
        has_content = bool(
            prior.established_facts
            or prior.open_gaps
            or prior.contradictions
            or prior.dead_ends
            or prior.tabulated_values
            or prior.papers_indexed
        )
        if not has_content:
            return
        # Bound memory: drop the oldest entry once at capacity (dict preserves
        # insertion order).
        max_entries = max(1, int(self.settings.prior_cache_max_entries))
        while len(self._prior_cache) >= max_entries:
            oldest = next(iter(self._prior_cache), None)
            if oldest is None:
                break
            self._prior_cache.pop(oldest, None)
        self._prior_cache[key] = (time.monotonic(), prior.model_copy(deep=True))

    async def check_novelty(self, *, finding: Finding, profile: dict[str, Any]) -> NoveltyResponse:
        return await _check_novelty(self.ctx, finding, profile)

    async def map_gaps(self, *, domain_id: str, profile: dict[str, Any]) -> GapsResponse:
        return await map_gaps(self.ctx, domain_id=domain_id, profile=profile)

    async def ingest(self, *, doi_or_arxiv_id: str, domain_id: str, profile: dict[str, Any]) -> IngestResponse:
        identifier = doi_or_arxiv_id.strip()
        full_doc = None
        source_name = ""

        if _ARXIV_ID_RE.match(normalize_arxiv_id(identifier)) or "arxiv.org" in identifier:
            arxiv = self.ctx.sources["arxiv"]
            raw = RawDocument(source="arxiv", external_id=normalize_arxiv_id(identifier))
            full_doc = await arxiv.fetch_full_text(raw)
            source_name = "arxiv"
        else:
            crossref = self.ctx.sources["crossref"]
            resolved = await crossref.resolve_doi(identifier)
            if resolved is not None:
                full_doc = await crossref.fetch_full_text(resolved)
                source_name = "crossref"

        if full_doc is None:
            return IngestResponse(claims_extracted=0, tables_extracted=0, open_problems_found=0)

        result = await process_document(self.ctx, full_doc)
        claims = result["claims"] + await extract_bibliography_annotations(full_doc)
        if claims:
            embeddings = await self.ctx.embedder.embed([c.verbatim for c in claims])
            for c, emb in zip(claims, embeddings):
                c.embedding = emb

        await self.ctx.structured_store.save_claims(domain_id, claims)
        await self.ctx.structured_store.save_tables(domain_id, result["tables"])
        await self.ctx.structured_store.save_open_problems(domain_id, result["open_problems"])
        await self.ctx.structured_store.mark_paper_indexed(domain_id, source_name, full_doc.external_id)
        await self.ctx.vector_store.upsert(claims)

        return IngestResponse(
            claims_extracted=len(claims),
            tables_extracted=len(result["tables"]),
            open_problems_found=len(result["open_problems"]),
            extraction_quality=full_doc.extraction_quality,
            extraction_method=full_doc.extraction_method,
        )

    async def coverage(self) -> CoverageResponse:
        rows = await self.ctx.structured_store.coverage()
        return CoverageResponse(domains=[CoverageDomain(**r) for r in rows])

    async def health(self) -> HealthResponse:
        names = list(self.ctx.sources.keys())
        results = await asyncio.gather(
            *(self.ctx.sources[n].health() for n in names), return_exceptions=True
        )
        sources_healthy = {n: (r is True) for n, r in zip(names, results)}
        status = "ok" if any(sources_healthy.values()) else "degraded"
        return HealthResponse(status=status, citation_verification_rate=None, sources_healthy=sources_healthy)
