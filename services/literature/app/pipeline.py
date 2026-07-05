"""
Top-level orchestration: wires sources, extractors, and storage together and
exposes the five operations the FastAPI routes need. This is the only module
that constructs concrete source/store instances — everything downstream
(``retriever/*``) only ever sees the abstract ``PipelineContext``.
"""
from __future__ import annotations

import asyncio
import re
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
from services.literature.app.retriever.query import process_document, run_query_pipeline
from services.literature.app.sources.arxiv import ArxivSource, normalize_arxiv_id
from services.literature.app.sources.biorxiv import BiorxivSource
from services.literature.app.sources.crossref import CrossrefSource
from services.literature.app.sources.europepmc import EuropePmcSource
from services.literature.app.sources.mathoverflow import MathOverflowSource
from services.literature.app.sources.oeis import OeisSource
from services.literature.app.sources.pubmed import PubmedSource
from services.literature.app.sources.semantic_scholar import SemanticScholarSource
from services.literature.app.sources.zbmath import ZbmathSource

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
        timeout = self.settings.depth_timeout_sec.get(depth, 60)
        try:
            return await asyncio.wait_for(
                run_query_pipeline(
                    self.ctx,
                    research_question=research_question,
                    domain_id=domain_id,
                    profile=profile,
                    depth=depth,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # A slow source must never fail the whole request — return
            # whatever partial structure is safe (empty), the caller falls
            # back to the domain plugin's built-in prior.
            from services.literature.app.models import NoveltyBar

            return PriorResponse(
                novelty_bar=NoveltyBar(criteria=profile.get("novelty_criteria", "")),
                sources_consulted=[],
                papers_indexed=0,
            )

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
