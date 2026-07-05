"""FastAPI app — the literature intelligence service's HTTP surface.

Every route is a thin wrapper over ``LiteraturePipeline``; no business logic
lives here. See ``docs/propab_ownership_contracts.md`` for the input/output
contract this API is bound to.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from services.literature.app.config import settings
from services.literature.app.models import (
    CoverageResponse,
    GapsRequest,
    GapsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    NoveltyRequest,
    NoveltyResponse,
    PriorRequest,
    PriorResponse,
)
from services.literature.app.pipeline import LiteraturePipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = await LiteraturePipeline.create(settings)
    yield
    await app.state.pipeline.aclose()


app = FastAPI(
    title="Propab Literature Intelligence Service",
    version="0.1.0",
    description=(
        "Structured knowledge about what is known, contested, and unknown in a "
        "research domain — sourced from published papers and authoritative sources. "
        "Owns nothing about hypotheses, campaigns, or domain-specific sequences; "
        "all domain knowledge arrives via each request's literature_profile."
    ),
    lifespan=lifespan,
)


@app.post("/prior", response_model=PriorResponse)
async def prior(req: PriorRequest) -> PriorResponse:
    if not req.research_question.strip():
        raise HTTPException(status_code=422, detail="research_question must not be empty")
    return await app.state.pipeline.build_prior(
        research_question=req.research_question,
        domain_id=req.domain_id,
        profile=req.literature_profile,
        depth=req.depth,
    )


@app.post("/novelty", response_model=NoveltyResponse)
async def novelty(req: NoveltyRequest) -> NoveltyResponse:
    if not req.finding.claim.strip():
        raise HTTPException(status_code=422, detail="finding.claim must not be empty")
    return await app.state.pipeline.check_novelty(finding=req.finding, profile=req.literature_profile)


@app.post("/gaps", response_model=GapsResponse)
async def gaps(req: GapsRequest) -> GapsResponse:
    return await app.state.pipeline.map_gaps(domain_id=req.domain_id, profile=req.literature_profile)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    if not req.doi_or_arxiv_id.strip():
        raise HTTPException(status_code=422, detail="doi_or_arxiv_id must not be empty")
    return await app.state.pipeline.ingest(
        doi_or_arxiv_id=req.doi_or_arxiv_id, domain_id=req.domain_id, profile=req.literature_profile
    )


@app.get("/coverage", response_model=CoverageResponse)
async def coverage() -> CoverageResponse:
    return await app.state.pipeline.coverage()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return await app.state.pipeline.health()
