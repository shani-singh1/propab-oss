"""
Regression tests for the /prior open_gaps overlay.

Root cause these lock down: for real questions the fetched corpus is
abstract-heavy — every extracted claim is "proven" and no paper body carries a
"Problem:"/conjecture marker — so GapsExtractor.find_gaps returns []. Before the
fix, PriorResponse.open_gaps was therefore empty even for domains with a
well-known open frontier. The fix overlays the domain's own frontier onto
open_gaps via two generic, profile-driven signals:

  1. scrape of profile["open_problem_sources"][].url  (inside run_query_pipeline)
  2. a bounded "<term> open problem" live search       (in pipeline.build_prior,
     under its own separate deadline so it can never blank the core prior)
"""
from __future__ import annotations

import pytest

from services.literature.app.context import PipelineContext
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.postgres_store import InMemoryStructuredStore
from services.literature.app.indexer.qdrant_store import InMemoryVectorStore
from services.literature.app.models import FullTextDocument, RawDocument
from services.literature.app.retriever import gap_mapper
from services.literature.app.retriever.query import (
    augment_open_gaps_with_search,
    run_query_pipeline,
)
from services.literature.app.sources.base import BaseSource


class ProvenOnlySource(BaseSource):
    """Mirrors the live failure mode: normal queries return an abstract-only,
    proven-only paper (no markers → find_gaps yields nothing); the dedicated
    "<term> open problem" query returns a survey with an explicit open problem."""

    name = "arxiv"

    def is_relevant(self, profile) -> bool:
        return True

    async def search(self, query, profile):
        if "open problem" in query.lower():
            return [RawDocument(source="arxiv", external_id="survey")]
        return [RawDocument(source="arxiv", external_id="result")]

    async def fetch_full_text(self, doc):
        if doc.external_id == "survey":
            return FullTextDocument(
                source="arxiv",
                external_id="survey",
                title="A Survey",
                year=2016,
                body_text=(
                    "Open problem: determine by exhaustive computational search the exact "
                    "extremal value f(n) for n >= 8."
                ),
            )
        return FullTextDocument(
            source="arxiv",
            external_id="result",
            title="A Result",
            year=2022,
            body_text="We prove that f(n) is at most 2.76^n for all n.",
        )

    async def check_tabulated(self, values):
        return []


def _ctx(source: BaseSource) -> PipelineContext:
    return PipelineContext(
        sources={source.name: source},
        embedder=EmbeddingClient(provider="offline", dim=32),
        vector_store=InMemoryVectorStore(),
        structured_store=InMemoryStructuredStore(),
    )


class TestOpenGapsSearchAugmentation:
    @pytest.mark.asyncio
    async def test_core_prior_has_no_gaps_from_proven_only_corpus(self):
        # Sanity: without the augmentation the proven-only corpus yields no gaps.
        ctx = _ctx(ProvenOnlySource())
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        resp = await run_query_pipeline(
            ctx, research_question="what is f(n)?", domain_id="d", profile=profile
        )
        assert resp.established_facts, "expected the proven claim to be extracted"
        assert resp.open_gaps == [], "proven-only corpus should leave find_gaps empty"

    @pytest.mark.asyncio
    async def test_augment_populates_open_gaps_from_search(self):
        ctx = _ctx(ProvenOnlySource())
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        extra = await augment_open_gaps_with_search(
            ctx, profile, existing_gaps=[], timeout=10.0
        )
        assert extra, "bounded open-problem search should surface the survey's open problem"
        assert any("f(n)" in g.what_is_open for g in extra)
        assert any(g.computationally_approachable for g in extra)

    @pytest.mark.asyncio
    async def test_augment_dedups_against_existing_gaps(self):
        from services.literature.app.models import KnowledgeGap

        ctx = _ctx(ProvenOnlySource())
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        # First run discovers the problem; capture the exact (extractor-cleaned)
        # statement so the second run can be seeded with it.
        first = await augment_open_gaps_with_search(
            ctx, profile, existing_gaps=[], timeout=10.0
        )
        assert first, "expected the first pass to discover the open problem"
        existing = [KnowledgeGap(description="dup", what_is_open=first[0].what_is_open)]
        extra = await augment_open_gaps_with_search(
            ctx, profile, existing_gaps=existing, timeout=10.0
        )
        assert extra == [], "already-known statement must not be re-added"

    @pytest.mark.asyncio
    async def test_augment_survives_zero_timeout(self):
        # A blown deadline degrades to "no extra gaps", never an exception.
        ctx = _ctx(ProvenOnlySource())
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        extra = await augment_open_gaps_with_search(
            ctx, profile, existing_gaps=[], timeout=0.0
        )
        assert extra == []


class TestBuildPriorWiring:
    @pytest.mark.asyncio
    async def test_build_prior_merges_search_gaps_into_open_gaps(self):
        from services.literature.app.config import Settings
        from services.literature.app.pipeline import LiteraturePipeline

        ctx = _ctx(ProvenOnlySource())
        pipeline = LiteraturePipeline(Settings(), ctx)
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        resp = await pipeline.build_prior(
            research_question="what is f(n)?", domain_id="d", profile=profile
        )
        # Core prior intact AND open_gaps populated by the augmentation step.
        assert resp.established_facts
        assert resp.open_gaps, "build_prior should merge search-discovered gaps into open_gaps"
        assert any("f(n)" in g.what_is_open for g in resp.open_gaps)

    @pytest.mark.asyncio
    async def test_slow_augmentation_never_blanks_core_prior(self, monkeypatch):
        # If the augmentation raises/hangs, the already-built core prior returns intact.
        from services.literature.app.config import Settings
        from services.literature.app import pipeline as pipeline_mod
        from services.literature.app.pipeline import LiteraturePipeline

        async def boom(*a, **k):
            raise RuntimeError("augmentation exploded")

        monkeypatch.setattr(pipeline_mod, "augment_open_gaps_with_search", boom)
        ctx = _ctx(ProvenOnlySource())
        pipeline = LiteraturePipeline(Settings(), ctx)
        profile = {"search_terms": ["extremal value"], "source_priorities": ["arxiv"]}
        resp = await pipeline.build_prior(
            research_question="what is f(n)?", domain_id="d", profile=profile
        )
        assert resp.established_facts, "core prior must survive a failing augmentation step"


class TestDeclaredOpenProblemScrape:
    @pytest.mark.asyncio
    async def test_scrape_declared_sources_parses_markers(self, monkeypatch):
        html = (
            "<html><body>"
            "Open problem: is every f(n) eventually 1 for large n?\n\n"
            "Problem 2: bound g(n) from below.\n\n"
            "</body></html>"
        )

        async def fake_scrape(url, *, timeout=15.0):
            # Exercise the real marker regex via the module-level helper on our HTML.
            import re

            from services.literature.app.extractors.open_problems import PROBLEM_MARKER_RE
            from services.literature.app.models import OpenProblem

            text = re.sub(r"<[^>]+>", " ", html)
            out = []
            for m in PROBLEM_MARKER_RE.finditer(text):
                stmt = re.sub(r"\s+", " ", m.group(2)).strip()[:400]
                if len(stmt) > 15:
                    out.append(OpenProblem(statement=stmt, context=f"scraped from {url}"))
            return out

        monkeypatch.setattr(gap_mapper, "_scrape_open_problem_source", fake_scrape)
        profile = {"open_problem_sources": [{"name": "list", "url": "https://x.test/"}]}
        problems = await gap_mapper.scrape_declared_open_problem_sources(profile)
        assert len(problems) >= 1
        assert any("f(n)" in p.statement or "g(n)" in p.statement for p in problems)

    @pytest.mark.asyncio
    async def test_scrape_ignores_entries_without_url(self):
        profile = {"open_problem_sources": [{"name": "no url here"}]}
        problems = await gap_mapper.scrape_declared_open_problem_sources(profile)
        assert problems == []
