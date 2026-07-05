import pytest

from services.literature.app.context import PipelineContext
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.postgres_store import InMemoryStructuredStore
from services.literature.app.indexer.qdrant_store import InMemoryVectorStore
from services.literature.app.models import FullTextDocument, RawDocument
from services.literature.app.retriever.query import run_query_pipeline
from services.literature.app.sources.base import BaseSource


class StubSource(BaseSource):
    """A fake source with canned search/fetch results — no network at all."""

    name = "arxiv"

    def __init__(self, raw_docs: list[RawDocument], full_texts: dict[str, FullTextDocument]) -> None:
        super().__init__()
        self._raw_docs = raw_docs
        self._full_texts = full_texts

    def is_relevant(self, profile) -> bool:
        return True

    async def search(self, query, profile):
        return list(self._raw_docs)

    async def fetch_full_text(self, doc):
        return self._full_texts[doc.external_id]

    async def check_tabulated(self, values):
        return []


def make_ctx_with_stub() -> tuple[PipelineContext, StubSource]:
    raw = RawDocument(source="arxiv", external_id="2401.00001", title="A Paper", authors="A. Author", year=2023, doi="10.1/x")
    full = FullTextDocument(
        source="arxiv",
        external_id="2401.00001",
        title="A Paper",
        authors="A. Author",
        year=2023,
        doi="10.1/x",
        latex_environments=[
            {"env": "theorem", "content": "F(n) is at least 0.5 for all n.", "location": "section 2"},
            {"env": "conjecture", "content": "F(n) is at least 0.7 for all n.", "location": "section 3"},
        ],
        tables_raw=[{"raw": "{c|c} n & F(n) \\\\ 1 & 1 \\\\ 2 & 2 \\\\ 3 & 3 \\\\", "location": "table 1", "is_appendix": False}],
        body_text="Open problem: is F(n) eventually equal to 1?",
    )
    stub = StubSource([raw], {"2401.00001": full})
    ctx = PipelineContext(
        sources={"arxiv": stub},
        embedder=EmbeddingClient(provider="offline", dim=64),
        vector_store=InMemoryVectorStore(),
        structured_store=InMemoryStructuredStore(),
    )
    return ctx, stub


class TestRunQueryPipeline:
    @pytest.mark.asyncio
    async def test_end_to_end_structure(self):
        ctx, _ = make_ctx_with_stub()
        response = await run_query_pipeline(
            ctx, research_question="What is known about F(n)?", domain_id="math_combinatorics", profile={}
        )
        assert response.papers_indexed == 1
        assert len(response.established_facts) >= 1
        assert any("0.5" in c.verbatim for c in response.established_facts)
        assert len(response.open_gaps) >= 1
        assert len(response.tabulated_values) >= 1
        assert response.novelty_bar.criteria

    @pytest.mark.asyncio
    async def test_persists_to_stores(self):
        ctx, _ = make_ctx_with_stub()
        await run_query_pipeline(ctx, research_question="q", domain_id="math_combinatorics", profile={})
        claims = await ctx.structured_store.get_claims("math_combinatorics")
        assert len(claims) >= 1
        assert await ctx.vector_store.count() >= 1


class TestFastAPIRoutes:
    @pytest.mark.asyncio
    async def test_prior_route_uses_injected_pipeline(self):
        import services.literature.app.main as main_module
        from services.literature.app.models import PriorRequest

        class FakePipeline:
            async def build_prior(self, **kwargs):
                from services.literature.app.models import NoveltyBar, PriorResponse

                return PriorResponse(novelty_bar=NoveltyBar(criteria="x"), sources_consulted=["arxiv"], papers_indexed=0)

        main_module.app.state.pipeline = FakePipeline()
        result = await main_module.prior(PriorRequest(research_question="q", domain_id="d"))
        assert result.sources_consulted == ["arxiv"]

    @pytest.mark.asyncio
    async def test_prior_route_rejects_empty_question(self):
        import services.literature.app.main as main_module
        from fastapi import HTTPException
        from services.literature.app.models import PriorRequest

        with pytest.raises(HTTPException):
            await main_module.prior(PriorRequest(research_question="   ", domain_id="d"))

    @pytest.mark.asyncio
    async def test_health_route(self):
        import services.literature.app.main as main_module

        class FakePipeline:
            async def health(self):
                from services.literature.app.models import HealthResponse

                return HealthResponse(status="ok", citation_verification_rate=None, sources_healthy={"arxiv": True})

        main_module.app.state.pipeline = FakePipeline()
        result = await main_module.health()
        assert result.status == "ok"
