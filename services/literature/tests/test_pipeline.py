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
        # depth="deep" exercises the full-text path (LaTeX environments, body
        # text, tables) this stub is built around. The standard path is
        # abstract-only and is covered separately in TestStandardAbstractPath.
        ctx, _ = make_ctx_with_stub()
        response = await run_query_pipeline(
            ctx, research_question="What is known about F(n)?", domain_id="math_combinatorics",
            profile={}, depth="deep",
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
        await run_query_pipeline(
            ctx, research_question="q", domain_id="math_combinatorics", profile={}, depth="deep"
        )
        claims = await ctx.structured_store.get_claims("math_combinatorics")
        assert len(claims) >= 1
        assert await ctx.vector_store.count() >= 1


class _FetchWouldRaiseSource(BaseSource):
    """A source whose ``search`` returns abstract-bearing hits but whose
    ``fetch_full_text`` blows up if ever called — lets a test assert the
    standard path never touches full text."""

    def __init__(self, name: str, raw_docs: list[RawDocument]) -> None:
        super().__init__()
        self.name = name
        self._raw_docs = raw_docs
        self.fetch_calls = 0

    def is_relevant(self, profile) -> bool:
        return True

    async def search(self, query, profile):
        return list(self._raw_docs)

    async def fetch_full_text(self, doc):
        self.fetch_calls += 1
        raise AssertionError("standard depth must not download full text")

    async def check_tabulated(self, values):
        return []


def _abstract_hit(source: str, ext_id: str) -> RawDocument:
    # An abstract containing a signal phrase the regex ClaimsExtractor keys on
    # ("we show", "we prove") so a proven ``established_fact`` is produced with
    # no LLM and no full text.
    return RawDocument(
        source=source,
        external_id=ext_id,
        title="Bounds for Sidon sets",
        authors="A. Author",
        year=2021,
        doi=f"10.1/{ext_id}",
        url=f"https://example.org/{ext_id}",
        abstract=(
            "We show that every Sidon set in [n] has size at most the square "
            "root of n plus a lower-order term. We prove this bound is tight."
        ),
    )


def _standard_ctx(source: BaseSource) -> PipelineContext:
    return PipelineContext(
        sources={source.name: source},
        embedder=EmbeddingClient(provider="offline", dim=64),
        vector_store=InMemoryVectorStore(),
        structured_store=InMemoryStructuredStore(),
    )


class TestStandardAbstractPath:
    @pytest.mark.asyncio
    async def test_standard_extracts_from_abstract_without_full_text(self):
        src = _FetchWouldRaiseSource("arxiv", [_abstract_hit("arxiv", "2401.00001")])
        ctx = _standard_ctx(src)
        response = await run_query_pipeline(
            ctx, research_question="How large can a Sidon set be?",
            domain_id="math_combinatorics", profile={}, depth="standard",
        )
        assert src.fetch_calls == 0, "standard path must not call fetch_full_text"
        assert response.papers_indexed > 0
        assert len(response.established_facts) >= 1
        assert response.sources_consulted == ["arxiv"]

    @pytest.mark.asyncio
    async def test_standard_partial_results_when_a_source_hangs(self):
        good = _FetchWouldRaiseSource("arxiv", [_abstract_hit("arxiv", "2401.00002")])

        class HangingSource(BaseSource):
            name = "semantic_scholar"

            def is_relevant(self, profile) -> bool:
                return True

            async def search(self, query, profile):
                import asyncio as _a
                await _a.sleep(3600)  # never returns within the deadline
                return []

            async def fetch_full_text(self, doc):
                raise AssertionError("unused")

            async def check_tabulated(self, values):
                return []

        ctx = PipelineContext(
            sources={"arxiv": good, "semantic_scholar": HangingSource()},
            embedder=EmbeddingClient(provider="offline", dim=64),
            vector_store=InMemoryVectorStore(),
            structured_store=InMemoryStructuredStore(),
        )
        # Both sources are relevant, so both are searched concurrently. The
        # hanging source's search never returns; the soft deadline cancels it
        # and the pipeline synthesizes a prior from the fast source alone. The
        # hung source is NOT reported as consulted (honest contract).
        response = await run_query_pipeline(
            ctx, research_question="How large can a Sidon set be?",
            domain_id="math_combinatorics", profile={}, depth="standard",
            deadline_sec=0.5,
        )
        assert response.papers_indexed > 0
        assert len(response.established_facts) >= 1
        assert response.sources_consulted == ["arxiv"]
        assert "semantic_scholar" not in response.sources_consulted

    @pytest.mark.asyncio
    async def test_standard_partial_results_when_a_doc_hangs(self):
        # A doc-processing task that never finishes must not sink the whole
        # prior: the deadline cancels it and we synthesize from the rest.
        from unittest.mock import patch

        fast = _abstract_hit("arxiv", "2401.00003")
        src = _FetchWouldRaiseSource("arxiv", [fast])
        ctx = _standard_ctx(src)

        import services.literature.app.retriever.query as q

        original = q._process_abstract

        # Add a second hit that hangs during processing.
        hang = _abstract_hit("arxiv", "2401.99999")
        src._raw_docs = [fast, hang]

        async def maybe_hang(ctx_, raw_doc):
            if raw_doc.external_id == "2401.99999":
                import asyncio as _a
                await _a.sleep(3600)
            return await original(ctx_, raw_doc)

        with patch.object(q, "_process_abstract", side_effect=maybe_hang):
            response = await run_query_pipeline(
                ctx, research_question="How large can a Sidon set be?",
                domain_id="math_combinatorics", profile={}, depth="standard",
                deadline_sec=2.0,
            )
        assert response.papers_indexed >= 1
        assert len(response.established_facts) >= 1

    @pytest.mark.asyncio
    async def test_deep_still_uses_full_text_path(self):
        # The stub built for the full-text path; deep must still fetch it.
        ctx, stub = make_ctx_with_stub()
        fetched = {"n": 0}
        original_fetch = stub.fetch_full_text

        async def counting_fetch(doc):
            fetched["n"] += 1
            return await original_fetch(doc)

        stub.fetch_full_text = counting_fetch  # type: ignore[assignment]
        response = await run_query_pipeline(
            ctx, research_question="What is known about F(n)?",
            domain_id="math_combinatorics", profile={}, depth="deep",
        )
        assert fetched["n"] == 1, "deep depth must call fetch_full_text"
        assert response.papers_indexed == 1
        assert len(response.established_facts) >= 1


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
