from unittest.mock import patch

import pytest

from services.literature.app.context import PipelineContext
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.postgres_store import InMemoryStructuredStore
from services.literature.app.indexer.qdrant_store import InMemoryVectorStore
from services.literature.app.models import ExtractedClaim, FullTextDocument, OpenProblem
from services.literature.app.retriever.gap_mapper import _alignment_score, _rank_key, map_gaps
from services.literature.app.retriever.query import (
    _build_novelty_bar,
    _specificity,
    decompose_question,
    dedup_claims,
    process_document,
)


def _make_ctx(**overrides) -> PipelineContext:
    defaults = dict(
        sources={},
        embedder=EmbeddingClient(provider="offline", dim=32),
        vector_store=InMemoryVectorStore(),
        structured_store=InMemoryStructuredStore(),
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestProcessDocumentLlmFallback:
    """process_document's decision to call the LLM claim locator: only
    when regex extraction found nothing AND an API key is configured."""

    @pytest.mark.asyncio
    async def test_calls_locator_when_no_regex_claims_and_key_present(self):
        # Abstract-only prose with no LaTeX environments and no signal
        # phrases claims.py's regex scan looks for.
        doc = FullTextDocument(
            source="pubmed", external_id="1", title="T",
            body_text="The knockout mice showed reduced tumor growth under these specific conditions.",
        )
        ctx = _make_ctx(llm_api_key="test-key", llm_model="test-model")

        async def fake_locate_claims(document, **kwargs):
            return [
                ExtractedClaim(
                    text="x", claim_type="observation", status="proven", verbatim="x", claim_id="1",
                )
            ]

        with patch(
            "services.literature.app.retriever.query.locate_claims", side_effect=fake_locate_claims
        ) as mock_locate:
            result = await process_document(ctx, doc)

        mock_locate.assert_called_once()
        assert len(result["claims"]) == 1

    @pytest.mark.asyncio
    async def test_skips_locator_when_no_api_key(self):
        doc = FullTextDocument(
            source="pubmed", external_id="1", title="T",
            body_text="The knockout mice showed reduced tumor growth under these specific conditions.",
        )
        ctx = _make_ctx(llm_api_key="", llm_model="")

        with patch("services.literature.app.retriever.query.locate_claims") as mock_locate:
            result = await process_document(ctx, doc)

        mock_locate.assert_not_called()
        assert result["claims"] == []

    @pytest.mark.asyncio
    async def test_skips_locator_when_regex_already_found_claims(self):
        doc = FullTextDocument(
            source="arxiv", external_id="1", title="T",
            latex_environments=[{"env": "theorem", "content": "F(n) > 0.5 for all n.", "location": "body"}],
        )
        ctx = _make_ctx(llm_api_key="test-key", llm_model="test-model")

        with patch("services.literature.app.retriever.query.locate_claims") as mock_locate:
            result = await process_document(ctx, doc)

        mock_locate.assert_not_called()
        assert len(result["claims"]) == 1


class TestDecomposeQuestion:
    def test_detects_density_claim_type(self):
        result = decompose_question("What is the density of Sidon sets as n grows?")
        assert result["claim_type"] == "density"

    def test_detects_threshold_claim_type(self):
        result = decompose_question("Is there a threshold at which the phase transition occurs?")
        assert result["claim_type"] == "threshold"

    def test_extracts_scope(self):
        result = decompose_question("For n in [1, 5000], what is the maximum Sidon set size?")
        assert "5000" in result["scope"] or "1" in result["scope"]

    def test_default_structural(self):
        result = decompose_question("Something entirely generic.")
        assert result["claim_type"] == "structural"


def _claim(text, doi="", embedding=None):
    return ExtractedClaim(
        text=text, claim_type="theorem", status="proven", verbatim=text,
        source_doi=doi, embedding=embedding or [], claim_id=text,
    )


class TestDedupClaims:
    @pytest.mark.asyncio
    async def test_near_duplicates_collapse_to_more_specific_one(self):
        embedder = EmbeddingClient(provider="offline", dim=64)
        text = "F(n) is at least the square root of n for all sufficiently large n."
        e1 = await embedder.embed_one(text)
        a = _claim(text, doi="", embedding=e1)
        b = _claim(text, doi="10.1234/real", embedding=e1)
        result = dedup_claims([a, b], threshold=0.95)
        assert len(result) == 1
        assert result[0].source_doi == "10.1234/real"

    def test_specificity_prefers_doi_and_length(self):
        short = _claim("short")
        with_doi = _claim("short but has a doi", doi="10.1/x")
        assert _specificity(with_doi) > _specificity(short)

    @pytest.mark.asyncio
    async def test_unrelated_claims_both_kept(self):
        embedder = EmbeddingClient(provider="offline", dim=64)
        a = _claim("Statement about Sidon sets.", embedding=await embedder.embed_one("Statement about Sidon sets."))
        b = _claim("Statement about gene expression.", embedding=await embedder.embed_one("Statement about gene expression."))
        result = dedup_claims([a, b], threshold=0.95)
        assert len(result) == 2


class TestNoveltyBar:
    def test_uses_profile_criteria_when_present(self):
        bar = _build_novelty_bar({"novelty_criteria": "custom bar text"}, [], [])
        assert bar.criteria == "custom bar text"

    def test_falls_back_to_default_criteria(self):
        bar = _build_novelty_bar({}, [], [])
        assert "not present in any tabulated source" in bar.criteria


class TestGapMapperRanking:
    def test_alignment_score_counts_term_hits(self):
        score = _alignment_score("This is about sidon sets and cap sets", ["sidon", "cap set"])
        assert score > 0

    def test_rank_key_prefers_approachable_and_older(self):
        approachable_old = OpenProblem(statement="a", computationally_approachable=True, year=2000)
        not_approachable_new = OpenProblem(statement="b", computationally_approachable=False, year=2024)
        assert _rank_key(approachable_old, []) > _rank_key(not_approachable_new, [])

    @pytest.mark.asyncio
    async def test_map_gaps_returns_existing_problems_sorted(self):
        store = InMemoryStructuredStore()
        await store.save_open_problems(
            "math_combinatorics",
            [
                OpenProblem(statement="Low priority, not approachable.", computationally_approachable=False, year=2023),
                OpenProblem(statement="High priority, computable and old.", computationally_approachable=True, year=1990),
            ],
        )
        ctx = PipelineContext(
            sources={},
            embedder=EmbeddingClient(provider="offline", dim=32),
            vector_store=InMemoryVectorStore(),
            structured_store=store,
        )
        response = await map_gaps(ctx, domain_id="math_combinatorics", profile={})
        assert len(response.frontier_questions) == 2
        assert response.frontier_questions[0].statement.startswith("High priority")
