import pytest

from services.literature.app.context import PipelineContext
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.qdrant_store import InMemoryVectorStore
from services.literature.app.indexer.postgres_store import InMemoryStructuredStore
from services.literature.app.models import ExtractedClaim, Finding
from services.literature.app.retriever.novelty_check import check_novelty
from services.literature.app.sources.oeis import OeisSource


async def make_ctx(*, prefill_claims: list[str] | None = None, with_oeis: bool = False) -> PipelineContext:
    embedder = EmbeddingClient(provider="offline", dim=64)
    vector_store = InMemoryVectorStore()
    sources = {}
    if with_oeis:
        sources["oeis"] = OeisSource(min_interval_sec=0.0)
    ctx = PipelineContext(
        sources=sources,
        embedder=embedder,
        vector_store=vector_store,
        structured_store=InMemoryStructuredStore(),
        novelty_similarity_floor=0.70,
        novelty_top_k=30,
        novelty_confidence_verdict_floor=0.85,
    )
    if prefill_claims:
        claims = []
        for i, text in enumerate(prefill_claims):
            emb = await embedder.embed_one(text)
            claims.append(
                ExtractedClaim(
                    text=text, claim_type="theorem", status="proven", verbatim=text,
                    source_title=f"Paper {i}", source_doi=f"10.1/{i}", embedding=emb, claim_id=str(i),
                )
            )
        await vector_store.upsert(claims)
    return ctx


class TestNoveltyCheck:
    @pytest.mark.asyncio
    async def test_empty_index_is_uncertain(self):
        ctx = await make_ctx()
        result = await check_novelty(ctx, Finding(claim="F(n) > 0.9 for all n."), {})
        assert result.verdict == "uncertain"

    @pytest.mark.asyncio
    async def test_sparse_index_no_match_is_uncertain(self):
        ctx = await make_ctx(prefill_claims=["Completely unrelated statement about genes and proteins."])
        result = await check_novelty(ctx, Finding(claim="F(n) is bounded by the square root of n for all large n."), {})
        assert result.verdict == "uncertain"

    @pytest.mark.asyncio
    async def test_identical_claim_is_known(self):
        text = "The maximum Sidon set size in one to n is at least the square root of n minus one."
        ctx = await make_ctx(prefill_claims=[text] * 6)
        result = await check_novelty(ctx, Finding(claim=text), {})
        assert result.verdict == "known"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_no_similar_claim_with_healthy_index_is_novel(self):
        prefill = [f"Unrelated established fact number {i} about protein folding structures." for i in range(6)]
        ctx = await make_ctx(prefill_claims=prefill)
        result = await check_novelty(
            ctx, Finding(claim="A brand new bound on Sidon set density for prime moduli was established."), {}
        )
        assert result.verdict == "novel"
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_tabulation_exact_match_short_circuits_to_known(self):
        ctx = await make_ctx(with_oeis=True)
        ctx.sources["oeis"]._sequence_cache["A003023"] = {
            "name": "Sidon set sizes", "data": "1,2,3,4,5,6,6,7", "offset": "1,1",
        }
        profile = {"tabulation_sources": [{"name": "OEIS", "identifiers": ["A003023"]}]}
        finding = Finding(claim="F(6) = 6", evidence={"index": 6, "value": 6})
        result = await check_novelty(ctx, finding, profile)
        assert result.verdict == "known"
        assert result.confidence == 0.99

    @pytest.mark.asyncio
    async def test_never_returns_novel_below_confidence_floor(self):
        # Even a borderline-similar claim in a healthy index should not
        # produce a "novel" verdict with high confidence unless truly absent.
        prefill = ["F(n) grows roughly like the square root of n for large n." for _ in range(6)]
        ctx = await make_ctx(prefill_claims=prefill)
        result = await check_novelty(ctx, Finding(claim="F(n) grows roughly like the square root of n for large n, restated."), {})
        assert result.verdict in ("known", "uncertain")
        if result.verdict == "novel":
            assert result.confidence >= 0.85
