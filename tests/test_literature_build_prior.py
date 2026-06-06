"""Mocked integration tests for build_prior — no campaign or 30-min run required."""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.literature import build_prior
from services.orchestrator.schemas import Prior

ML_PAPERS = [
    {
        "id": "2601.22095v1",
        "title": "GeoNorm: Geometry-aware normalization",
        "abstract": "We study neural network normalization for language models.",
        "authors": ["A Author"],
        "pdf_url": "https://arxiv.org/pdf/2601.22095v1.pdf",
        "sections_json": None,
    },
    {
        "id": "2602.08064v1",
        "title": "SiameseNorm",
        "abstract": "Batch normalization variants for deep learning.",
        "authors": ["B Author"],
        "pdf_url": "https://arxiv.org/pdf/2602.08064v1.pdf",
        "sections_json": None,
    },
]

NT_PAPERS = [
    {
        "id": "0801.1234v1",
        "title": "On Egyptian fractions and unit fraction decompositions",
        "abstract": "We survey classical results on representing rationals as sums of unit fractions.",
        "authors": ["C Author"],
        "pdf_url": "https://arxiv.org/pdf/0801.1234v1.pdf",
        "sections_json": {"body": "unit fractions modularity density"},
    },
    {
        "id": "0902.5678v1",
        "title": "Density of Egyptian fraction representations",
        "abstract": "Number-theoretic bounds for Egyptian fraction representations of rationals.",
        "authors": ["D Author"],
        "pdf_url": "https://arxiv.org/pdf/0902.5678v1.pdf",
        "sections_json": {"body": "Erdős Straus conjecture related bounds"},
    },
    {
        "id": "1003.9012v1",
        "title": "Additive combinatorics and diophantine equations",
        "abstract": "Methods from additive combinatorics applied to diophantine problems.",
        "authors": ["E Author"],
        "pdf_url": "https://arxiv.org/pdf/1003.9012v1.pdf",
        "sections_json": {"body": "modular obstructions density arguments"},
    },
]

CHUNKS = [
    {"paper_id": p["id"], "chunk_index": i, "text": (p["abstract"] or "") * 2}
    for i, p in enumerate(NT_PAPERS)
]


def _parsed(question: str = "When does every rational admit an Egyptian fraction representation?") -> ParsedQuestion:
    return ParsedQuestion(text=question, domain="math", sub_questions=[question])


@pytest.fixture
def emitter() -> MagicMock:
    em = MagicMock()
    em.emit = AsyncMock()
    return em


@pytest.fixture
def llm() -> MagicMock:
    return MagicMock()


@pytest.mark.asyncio
async def test_cache_poisoning_triggers_refetch_and_ready_prior(emitter, llm) -> None:
    """Simulates wrong cached corpus → gate fail → live fetch → READY prior."""
    fetch_calls = {"n": 0}

    async def fake_fetch(*_a, **_k):
        fetch_calls["n"] += 1
        return list(NT_PAPERS)

    with (
        patch("services.orchestrator.literature.expand_query", new_callable=AsyncMock) as eq,
        patch("services.orchestrator.literature.lookup_cached_paper_ids", new_callable=AsyncMock) as cache,
        patch("services.orchestrator.literature.load_papers_by_ids", new_callable=AsyncMock) as load,
        patch("services.orchestrator.literature._fetch_papers_multi_intent", side_effect=fake_fetch),
        patch("services.orchestrator.literature._upsert_papers", new_callable=AsyncMock),
        patch("services.orchestrator.literature._enrich_papers_with_pdf", new_callable=AsyncMock),
        patch("services.orchestrator.literature._expand_papers_via_citations", new_callable=AsyncMock) as cite,
        patch("services.orchestrator.literature.run_hybrid_retrieval", new_callable=AsyncMock) as retr,
        patch("services.orchestrator.literature.compute_evidence_coverage", new_callable=AsyncMock) as cov,
        patch("services.orchestrator.literature.synthesize_prior_from_papers", new_callable=AsyncMock) as synth,
        patch("services.orchestrator.literature.store_query_cache", new_callable=AsyncMock),
    ):
        eq.return_value = {"rephrasings": ["Egyptian fractions"], "concepts": ["unit fractions"]}
        cache.return_value = (["2601.22095v1", "2602.08064v1"], "exact")
        load.return_value = list(ML_PAPERS)
        cite.side_effect = lambda _sf, papers: papers

        async def fake_filter(question, papers, **kwargs):
            threshold = kwargs.get("threshold", 0.4)
            if papers[0]["id"].startswith("260"):
                return [], [{"id": p["id"], "_relevance_score": 0.1} for p in papers]
            kept = [dict(p, _relevance_score=0.85) for p in papers if not p["id"].startswith("260")]
            return kept, []

        with patch(
            "services.orchestrator.literature._filter_papers_by_question_relevance",
            side_effect=fake_filter,
        ):
            retr.side_effect = lambda **kwargs: [
                c for c in CHUNKS if c["paper_id"] in {str(p.get("id")) for p in kwargs.get("papers", [])}
            ] or CHUNKS
            cov.side_effect = lambda _q, texts: 0.85 if len(texts) >= 3 else 0.1
            synth.return_value = Prior(
                established_facts=[{"text": "Classical density results exist.", "confidence": 0.7, "paper_ids": ["0801.1234v1"]}],
                contested_claims=[],
                open_gaps=[],
                dead_ends=[],
                key_papers=[{"paper_id": "0801.1234v1", "summary": "survey"}],
            )

            prior = await build_prior(
                _parsed(),
                session_id=str(uuid.uuid4()),
                emitter=emitter,
                session_factory=MagicMock(),
                paper_ttl_days=30,
                llm=llm,
            )

    assert fetch_calls["n"] >= 1, "expected live refetch after bad cache"
    assert prior.evidence_status == "READY"
    assert prior.retrieval_diagnostics is not None
    assert prior.retrieval_diagnostics.get("cache_match") is None or fetch_calls["n"] >= 1
    synth.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_failure_returns_insufficient_without_llm(emitter, llm) -> None:
    with (
        patch("services.orchestrator.literature.expand_query", new_callable=AsyncMock) as eq,
        patch("services.orchestrator.literature.lookup_cached_paper_ids", new_callable=AsyncMock, return_value=([], None)),
        patch("services.orchestrator.literature._fetch_papers_multi_intent", new_callable=AsyncMock, return_value=ML_PAPERS[:1]),
        patch("services.orchestrator.literature._fetch_papers_from_db_fallback", new_callable=AsyncMock, return_value=[]),
        patch("services.orchestrator.literature._upsert_papers", new_callable=AsyncMock),
        patch("services.orchestrator.literature._enrich_papers_with_pdf", new_callable=AsyncMock),
        patch("services.orchestrator.literature._expand_papers_via_citations", new_callable=AsyncMock, side_effect=lambda _s, p: p),
        patch(
            "services.orchestrator.literature._filter_papers_by_question_relevance",
            new_callable=AsyncMock,
            return_value=([], ML_PAPERS[:1]),
        ),
        patch("services.orchestrator.literature.run_hybrid_retrieval", new_callable=AsyncMock, return_value=[]),
        patch("services.orchestrator.literature.compute_evidence_coverage", new_callable=AsyncMock, return_value=0.05),
        patch("services.orchestrator.literature.synthesize_prior_from_papers", new_callable=AsyncMock) as synth,
        patch("services.orchestrator.literature.store_query_cache", new_callable=AsyncMock),
    ):
        eq.return_value = {"rephrasings": [], "concepts": []}
        prior = await build_prior(
            _parsed("Unrelated quantum gravity question"),
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            session_factory=MagicMock(),
            paper_ttl_days=30,
            llm=llm,
        )

    assert prior.evidence_status == "INSUFFICIENT_EVIDENCE"
    synth.assert_not_awaited()
    assert prior.retrieval_diagnostics is not None
    assert prior.retrieval_diagnostics.get("gate_reasons")


@pytest.mark.asyncio
async def test_adaptive_threshold_relaxes_when_few_papers(emitter, llm) -> None:
    thresholds_seen: list[float] = []

    async def track_filter(_q, _p, **kwargs):
        thresholds_seen.append(kwargs["threshold"])
        t = kwargs["threshold"]
        if t >= 0.4:
            return [dict(NT_PAPERS[0], _relevance_score=0.35)], []
        return [dict(p, _relevance_score=0.32) for p in NT_PAPERS], []

    with (
        patch("services.orchestrator.literature.expand_query", new_callable=AsyncMock, return_value={"rephrasings": [], "concepts": []}),
        patch("services.orchestrator.literature.lookup_cached_paper_ids", new_callable=AsyncMock, return_value=([], None)),
        patch("services.orchestrator.literature._fetch_papers_multi_intent", new_callable=AsyncMock, return_value=NT_PAPERS),
        patch("services.orchestrator.literature._upsert_papers", new_callable=AsyncMock),
        patch("services.orchestrator.literature._enrich_papers_with_pdf", new_callable=AsyncMock),
        patch("services.orchestrator.literature._expand_papers_via_citations", new_callable=AsyncMock, side_effect=lambda _s, p: p),
        patch("services.orchestrator.literature._filter_papers_by_question_relevance", side_effect=track_filter),
        patch("services.orchestrator.literature.run_hybrid_retrieval", new_callable=AsyncMock, return_value=CHUNKS),
        patch("services.orchestrator.literature.compute_evidence_coverage", new_callable=AsyncMock, return_value=0.9),
        patch("services.orchestrator.literature.synthesize_prior_from_papers", new_callable=AsyncMock) as synth,
        patch("services.orchestrator.literature.store_query_cache", new_callable=AsyncMock),
    ):
        synth.return_value = Prior(
            established_facts=[{"text": "fact", "confidence": 0.8, "paper_ids": ["0801.1234v1"]}],
            contested_claims=[],
            open_gaps=[],
            dead_ends=[],
            key_papers=[],
        )
        prior = await build_prior(
            _parsed(),
            session_id=str(uuid.uuid4()),
            emitter=emitter,
            session_factory=MagicMock(),
            paper_ttl_days=30,
            llm=llm,
        )

    assert any(t < 0.4 for t in thresholds_seen), "expected relaxed threshold retry"
    assert prior.evidence_status == "READY"
