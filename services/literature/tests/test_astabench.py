import json

import pytest

from services.literature.app.evaluator.astabench import record_score, run_litqa2_proxy
from services.literature.app.indexer.embeddings import EmbeddingClient
from services.literature.app.indexer.qdrant_store import InMemoryVectorStore
from services.literature.app.models import ExtractedClaim


class _FakeCtx:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store


class _FakePipeline:
    def __init__(self, embedder, vector_store):
        self.ctx = _FakeCtx(embedder, vector_store)


async def _make_pipeline_with_claims(claims_text: list[str]) -> _FakePipeline:
    embedder = EmbeddingClient(provider="offline", dim=128)
    store = InMemoryVectorStore()
    claims = []
    for i, text in enumerate(claims_text):
        emb = await embedder.embed_one(text)
        claims.append(
            ExtractedClaim(
                text=text, claim_type="theorem", status="proven", verbatim=text,
                embedding=emb, claim_id=str(i),
            )
        )
    await store.upsert(claims)
    return _FakePipeline(embedder, store)


class TestLitQA2Proxy:
    @pytest.mark.asyncio
    async def test_picks_correct_answer_when_it_matches_an_indexed_claim(self):
        pipeline = await _make_pipeline_with_claims(
            [
                "The maximum Sidon set size F(n) is at least sqrt(n) times one minus a vanishing term.",
                "Gene expression varies substantially across human tissues in GTEx.",
            ]
        )
        cases = [
            {
                "question": "What is the growth rate of the maximum Sidon set size F(n)?",
                "correct_answer": "F(n) is at least sqrt(n) times one minus a vanishing term.",
                "distractors": ["F(n) grows linearly in n.", "F(n) is bounded by a constant."],
            }
        ]
        result = await run_litqa2_proxy(pipeline, cases)
        assert result["accuracy"] == 1.0
        assert result["per_case"][0]["correct"] is True
        assert result["per_case"][0]["n_retrieved"] >= 1

    @pytest.mark.asyncio
    async def test_falls_back_to_question_similarity_when_index_empty(self):
        pipeline = await _make_pipeline_with_claims([])
        cases = [
            {
                "question": "What is the growth rate of the maximum Sidon set size F(n)?",
                "correct_answer": "F(n) is at least sqrt(n) for all n.",
                "distractors": ["Completely unrelated statement about gene expression in tissues."],
            }
        ]
        result = await run_litqa2_proxy(pipeline, cases)
        assert result["per_case"][0]["n_retrieved"] == 0
        assert 0.0 <= result["accuracy"] <= 1.0

    @pytest.mark.asyncio
    async def test_accuracy_is_fraction_correct_across_cases(self):
        pipeline = await _make_pipeline_with_claims(
            ["F(n) is at least sqrt(n) for all n.", "Cap sets in F_3^n have size at most 2.756^n."]
        )
        cases = [
            {
                "question": "Sidon set growth?",
                "correct_answer": "F(n) is at least sqrt(n) for all n.",
                "distractors": ["Totally unrelated nonsense about tissue expression."],
            },
            {
                "question": "Cap set upper bound?",
                "correct_answer": "Cap sets in F_3^n have size at most 2.756^n.",
                "distractors": ["Totally unrelated nonsense about enzyme kinetics."],
            },
        ]
        result = await run_litqa2_proxy(pipeline, cases)
        assert result["n_cases"] == 2
        assert result["accuracy"] == 1.0


def test_record_score_appends_not_overwrites(tmp_path):
    record_score({"subtask": "litqa2_proxy", "accuracy": 0.5}, artifacts_dir=str(tmp_path))
    record_score({"subtask": "litqa2_proxy", "accuracy": 0.8}, artifacts_dir=str(tmp_path))
    path = tmp_path / "astabench_literature_scores.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["accuracy"] == 0.5
    assert data[1]["accuracy"] == 0.8
