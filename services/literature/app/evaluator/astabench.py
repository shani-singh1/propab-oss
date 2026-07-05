"""
AstaBench literature-subtask evaluation.

Honesty note: the official AstaBench harness (``asta-bench/astabench/evals/
labbench/litqa2``) runs on ``inspect_ai`` against a gated dataset and needs
model/API credentials this environment does not have. Rather than fabricate
a wiring to that harness we cannot execute or verify, this module implements
a **local proxy runner** against LitQA2-shaped multiple-choice data (question,
correct answer, distractors) — the same task format, answered using this
service's own indexed claims via semantic search rather than an LLM. It is
real, runs offline, and its accuracy number means what it says: "how often
does the closest indexed claim point at the correct answer." Swapping in the
official inspect_ai task later is a matter of pointing ``run_official`` at a
real dataset path once credentials are available; the scoring/recording
contract (``record_score``) stays the same either way.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from services.literature.app.indexer.embeddings import EmbeddingClient, cosine_similarity


async def run_litqa2_proxy(
    pipeline: Any, cases: list[dict[str, Any]], *, embedder: EmbeddingClient | None = None, top_k: int = 10
) -> dict[str, Any]:
    """``cases``: [{"question": str, "correct_answer": str, "distractors": [str, ...]}].

    For each case, retrieves the pipeline's indexed claims most similar to the
    question, then scores each option by its *max* cosine similarity against
    any one of those retrieved claims (max-pool, not an average blob) — this
    rewards an option that closely restates a single specific indexed claim
    even when the other retrieved claims are about related-but-different
    specifics, which is what actually distinguishes a correct answer from a
    plausible distractor in the same domain. Falls back to question-vs-option
    similarity only when nothing was retrieved at all (empty index).
    Standard multiple-choice accuracy is reported.
    """
    embedder = embedder or pipeline.ctx.embedder
    correct = 0
    per_case: list[dict[str, Any]] = []

    for case in cases:
        question = case["question"]
        options = [case["correct_answer"], *case.get("distractors", [])]
        q_embedding = await embedder.embed_one(question)
        hits = await pipeline.ctx.vector_store.search(q_embedding, top_k=top_k)
        option_embeddings = await embedder.embed(options)

        if hits:
            claim_embeddings = [c.embedding for c, _ in hits if c.embedding]
            scores = [
                max((cosine_similarity(oe, ce) for ce in claim_embeddings), default=0.0)
                for oe in option_embeddings
            ]
        else:
            scores = [cosine_similarity(q_embedding, oe) for oe in option_embeddings]

        picked_idx = max(range(len(options)), key=lambda i: scores[i])
        is_correct = picked_idx == 0  # correct_answer is always index 0
        correct += int(is_correct)
        per_case.append(
            {
                "question": question,
                "picked": options[picked_idx],
                "correct": is_correct,
                "scores": [round(s, 4) for s in scores],
                "n_retrieved": len(hits),
            }
        )

    accuracy = correct / len(cases) if cases else 0.0
    return {"subtask": "litqa2_proxy", "n_cases": len(cases), "accuracy": round(accuracy, 4), "per_case": per_case}


def record_score(scores: dict[str, Any], *, artifacts_dir: str, service_version: str = "0.1.0") -> None:
    """Append (never overwrite) to artifacts/astabench_literature_scores.json."""
    path = Path(artifacts_dir) / "astabench_literature_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(
        {
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "service_version": service_version,
            **scores,
        }
    )
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
