from __future__ import annotations

import math
from typing import Any

from propab.config import settings
from propab.embeddings import embed_texts
from propab.events import EventEmitter
from propab.types import EventType
from services.orchestrator.schemas import Prior


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


async def evaluate_literature_short_circuit(
    *,
    question: str,
    prior: Prior,
    session_id: str,
    emitter: EventEmitter,
) -> dict[str, Any] | None:
    """
    If an established fact embedding is very close to the question embedding,
    short-circuit hypothesis generation (ARCHITECTURE §5.4).
    """
    facts = prior.established_facts or []
    if not facts or not settings.openai_api_key.strip():
        return None

    texts: list[str] = []
    fact_indices: list[int] = []
    for i, fact in enumerate(facts):
        t = str(fact.get("text", "")).strip()
        if len(t) < 12:
            continue
        texts.append(t[:8000])
        fact_indices.append(i)

    if not texts:
        return None

    try:
        vectors = await embed_texts(
            texts=[question[:8000], *texts],
            api_key=settings.openai_api_key,
            model=settings.embed_model,
        )
    except Exception:
        return None

    if len(vectors) != len(texts) + 1:
        return None

    qv = vectors[0]
    threshold = float(settings.literature_answer_similarity)
    best_sim = -1.0
    best_local = 0
    for j, tv in enumerate(vectors[1:], start=0):
        sim = _cosine_similarity(qv, tv)
        if sim > best_sim:
            best_sim = sim
            best_local = j

    if best_sim < threshold:
        return None

    fact = facts[fact_indices[best_local]]
    paper_ids = fact.get("paper_ids") if isinstance(fact.get("paper_ids"), list) else []
    paper_ids_s = [str(x) for x in paper_ids if x]

    payload = {
        "similarity": round(best_sim, 4),
        "threshold": threshold,
        "established_fact": str(fact.get("text", ""))[:2000],
        "supporting_paper_ids": paper_ids_s,
    }
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_ANSWER_FOUND,
        step="literature.answer_gate",
        payload=payload,
    )
    return payload
