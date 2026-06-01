"""Literature corpus quality gates and evidence coverage (domain-agnostic)."""
from __future__ import annotations

from typing import Any

from propab.config import settings
from propab.embeddings import embed_texts

from services.orchestrator.schemas import Prior

EvidenceStatus = str  # READY | INSUFFICIENT_EVIDENCE | CONFLICTING_EVIDENCE | LOW_COVERAGE


def build_search_intents(question: str, expansion: dict[str, Any] | None) -> list[str]:
    """Merge original question with retrieval expansion outputs."""
    intents: list[str] = [question.strip()]
    if not expansion:
        return _dedupe_intents(intents)

    for key in ("rephrasings", "concepts"):
        for item in expansion.get(key) or []:
            text = str(item).strip()
            if text and text not in intents:
                intents.append(text)
    return _dedupe_intents(intents)


def _dedupe_intents(intents: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in intents:
        norm = " ".join(q.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(q)
    return out[:8]


async def compute_evidence_coverage(question: str, chunk_texts: list[str]) -> float:
    """Max cosine similarity between question embedding and chunk embeddings."""
    if not chunk_texts:
        return 0.0
    api_key = (settings.embed_api_secret or "").strip()
    if not api_key:
        return 0.0
    try:
        texts = [question[:8000]] + [t[:4000] for t in chunk_texts[:40]]
        vecs = await embed_texts(
            texts=texts,
            api_key=api_key,
            model=settings.embed_model,
            provider=settings.embed_provider,
        )
        if len(vecs) < 2:
            return 0.0
        q_vec = vecs[0]
        best = 0.0
        for c_vec in vecs[1:]:
            best = max(best, _cosine(q_vec, c_vec))
        return round(best, 4)
    except Exception:
        return 0.0


def gate_corpus_quality(
    *,
    papers_kept: list[dict[str, Any]],
    chunk_count: int,
    evidence_coverage: float,
) -> tuple[bool, list[str]]:
    """Return (passed, reasons). Fail closed when corpus does not support the question."""
    reasons: list[str] = []

    if len(papers_kept) < settings.literature_min_papers_kept:
        reasons.append(
            f"too_few_papers_kept ({len(papers_kept)} < {settings.literature_min_papers_kept})"
        )
    if chunk_count < settings.literature_min_retrieval_chunks:
        reasons.append(
            f"too_few_chunks ({chunk_count} < {settings.literature_min_retrieval_chunks})"
        )
    if evidence_coverage < settings.literature_min_evidence_coverage:
        reasons.append(
            f"low_evidence_coverage ({evidence_coverage:.3f} < {settings.literature_min_evidence_coverage})"
        )

    return len(reasons) == 0, reasons


def classify_evidence_status(
    prior: Prior,
    *,
    evidence_coverage: float,
    gate_passed: bool,
) -> EvidenceStatus:
    if not gate_passed or _prior_indicates_missing_corpus(prior):
        return "INSUFFICIENT_EVIDENCE"

    if evidence_coverage < settings.literature_min_evidence_coverage:
        return "LOW_COVERAGE"

    contested = prior.contested_claims or []
    established = prior.established_facts or []
    if len(contested) >= 2 and len(established) == 0:
        return "CONFLICTING_EVIDENCE"

    return "READY"


def _prior_indicates_missing_corpus(prior: Prior) -> bool:
    if prior.established_facts:
        return False
    gaps = prior.open_gaps or []
    if not gaps:
        return False
    return all((g.get("gap_type") or "") == "missing_data" for g in gaps)


def insufficient_prior(question: str, diagnostics: dict[str, Any]) -> Prior:
    """Structured prior when retrieval/quality gates fail — no LLM hallucination."""
    return Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[
            {
                "text": (
                    "Retrieved literature does not adequately cover the research question "
                    "after query-scoped fetch and expansion."
                ),
                "source_paper": "bootstrap",
                "gap_type": "missing_data",
            }
        ],
        dead_ends=[],
        key_papers=[],
        evidence_status="INSUFFICIENT_EVIDENCE",
        evidence_coverage=diagnostics.get("evidence_coverage", 0.0),
        retrieval_diagnostics=diagnostics,
    )


def build_retrieval_diagnostics(
    *,
    question: str,
    search_intents: list[str],
    cache_match: str | None,
    papers_fetched: list[dict[str, Any]],
    papers_kept: list[dict[str, Any]],
    papers_dropped: list[dict[str, Any]],
    chunk_count: int,
    evidence_coverage: float,
    expansion_round: int,
    gate_reasons: list[str],
) -> dict[str, Any]:
    return {
        "question": question[:500],
        "search_intents": search_intents,
        "cache_match": cache_match,
        "expansion_round": expansion_round,
        "papers_fetched_count": len(papers_fetched),
        "papers_kept_count": len(papers_kept),
        "papers_dropped_count": len(papers_dropped),
        "papers_kept": [
            {
                "paper_id": p.get("id") or p.get("arxiv_id"),
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "relevance_score": p.get("_relevance_score"),
            }
            for p in papers_kept
        ],
        "papers_dropped": [
            {
                "paper_id": p.get("id") or p.get("arxiv_id"),
                "arxiv_id": p.get("arxiv_id"),
                "title": p.get("title"),
                "relevance_score": p.get("_relevance_score"),
            }
            for p in papers_dropped
        ],
        "chunk_count": chunk_count,
        "evidence_coverage": evidence_coverage,
        "gate_reasons": gate_reasons,
    }


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
