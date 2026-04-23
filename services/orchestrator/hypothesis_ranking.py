"""
Hypothesis ranking per ARCHITECTURE.md §6.2 (novelty, testability, impact, scope_fit).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from propab.config import settings
from propab.embeddings import embed_texts
from propab.llm import LLMClient
from services.orchestrator.schemas import Prior, RankedHypothesis


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    d = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(d)]


def _fact_texts_for_embedding(prior: Prior) -> list[str]:
    texts: list[str] = []
    for f in prior.established_facts or []:
        t = str(f.get("text", "")).strip()
        if len(t) >= 16:
            texts.append(t[:6000])
    if not texts:
        for kp in (prior.key_papers or [])[:6]:
            t = (str(kp.get("summary", "")) or str(kp.get("title", ""))).strip()
            if len(t) >= 12:
                texts.append(t[:4000])
    return texts


async def compute_novelty_scores(hypothesis_texts: list[str], prior: Prior) -> list[float]:
    """
    Novelty = how far hypothesis embedding is from the centroid of prior fact embeddings
    (1 - cosine_sim), scaled to ~0..1. Higher = more novel vs established facts.
    """
    if not hypothesis_texts:
        return []
    facts = _fact_texts_for_embedding(prior)
    if not facts or not settings.openai_api_key.strip():
        return [0.55] * len(hypothesis_texts)

    texts = facts + [t[:6000] for t in hypothesis_texts]
    try:
        vecs = await embed_texts(texts=texts, api_key=settings.openai_api_key, model=settings.embed_model)
    except Exception:
        return [0.55] * len(hypothesis_texts)

    if len(vecs) != len(texts):
        return [0.55] * len(hypothesis_texts)

    fact_vecs = vecs[: len(facts)]
    hyp_vecs = vecs[len(facts) :]
    c = _centroid(fact_vecs)
    if not c:
        return [0.55] * len(hypothesis_texts)

    out: list[float] = []
    for hv in hyp_vecs:
        cos = _cosine_similarity(hv, c)
        # Far from centroid => more novel; cos=1 aligned => low novelty
        raw = (1.0 - max(-1.0, min(1.0, cos))) / 2.0
        out.append(round(max(0.05, min(0.98, raw + 0.1)), 3))
    return out


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[[\s\S]*\]\s*$", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


async def score_hypothesis_dimensions_llm(
    *,
    llm: LLMClient,
    session_id: str,
    question: str,
    prior: Prior,
    hypotheses: list[RankedHypothesis],
) -> dict[str, dict[str, float]]:
    """LLM scores testability, impact, scope_fit in [0,1] per hypothesis id."""
    if not hypotheses:
        return {}

    brief = [{"id": h.id, "text": h.text[:1200], "test_methodology": (h.test_methodology or "")[:600]} for h in hypotheses]
    prompt = f"""You score research hypotheses for a single autonomous session.

Research question:
{question}

Open gaps (JSON):
{json.dumps(prior.open_gaps[:8], ensure_ascii=False)}

Hypotheses (JSON):
{json.dumps(brief, ensure_ascii=False)}

For EACH hypothesis, output three scores in [0.0, 1.0]:
- testability: can this be probed with code, data analysis, or standard STEM tools in one short session?
- impact: scientific importance if confirmed, relative to the gaps above.
- scope_fit: is the claim scoped so it could realistically be tested once here (not a lifetime program)?

Return JSON array ONLY, same length and order as input, shape:
[{{"id": "<id>", "testability": 0.0, "impact": 0.0, "scope_fit": 0.0}}, ...]
"""

    raw = await llm.call(prompt=prompt, purpose="hypothesis_ranking", session_id=session_id)
    rows = _extract_json_array(raw) or []
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        hid = str(row.get("id", ""))
        if not hid:
            continue

        def _f(key: str) -> float:
            try:
                v = float(row.get(key, 0.5))
            except (TypeError, ValueError):
                v = 0.5
            return max(0.0, min(1.0, v))

        out[hid] = {
            "testability": _f("testability"),
            "impact": _f("impact"),
            "scope_fit": _f("scope_fit"),
        }
    return out


def composite_score(novelty: float, testability: float, impact: float, scope_fit: float) -> float:
    return round(0.30 * novelty + 0.30 * testability + 0.25 * impact + 0.15 * scope_fit, 4)


async def apply_architecture_ranking(
    *,
    hypotheses: list[RankedHypothesis],
    prior: Prior,
    question: str,
    llm: LLMClient,
    session_id: str,
) -> list[RankedHypothesis]:
    texts = [h.text for h in hypotheses]
    novelties = await compute_novelty_scores(texts, prior)
    llm_dims = await score_hypothesis_dimensions_llm(
        llm=llm, session_id=session_id, question=question, prior=prior, hypotheses=hypotheses
    )

    for i, h in enumerate(hypotheses):
        n = novelties[i] if i < len(novelties) else 0.55
        d = llm_dims.get(h.id, {"testability": 0.5, "impact": 0.5, "scope_fit": 0.5})
        t, imp, sc = d["testability"], d["impact"], d["scope_fit"]
        comp = composite_score(n, t, imp, sc)
        h.scores = {
            "novelty": n,
            "testability": round(t, 3),
            "impact": round(imp, 3),
            "scope_fit": round(sc, 3),
            "composite": comp,
        }

    hypotheses.sort(key=lambda x: -float(x.scores.get("composite", 0)))
    for i, h in enumerate(hypotheses, start=1):
        h.rank = i
    return hypotheses
