"""Match paper prose sentences to experiment trace text (v1 lexical overlap guard)."""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker


def _tokens(s: str) -> set[str]:
    s = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s.lower())
    return {t for t in s.split() if len(t) >= 4}


def sentence_grounding_score(sentence: str, evidence: str) -> float:
    st = _tokens(sentence)
    et = _tokens(evidence)
    if not st or not et:
        return 0.0
    inter = len(st & et)
    return float(inter) / float(len(st))


def partition_claims(
    sentences: list[str],
    evidence: str,
    *,
    threshold: float = 0.12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grounded: list[dict[str, Any]] = []
    ungrounded: list[dict[str, Any]] = []
    for sent in sentences:
        s = sent.strip()
        if len(s) < 20:
            continue
        score = sentence_grounding_score(s, evidence)
        row = {"sentence": s[:500], "score": round(score, 4)}
        if score >= threshold:
            grounded.append(row)
        else:
            ungrounded.append(row)
    return grounded, ungrounded


def prose_to_sentences(prose: dict[str, str]) -> list[str]:
    parts: list[str] = []
    for key in ("abstract", "introduction", "discussion", "conclusion"):
        blob = prose.get(key) or ""
        parts.extend(re.split(r"(?<=[.!?])\s+", blob))
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) >= 12:
            out.append(p)
    return out[:80]


async def load_session_evidence_text(session_factory: async_sessionmaker, session_id: str) -> str:
    q = text(
        """
        SELECT COALESCE(es.output_json::text, '') || ' ' || COALESCE(es.input_json::text, '')
        FROM experiment_steps es
        JOIN hypotheses h ON h.id = es.hypothesis_id
        WHERE h.session_id = CAST(:sid AS uuid)
        ORDER BY es.created_at ASC
        """
    )
    chunks: list[str] = []
    async with session_factory() as session:
        res = await session.execute(q, {"sid": session_id})
        for row in res.fetchall():
            if row[0]:
                chunks.append(str(row[0]))
    blob = " ".join(chunks)
    return blob[:120_000]


async def ground_session_claims(
    session_factory: async_sessionmaker,
    session_id: str,
    prose: dict[str, str],
    *,
    threshold: float = 0.12,
) -> dict[str, Any]:
    evidence = await load_session_evidence_text(session_factory, session_id)
    sentences = prose_to_sentences(prose)
    grounded, ungrounded = partition_claims(sentences, evidence, threshold=threshold)
    return {
        "evidence_chars": len(evidence),
        "sentences_checked": len(sentences),
        "grounded": grounded[:40],
        "ungrounded": ungrounded[:40],
        "threshold": threshold,
    }
