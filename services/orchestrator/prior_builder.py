from __future__ import annotations

import json
import re
from typing import Any

from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.schemas import Prior


def _prior_fallback(question: str, papers: list[dict[str, Any]]) -> Prior:
    return Prior(
        established_facts=[],
        contested_claims=[],
        open_gaps=[
            {
                "text": f"Structured prior unavailable; inspect key papers for: {question}",
                "source_paper": "bootstrap",
                "gap_type": "missing_data",
            }
        ],
        dead_ends=[],
        key_papers=[
            {
                "paper_id": p.get("id", ""),
                "summary": (p.get("abstract") or "")[:220],
                "title": p.get("title", ""),
            }
            for p in papers
            if p.get("id")
        ],
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}\s*$", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalize_prior(data: dict[str, Any], papers: list[dict[str, Any]]) -> Prior:
    def _list(key: str) -> list:
        v = data.get(key)
        return v if isinstance(v, list) else []

    key_papers = _list("key_papers")
    if not key_papers and papers:
        key_papers = [
            {"paper_id": p.get("id", ""), "summary": (p.get("abstract") or "")[:220], "title": p.get("title", "")}
            for p in papers
            if p.get("id")
        ]
    return Prior(
        established_facts=_list("established_facts"),
        contested_claims=_list("contested_claims"),
        open_gaps=_list("open_gaps") or [
            {
                "text": "No explicit gaps returned by prior builder.",
                "source_paper": "bootstrap",
                "gap_type": "unanswered_question",
            }
        ],
        dead_ends=_list("dead_ends"),
        key_papers=key_papers,
    )


async def synthesize_prior_from_papers(
    *,
    llm: LLMClient,
    session_id: str,
    question: str,
    papers: list[dict[str, Any]],
    emitter: EventEmitter,
    retrieval_chunks: list[dict[str, Any]] | None = None,
) -> Prior:
    """
    LLM prior builder aligned to ARCHITECTURE §5.4 (structured Prior JSON).
    Emits llm.parse_error on malformed JSON, then returns a safe fallback.
    """
    snippets = []
    for p in papers[:8]:
        body = ""
        sec = p.get("sections_json")
        if isinstance(sec, dict):
            body = (sec.get("body") or sec.get("text") or "")[:6000]
        snippets.append(
            {
                "paper_id": p.get("id"),
                "title": p.get("title"),
                "abstract": (p.get("abstract") or "")[:4000],
                "body_excerpt": body[:4000] if body else None,
            }
        )

    retrieval = retrieval_chunks or []
    retrieval_preview = [
        {"paper_id": c.get("paper_id"), "chunk_index": c.get("chunk_index"), "text": (c.get("text") or "")[:2500]}
        for c in retrieval[:20]
    ]

    prompt = f"""You are a research prior builder.

Research question:
{question}

Top retrieval chunks (hybrid BM25 + dense, RRF) (JSON):
{json.dumps(retrieval_preview, ensure_ascii=False)}

Paper snippets (JSON):
{json.dumps(snippets, ensure_ascii=False)}

Return JSON only with this exact top-level shape:
{{
  "established_facts": [{{"text": str, "confidence": float, "paper_ids": [str]}}],
  "contested_claims": [{{"text": str, "paper_ids": [str]}}],
  "open_gaps": [{{"text": str, "source_paper": str, "gap_type": "unanswered_question"|"missing_data"|"untested_assumption"}}],
  "dead_ends": [{{"text": str, "paper_ids": [str]}}],
  "key_papers": [{{"paper_id": str, "summary": str}}]
}}
Use empty arrays where unknown. Every claim must cite paper_ids from snippets when possible.
"""

    raw = await llm.call(
        prompt=prompt,
        purpose="prior_build",
        session_id=session_id,
        hypothesis_id=None,
    )
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LLM_PARSE_ERROR,
            step="literature.prior_build",
            payload={"purpose": "prior_build", "preview": raw[:500]},
        )
        return _prior_fallback(question, papers)

    return _normalize_prior(parsed, papers)
