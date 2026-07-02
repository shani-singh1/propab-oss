"""LLM extraction of attributed claims, contradictions, and gaps."""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    for marker in ("```json", "```JSON", "```"):
        if marker in text:
            inner = text.split(marker, 1)[1].split("```", 1)[0].strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
    return None


def _openai_chat(prompt: str, *, model: str, api_key: str, base_url: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract structured research priors from paper abstracts. "
                    "Return JSON only. Every claim must cite paper_ids. "
                    "Quote specific findings, not vague summaries."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return str(data["choices"][0]["message"]["content"])


def _fallback_prior(
    question: str,
    papers: list[dict[str, Any]],
    *,
    focus: str,
) -> dict[str, Any]:
    """Abstract-only heuristic prior when no LLM key is available."""
    facts: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for p in papers[:12]:
        abstract = (p.get("abstract") or "").strip()
        if len(abstract) < 80:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", abstract)
        claim = next((s for s in sentences if len(s) > 40), abstract[:220])
        facts.append({
            "text": claim,
            "confidence": 0.45,
            "paper_ids": [p.get("id", "")],
            "attribution": {
                "title": p.get("title"),
                "authors": p.get("authors"),
                "year": p.get("year"),
            },
        })
        if re.search(r"future work|remain(s)? unclear|open question|limited to", abstract, re.I):
            gaps.append({
                "text": f"Paper flags unresolved area: {(abstract[-180:]).strip()}",
                "source_paper": p.get("id", ""),
                "gap_type": "unanswered_question",
            })
    return {
        "established_facts": facts[:20],
        "contested_claims": [],
        "open_gaps": gaps[:10] or [{
            "text": f"LLM extraction unavailable; inspect papers manually for: {focus}",
            "source_paper": "bootstrap",
            "gap_type": "missing_data",
        }],
        "dead_ends": [],
        "key_papers": [
            {
                "paper_id": p.get("id", ""),
                "title": p.get("title", ""),
                "summary": (p.get("abstract") or "")[:280],
                "citation_count": p.get("citation_count"),
                "year": p.get("year"),
            }
            for p in papers
            if p.get("id")
        ],
        "evidence_status": "HEURISTIC_FALLBACK",
        "evidence_coverage": min(1.0, len(facts) / max(1, len(papers))),
        "research_question": question,
        "extraction_method": "abstract_heuristic",
    }


def extract_prior_from_papers(
    *,
    question: str,
    papers: list[dict[str, Any]],
    focus: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    mdl = model or os.environ.get("LLM_MODEL") or "gpt-4o"
    base = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    snippets = [
        {
            "paper_id": p.get("id"),
            "title": p.get("title"),
            "year": p.get("year"),
            "citation_count": p.get("citation_count"),
            "abstract": (p.get("abstract") or "")[:3500],
        }
        for p in papers
    ]

    if not key:
        out = _fallback_prior(question, papers, focus=focus)
        out["retrieval_diagnostics"] = {"paper_count": len(papers), "llm_used": False}
        return out

    prompt = f"""Research question:
{question}

Extraction focus:
{focus}

Paper corpus (JSON):
{json.dumps(snippets, ensure_ascii=False)}

Build a literature prior with THREE sections only:

1. established_facts — specific attributed claims (not summaries). Each entry:
   {{"text": str, "confidence": float 0-1, "paper_ids": [str], "attribution": {{"title": str, "authors": [str], "year": int|null}}}}

2. contested_claims — pairs or groups of papers that disagree. Each entry:
   {{"text": str describing the contradiction, "paper_ids": [str], "sides": [{{"claim": str, "paper_ids": [str]}}]}}

3. open_gaps — questions the literature explicitly flags as open/unresolved. Each entry:
   {{"text": str, "source_paper": str, "gap_type": "unanswered_question"|"missing_data"|"untested_assumption"}}

Also include:
- dead_ends: methods or hypotheses papers report as failed/unpromising
- key_papers: [{{"paper_id": str, "title": str, "summary": str (one specific finding)}}]

Rules:
- Every fact and contradiction MUST cite paper_ids from the corpus.
- Prefer claims about descriptor→dielectric relationships and cross-crystal-system generalization.
- If abstracts lack LOFO/cross-family evidence, put that in open_gaps.
- Return JSON only with keys: established_facts, contested_claims, open_gaps, dead_ends, key_papers
"""

    raw = _openai_chat(prompt, model=mdl, api_key=key, base_url=base)
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        out = _fallback_prior(question, papers, focus=focus)
        out["extraction_error"] = "llm_json_parse_failed"
        out["retrieval_diagnostics"] = {"paper_count": len(papers), "llm_used": True}
        return out

    for key_name in ("established_facts", "contested_claims", "open_gaps", "dead_ends", "key_papers"):
        if not isinstance(parsed.get(key_name), list):
            parsed[key_name] = []

    parsed["evidence_status"] = "READY"
    parsed["evidence_coverage"] = min(1.0, len(parsed["established_facts"]) / max(1, len(papers) * 0.4))
    parsed["research_question"] = question
    parsed["extraction_method"] = "llm_abstract"
    parsed["retrieval_diagnostics"] = {
        "paper_count": len(papers),
        "llm_used": True,
        "model": mdl,
    }
    return parsed
