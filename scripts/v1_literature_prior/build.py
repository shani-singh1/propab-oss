"""Orchestrate fetch + extract for the V1 literature prior experiment."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domains import DEFAULT_DOMAIN, DOMAIN_QUERIES
from .extract import extract_prior_from_papers
from .fetch import fetch_domain_papers


def build_literature_prior(
    *,
    domain: str = DEFAULT_DOMAIN,
    question: str | None = None,
    max_papers: int = 30,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    cfg = DOMAIN_QUERIES.get(domain)
    if cfg is None:
        raise ValueError(f"Unknown domain {domain!r}; known: {sorted(DOMAIN_QUERIES)}")

    rq = question or str(cfg.get("research_question") or "")
    focus = str(cfg.get("extraction_focus") or "")

    papers = fetch_domain_papers(cfg, max_papers=max_papers)
    prior = extract_prior_from_papers(
        question=rq,
        papers=papers,
        focus=focus,
        model=model,
        api_key=api_key,
    )

    return {
        "meta": {
            "generator": "scripts/v1_literature_prior",
            "version": "0.1.0",
            "domain": domain,
            "created_at": datetime.now(UTC).isoformat(),
            "paper_count": len(papers),
            "deletable_note": "V1 side experiment — remove scripts/v1_literature_prior/ to drop this layer.",
        },
        "research_question": rq,
        "papers": papers,
        "prior": prior,
    }


def write_prior_artifact(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def prior_for_campaign_loop(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip to the shape Propab campaign_loop expects in prior_json."""
    prior = dict(payload.get("prior") or {})
    prior.setdefault("established_facts", [])
    prior.setdefault("contested_claims", [])
    prior.setdefault("open_gaps", [])
    prior.setdefault("dead_ends", [])
    prior.setdefault("key_papers", [])
    prior["v1_literature_prior_meta"] = payload.get("meta")
    return prior
