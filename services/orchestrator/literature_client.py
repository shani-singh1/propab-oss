"""Client for the standalone literature intelligence service (``services/literature/``).

Campaigns build their literature prior by POSTing to ``{literature_service_url}/prior``
instead of running the OLD orchestrator-embedded ``build_prior`` path. This module
owns the request construction, the PriorResponse → orchestrator ``Prior`` mapping, and
the honest fallback to the OLD path when the service is unreachable / erroring.

The literature service is domain-agnostic: everything it needs about the domain
(seed papers, search terms, source priorities, novelty criteria, …) arrives in the
request's ``literature_profile``, which each domain plugin supplies via
``literature_profile()``.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.config import settings
from propab.events import EventEmitter
from propab.llm import LLMClient
from propab.types import EventType
from services.orchestrator.intake import ParsedQuestion
from services.orchestrator.literature import build_prior
from services.orchestrator.schemas import Prior

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# PriorResponse (literature service) → orchestrator Prior mapping
# --------------------------------------------------------------------------


def _claim_to_established_fact(claim: dict[str, Any]) -> dict[str, Any]:
    """ExtractedClaim → established_facts item.

    OLD build_prior shape: {"text": str, "confidence": float, "paper_ids": [str]}.
    A verified citation (has doi/title) is treated as high-confidence; paper_ids
    carry the source identifiers so downstream citation-verification can see them.
    """
    paper_ids = [
        pid
        for pid in (claim.get("source_doi"), claim.get("source_url"))
        if pid
    ]
    return {
        "text": claim.get("text") or claim.get("verbatim") or "",
        "confidence": 0.8 if claim.get("status") == "proven" else 0.6,
        "paper_ids": paper_ids,
        # Extra provenance carried through for citation verification / UI.
        "verbatim": claim.get("verbatim", ""),
        "claim_type": claim.get("claim_type", ""),
        "status": claim.get("status", ""),
        "source": claim.get("source", ""),
        "source_title": claim.get("source_title", ""),
        "source_doi": claim.get("source_doi", ""),
        "source_year": claim.get("source_year", 0),
    }


def _dead_end_from_claim(claim: dict[str, Any]) -> dict[str, Any]:
    """ExtractedClaim → dead_ends item.

    OLD build_prior shape: {"text": str, "paper_ids": [str]}.
    """
    paper_ids = [
        pid
        for pid in (claim.get("source_doi"), claim.get("source_url"))
        if pid
    ]
    return {
        "text": claim.get("text") or claim.get("verbatim") or "",
        "paper_ids": paper_ids,
        "status": claim.get("status", ""),
        "source_title": claim.get("source_title", ""),
    }


def _contradiction_to_contested(contra: dict[str, Any]) -> dict[str, Any]:
    """Contradiction → contested_claims item.

    OLD build_prior shape: {"text": str, "paper_ids": [str]}. A Contradiction has
    two opposing claims; the contested "text" describes the conflict and paper_ids
    reference both sides.
    """
    claim_a = contra.get("claim_a") or {}
    claim_b = contra.get("claim_b") or {}
    a_text = claim_a.get("text") or claim_a.get("verbatim") or ""
    b_text = claim_b.get("text") or claim_b.get("verbatim") or ""
    paper_ids = [
        pid
        for pid in (
            claim_a.get("source_doi"),
            claim_a.get("source_url"),
            claim_b.get("source_doi"),
            claim_b.get("source_url"),
        )
        if pid
    ]
    return {
        "text": f"CONTESTED ({contra.get('contradiction_type', 'direct')}): "
        f"{a_text} <> {b_text}",
        "paper_ids": paper_ids,
        "contradiction_type": contra.get("contradiction_type", "direct"),
        "resolution": contra.get("resolution"),
        "requires_investigation": contra.get("requires_investigation", True),
    }


def _gap_to_open_gap(gap: dict[str, Any]) -> dict[str, Any]:
    """KnowledgeGap → open_gaps item.

    OLD build_prior shape: {"text": str, "source_paper": str, "gap_type": ...}.
    """
    return {
        "text": gap.get("description") or gap.get("what_is_open") or "",
        "source_paper": "literature_service",
        "gap_type": "unanswered_question",
        "what_is_known": gap.get("what_is_known", ""),
        "what_is_open": gap.get("what_is_open", ""),
        "best_known_bound": gap.get("best_known_bound", ""),
        "computationally_approachable": gap.get("computationally_approachable", False),
        "approachable_angle": gap.get("approachable_angle", ""),
    }


def _key_papers_from_sources(
    sources_consulted: list[str],
    established_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """sources_consulted (+ claim provenance) → key_papers items.

    OLD build_prior shape: {"paper_id": str, "summary": str, "title": str}. The
    literature service returns sources as opaque identifier strings; enrich with
    any titles we can recover from the extracted claims' provenance.
    """
    title_by_source: dict[str, str] = {}
    for fact in established_facts:
        src = fact.get("source_doi") or fact.get("source") or ""
        title = fact.get("source_title") or ""
        if src and title and src not in title_by_source:
            title_by_source[src] = title
    return [
        {
            "paper_id": src,
            "summary": "",
            "title": title_by_source.get(src, ""),
        }
        for src in sources_consulted
        if src
    ]


def map_prior_response(payload: dict[str, Any]) -> Prior:
    """Map a literature-service ``PriorResponse`` dict → orchestrator ``Prior``.

    Field mapping:
      established_facts          → established_facts
      contradictions             → contested_claims
      open_gaps                  → open_gaps
      dead_ends                  → dead_ends
      sources_consulted (+facts) → key_papers
      citation_verification_rate → evidence_coverage
      tabulated_values           → retrieval_diagnostics["tabulated_values"]
    """
    established = [_claim_to_established_fact(c) for c in payload.get("established_facts", [])]
    contested = [_contradiction_to_contested(c) for c in payload.get("contradictions", [])]
    open_gaps = [_gap_to_open_gap(g) for g in payload.get("open_gaps", [])]
    dead_ends = [_dead_end_from_claim(c) for c in payload.get("dead_ends", [])]
    key_papers = _key_papers_from_sources(
        payload.get("sources_consulted", []) or [],
        established,
    )

    citation_rate = payload.get("citation_verification_rate")
    evidence_coverage = float(citation_rate) if citation_rate is not None else 0.0

    diagnostics: dict[str, Any] = {
        "source": "literature_service",
        "papers_indexed": payload.get("papers_indexed", 0),
        "sources_consulted": payload.get("sources_consulted", []) or [],
        "citation_verification_rate": citation_rate,
        # Numerical seeds are valuable — carry them through, do not drop.
        "tabulated_values": payload.get("tabulated_values", []) or [],
        "novelty_bar": payload.get("novelty_bar") or {},
    }

    # An empty corpus must not masquerade as READY evidence. Only ACTUAL extracted
    # evidence counts here — NOT ``key_papers``: those are derived purely from
    # ``sources_consulted`` (the names/ids of sources that were *searched*), so a
    # retrieval that consulted sources but extracted nothing would otherwise flip
    # to READY on the strength of source names alone and be fed to generation as
    # real evidence. Tabulated numerical seeds ARE real evidence, so they count.
    has_content = bool(
        established
        or contested
        or open_gaps
        or dead_ends
        or diagnostics["tabulated_values"]
    )
    evidence_status = "READY" if has_content else "INSUFFICIENT_EVIDENCE"

    return Prior(
        established_facts=established,
        contested_claims=contested,
        open_gaps=open_gaps,
        dead_ends=dead_ends,
        key_papers=key_papers,
        evidence_status=evidence_status,
        evidence_coverage=evidence_coverage,
        retrieval_diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------
# Service client
# --------------------------------------------------------------------------


async def _post_prior(request_body: dict[str, Any]) -> dict[str, Any]:
    """POST /prior and return the parsed JSON PriorResponse. Raises on error."""
    base = settings.literature_service_url.rstrip("/")
    timeout = httpx.Timeout(settings.literature_service_timeout_sec)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Best-effort health probe; a failing/unreachable service surfaces here
        # so we fall back rather than hang on /prior.
        health = await client.get(f"{base}/health")
        health.raise_for_status()
        response = await client.post(f"{base}/prior", json=request_body)
        response.raise_for_status()
        return response.json()


async def _post_novelty(request_body: dict[str, Any]) -> dict[str, Any]:
    """POST /novelty and return the parsed JSON NoveltyResponse. Raises on error."""
    base = settings.literature_service_url.rstrip("/")
    timeout = httpx.Timeout(settings.literature_novelty_timeout_sec)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Best-effort health probe; a failing/unreachable service surfaces here so
        # we return the safe default rather than hang on /novelty.
        health = await client.get(f"{base}/health")
        health.raise_for_status()
        response = await client.post(f"{base}/novelty", json=request_body)
        response.raise_for_status()
        return response.json()


# The safe, non-blocking default returned whenever the novelty service is
# unavailable, unconfigured, or errors. It is deliberately NOT "known": an
# outage must never demote a confirmed finding to a rediscovery — it only
# skips the demotion, honestly labelled so the fallback is never silent.
_NOVELTY_UNAVAILABLE: dict[str, Any] = {"verdict": "uncertain", "source": "novelty_unavailable"}


async def check_finding_novelty(
    finding_claim: str,
    evidence: dict[str, Any] | None,
    *,
    domain_plugin: Any | None = None,
    session_id: str,
    emitter: EventEmitter,
) -> dict[str, Any]:
    """Ask the literature service whether a confirmed finding is already KNOWN.

    POSTs ``{finding: {claim, evidence, domain_id}, literature_profile}`` to
    ``{literature_service_url}/novelty`` and returns the parsed ``NoveltyResponse``
    (``{verdict: "known"|"novel"|"uncertain", confidence, explanation,
    matching_sources, recommendation}``).

    This is a DISCOVERY-quality label applied AFTER a verdict is decided — it never
    changes confirm/refute. On any of: an empty ``literature_service_url`` (backward
    compatible), an unreachable/erroring service, or a timeout, it returns the safe
    default ``{"verdict": "uncertain", "source": "novelty_unavailable"}`` and logs —
    it must never raise or block finalize.
    """
    if not (settings.literature_service_url or "").strip():
        # Backward compatible: no service configured → skip the check entirely.
        return dict(_NOVELTY_UNAVAILABLE)

    claim = (finding_claim or "").strip()
    if not claim:
        # An empty claim would be rejected by the service (422); skip honestly.
        return dict(_NOVELTY_UNAVAILABLE)

    if domain_plugin is not None:
        try:
            profile = domain_plugin.literature_profile()
        except Exception:  # noqa: BLE001 — a broken profile must not block finalize
            logger.exception("literature_profile() failed for domain plugin; using empty profile")
            profile = {}
        domain_id = getattr(domain_plugin, "domain_id", "") or ""
    else:
        profile = {}
        domain_id = ""

    request_body = {
        "finding": {
            "claim": claim,
            "evidence": evidence or {},
            "domain_id": domain_id,
        },
        "literature_profile": profile,
    }

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.novelty_check",
        payload={
            "claim": claim[:400],
            "domain_id": domain_id,
            "service_url": settings.literature_service_url,
        },
    )

    try:
        payload = await _post_novelty(request_body)
    except Exception as exc:  # noqa: BLE001 — any failure → safe non-blocking default
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[session %s] literature service /novelty FAILED (%s); "
            "treating finding as novelty-unavailable (no demotion)",
            session_id,
            reason,
        )
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_FETCH_STARTED,
            step="literature.novelty_unavailable",
            payload={
                "claim": claim[:400],
                "novelty_service_error": reason,
                "service_url": settings.literature_service_url,
            },
        )
        return dict(_NOVELTY_UNAVAILABLE)

    verdict = str(payload.get("verdict") or "uncertain")
    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.novelty_checked",
        payload={
            "claim": claim[:400],
            "verdict": verdict,
            "confidence": payload.get("confidence"),
            "matching_sources": payload.get("matching_sources") or [],
        },
    )
    return payload


async def build_prior_via_service(
    parsed: ParsedQuestion,
    *,
    session_id: str,
    emitter: EventEmitter,
    session_factory: async_sessionmaker,
    llm: LLMClient,
    domain_plugin: Any | None = None,
) -> Prior:
    """Build a literature ``Prior`` via the standalone literature service.

    On any transport/HTTP failure, logs LOUDLY and falls back to the OLD embedded
    ``build_prior``, recording ``literature_service_fallback`` in the prior's
    ``retrieval_diagnostics`` so the fallback is never silent.
    """
    question = parsed.text

    if domain_plugin is not None:
        try:
            profile = domain_plugin.literature_profile()
        except Exception:  # noqa: BLE001 — a broken profile must not kill the campaign
            logger.exception("literature_profile() failed for domain plugin; using empty profile")
            profile = {}
        domain_id = getattr(domain_plugin, "domain_id", "") or ""
    else:
        profile = {}
        domain_id = ""

    request_body = {
        "research_question": question,
        "domain_id": domain_id,
        "literature_profile": profile,
        "depth": settings.literature_service_depth,
    }

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_FETCH_STARTED,
        step="literature.fetch",
        payload={
            "query": question,
            "via": "literature_service",
            "domain_id": domain_id,
            "service_url": settings.literature_service_url,
        },
    )

    try:
        payload = await _post_prior(request_body)
    except Exception as exc:  # noqa: BLE001 — any failure → honest fallback to OLD path
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "[session %s] literature service /prior FAILED (%s); "
            "falling back to embedded build_prior",
            session_id,
            reason,
        )
        await emitter.emit(
            session_id=session_id,
            event_type=EventType.LIT_FETCH_STARTED,
            step="literature.service_fallback",
            payload={
                "query": question,
                "literature_service_fallback": reason,
                "service_url": settings.literature_service_url,
            },
        )
        prior = await build_prior(
            parsed,
            session_id=session_id,
            emitter=emitter,
            session_factory=session_factory,
            paper_ttl_days=30,
            llm=llm,
        )
        diagnostics = dict(prior.retrieval_diagnostics or {})
        diagnostics["literature_service_fallback"] = reason
        prior.retrieval_diagnostics = diagnostics
        return prior

    prior = map_prior_response(payload)

    await emitter.emit(
        session_id=session_id,
        event_type=EventType.LIT_PRIOR_BUILT,
        step="literature.prior_build",
        payload={"prior": prior.to_dict(), "via": "literature_service"},
    )
    return prior
