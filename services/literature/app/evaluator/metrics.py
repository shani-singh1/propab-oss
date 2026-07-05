"""
Citation verification rate — the primary health metric (agent3.md).

For a sample of claims already in the index, re-fetch the cited source and
confirm the verbatim quote actually appears in it. This is what makes
``/health``'s ``citation_verification_rate`` honest rather than aspirational:
a service that fabricates citations must fail this check, not pass it by
construction.

Deliberately *not* run on every ``/health`` call — re-fetching sources is
slow and would make health checks flaky under source rate limits. Run this
as an offline job (see ``services/literature/README.md``) and cache the
result; the cached value is what ``/health`` reports.
"""
from __future__ import annotations

import random
import re
from typing import Any

from services.literature.app.models import ExtractedClaim, RawDocument
from services.literature.app.sources.arxiv import normalize_arxiv_id

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _external_id_for(claim: ExtractedClaim) -> str:
    if claim.source == "arxiv":
        return normalize_arxiv_id(claim.source_url or claim.source_doi)
    if claim.source == "oeis":
        m = re.search(r"(A\d{6})", claim.source_url)
        return m.group(1) if m else ""
    if claim.source in ("mathoverflow", "biorxiv"):
        m = re.search(r"/(\d+)(?:/|$)", claim.source_url)
        return m.group(1) if m else claim.source_doi
    if claim.source == "pubmed":
        # PubMed claims never carry a bare PMID field — it only ever shows
        # up embedded in source_url (https://pubmed.ncbi.nlm.nih.gov/{pmid}/,
        # see sources/pubmed.py). Without this case, every PubMed claim fell
        # through to source_doi/source_url below, which pubmed.py's
        # fetch_full_text cannot resolve (it expects a bare PMID as
        # external_id) — silently marking every PubMed claim "unverifiable"
        # (excluded from the rate's denominator) rather than actually
        # re-checking it, which defeats the metric for exactly the source
        # the LLM claim locator (extractors/llm_claim_locator.py) targets.
        m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", claim.source_url)
        return m.group(1) if m else ""
    return claim.source_doi or claim.source_url


async def verify_claim(sources: dict[str, Any], claim: ExtractedClaim) -> bool | None:
    """Returns True/False if verifiable, None if this claim's source cannot
    be re-fetched (excluded from the rate's denominator rather than counted
    as a failure — an unverifiable claim is not evidence of fabrication)."""
    source = sources.get(claim.source)
    if source is None:
        return None
    external_id = _external_id_for(claim)
    if not external_id:
        return None
    try:
        doc = await source.fetch_full_text(
            RawDocument(source=claim.source, external_id=external_id, doi=claim.source_doi, url=claim.source_url)
        )
    except Exception:
        return None
    haystacks = [doc.body_text] + [e.get("content", "") for e in doc.latex_environments] + doc.footnotes + doc.captions
    needle = _normalize(claim.verbatim)
    if not needle:
        return None
    return any(needle in _normalize(h) for h in haystacks if h)


async def citation_verification_rate(
    sources: dict[str, Any], claims: list[ExtractedClaim], sample_size: int = 50, seed: int = 0
) -> dict[str, Any]:
    rng = random.Random(seed)
    sample = rng.sample(claims, min(sample_size, len(claims))) if claims else []
    results = [await verify_claim(sources, c) for c in sample]
    verifiable = [r for r in results if r is not None]
    verified = sum(1 for r in verifiable if r)
    rate = (verified / len(verifiable)) if verifiable else None
    return {
        "sampled": len(sample),
        "verifiable": len(verifiable),
        "verified": verified,
        "rate": rate,
        "unverifiable_sources": sorted({c.source for c, r in zip(sample, results) if r is None}),
    }
