"""
Novelty checking — the six-step algorithm from agent3.md, implemented
conservatively: a false "novel" verdict costs a whole campaign (Propab
chases a result the field already has), while a false "known" verdict is
recoverable (a domain expert corrects it). Every branch below is biased
toward "uncertain" whenever the evidence is thin.
"""
from __future__ import annotations

import re
from typing import Any

from services.literature.app.context import PipelineContext
from services.literature.app.extractors._bounds import intervals_disjoint, parse_bounds
from services.literature.app.models import ExtractedClaim, Finding, NoveltyResponse

_SCOPE_TOKEN_RE = re.compile(r"\b(?:n|q)\s*(?:∈|in|up to|<=|≤|=)\s*\[?(\d+(?:,\s*\d+)?|\d+)\]?", re.I)
_CONSTRUCTION_WORDS = ("greedy", "optimal", "random", "explicit", "probabilistic", "algebraic")
_MIN_INDEX_SIZE_FOR_NOVEL = 5


def _scope_tokens(text: str) -> set[str]:
    scopes = {m.group(1) for m in _SCOPE_TOKEN_RE.finditer(text)}
    constructions = {w for w in _CONSTRUCTION_WORDS if w in text.lower()}
    return scopes | constructions


def _scope_compatible(finding_text: str, claim_text: str) -> bool:
    f_scope = _scope_tokens(finding_text)
    c_scope = _scope_tokens(claim_text)
    if not f_scope or not c_scope:
        return True  # no scope markers detected on either side — can't rule out compatibility
    return bool(f_scope & c_scope) or f_scope.issubset(c_scope) or c_scope.issubset(f_scope)


def _implied_by(finding_text: str, claim_text: str) -> bool:
    """Does ``claim_text`` (indexed, established) logically imply
    ``finding_text`` (the candidate)? A claim that 'F(n) > 0.7' implies
    'F(n) > 0.6' — the weaker bound is not novel if the stronger one holds."""
    finding_bounds = parse_bounds(finding_text)
    claim_bounds = parse_bounds(claim_text)
    for f_subj, f_lo, f_hi in finding_bounds:
        for c_subj, c_lo, c_hi in claim_bounds:
            if f_subj != c_subj:
                continue
            # claim implies finding iff claim's interval is a subset of finding's
            # interval (claim is at least as strong / specific).
            if c_lo >= f_lo and c_hi <= f_hi and not intervals_disjoint((f_lo, f_hi), (c_lo, c_hi)):
                return True
    return False


async def check_novelty(ctx: PipelineContext, finding: Finding, profile: dict[str, Any]) -> NoveltyResponse:
    claim_text = finding.claim

    # Step 4 (tabulation check) is run first — an exact tabulated match is the
    # highest-confidence signal available and short-circuits everything else.
    tab_match = await _tabulation_check(ctx, finding, profile)
    if tab_match is not None:
        return NoveltyResponse(
            verdict="known",
            confidence=0.99,
            explanation=f"Exact value match in tabulated source {tab_match['identifier']}.",
            matching_sources=[tab_match],
            recommendation=f"Do not claim as novel — already tabulated in {tab_match['identifier']}.",
        )

    total_indexed = await ctx.vector_store.count()
    if total_indexed == 0:
        return NoveltyResponse(
            verdict="uncertain",
            confidence=0.3,
            explanation="No indexed literature available for this domain yet — cannot assess novelty.",
            matching_sources=[],
            recommendation="Run /prior for this domain first to build an index before trusting novelty verdicts.",
        )

    # Step 1-2: decompose + embed + semantic search.
    embedding = await ctx.embedder.embed_one(claim_text)
    hits = await ctx.vector_store.search(embedding, top_k=ctx.novelty_top_k)
    relevant = [(c, score) for c, score in hits if score >= ctx.novelty_similarity_floor]

    if not relevant:
        if total_indexed < _MIN_INDEX_SIZE_FOR_NOVEL:
            return NoveltyResponse(
                verdict="uncertain",
                confidence=0.5,
                explanation=(
                    f"No similar indexed claims found, but the index only has {total_indexed} "
                    "claims — too sparse to confidently declare novelty."
                ),
                matching_sources=[],
                recommendation="Expand literature coverage for this domain before trusting this verdict.",
            )
        return NoveltyResponse(
            verdict="novel",
            confidence=0.85,
            explanation="No similar claim found in indexed literature, and the index has meaningful coverage.",
            matching_sources=[],
            recommendation="Appears to extend beyond indexed literature — proceed, but keep the novelty_check attached to the evidence.",
        )

    # Step 3 + 5: scope matching + implication check against best match.
    best_claim, best_score = relevant[0]
    scope_ok = _scope_compatible(claim_text, best_claim.verbatim)
    implied = _implied_by(claim_text, best_claim.verbatim)

    matching_sources = [_to_source_dict(c, s) for c, s in relevant[:5]]

    if best_score >= ctx.novelty_confidence_verdict_floor and scope_ok and (implied or best_score >= 0.95):
        return NoveltyResponse(
            verdict="known",
            confidence=round(best_score, 3),
            explanation=(
                f"Closely matches an indexed claim from {best_claim.source_title or best_claim.source_doi}: "
                f'"{best_claim.verbatim[:200]}"'
            ),
            matching_sources=matching_sources,
            recommendation="Do not claim as novel — appears established. Verify by reading the cited source.",
        )

    if best_score >= ctx.novelty_confidence_verdict_floor and not scope_ok:
        return NoveltyResponse(
            verdict="uncertain",
            confidence=round(min(best_score, 0.7), 3),
            explanation=(
                "A highly similar claim exists but covers a different scope/construction — "
                "manual comparison needed before ruling on novelty."
            ),
            matching_sources=matching_sources,
            recommendation="Manually compare scope against the listed sources before claiming novelty.",
        )

    return NoveltyResponse(
        verdict="uncertain",
        confidence=round(best_score, 3),
        explanation="Similar but not identical claims found in the literature; not confident enough to call known or novel.",
        matching_sources=matching_sources,
        recommendation="Manually verify against the listed sources before claiming novelty.",
    )


def _to_source_dict(claim: ExtractedClaim, score: float) -> dict[str, Any]:
    return {
        "title": claim.source_title,
        "doi": claim.source_doi,
        "url": claim.source_url,
        "verbatim": claim.verbatim,
        "similarity": round(score, 4),
        "location": claim.location,
    }


async def _tabulation_check(ctx: PipelineContext, finding: Finding, profile: dict[str, Any]) -> dict[str, Any] | None:
    oeis = ctx.sources.get("oeis")
    if oeis is None:
        return None
    values = dict(finding.evidence or {})
    tab_sources = profile.get("tabulation_sources", []) or []
    oeis_ids = [
        ident
        for tab in tab_sources
        if tab.get("name", "").lower() == "oeis"
        for ident in tab.get("identifiers", []) or []
    ]
    if not oeis_ids or ("index" not in values and "value" not in values):
        return None
    values["candidate_sequence_ids"] = oeis_ids
    matches = await oeis.check_tabulated(values)
    for m in matches:
        if m.matched:
            return {"identifier": m.identifier, "url": m.url, "matched_index": m.matched_index, "matched_value": m.matched_value}
    return None
