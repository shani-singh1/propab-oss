"""Claim typing for hypothesis outcomes (fixes.md P0.1)."""
from __future__ import annotations

from typing import Any

CLAIM_SYMBOLIC = "CLAIM_SYMBOLIC"
CLAIM_FINITE_VERIFIED = "CLAIM_FINITE_VERIFIED"
CLAIM_STATISTICAL = "CLAIM_STATISTICAL"
CLAIM_PERFORMANCE = "CLAIM_PERFORMANCE"
CLAIM_COUNTEREXAMPLE = "CLAIM_COUNTEREXAMPLE"
CLAIM_CONSTRUCTIVE_FAMILY = "CLAIM_CONSTRUCTIVE_FAMILY"
CLAIM_THEOREM = "CLAIM_THEOREM"

ALL_CLAIM_TYPES = (
    CLAIM_SYMBOLIC,
    CLAIM_FINITE_VERIFIED,
    CLAIM_STATISTICAL,
    CLAIM_PERFORMANCE,
    CLAIM_COUNTEREXAMPLE,
    CLAIM_CONSTRUCTIVE_FAMILY,
    CLAIM_THEOREM,
)

_PERFORMANCE_HINTS = (
    "speed",
    "latency",
    "throughput",
    "memory",
    "cache",
    "simd",
    "heap",
    "benchmark",
    "runtime",
    "wall clock",
)
_CONSTRUCTIVE_HINTS = ("parametric", "family", "identity", "closed-form", "closed form")
_FINITE_HINTS = ("exhaust", "scan", "up to", "for all n", "for n in", "range", "≤", "<=")
_THEOREM_HINTS = ("proof", "theorem", "identically", "q.e.d", "qed")


def classify_claim_type(
    evidence: dict[str, Any],
    verdict: str,
    *,
    hypothesis_text: str = "",
) -> str | None:
    """
    Classify the kind of claim supported or refuted by evidence.

    Returns None for inconclusive or unclassified outcomes.
    """
    text_l = f"{hypothesis_text} {evidence.get('verdict_reason') or ''}".lower()

    if verdict == "refuted":
        if int(evidence.get("verified_false_steps") or 0) > 0:
            return CLAIM_COUNTEREXAMPLE
        return None

    if verdict != "confirmed":
        return None

    vt = int(evidence.get("verified_true_steps") or 0)
    if vt > 0:
        if any(k in text_l for k in _CONSTRUCTIVE_HINTS):
            return CLAIM_CONSTRUCTIVE_FAMILY
        if any(k in text_l for k in _FINITE_HINTS):
            return CLAIM_FINITE_VERIFIED
        if any(k in text_l for k in _THEOREM_HINTS):
            return CLAIM_THEOREM
        return CLAIM_SYMBOLIC

    if int(evidence.get("n_metric_steps") or 0) > 0:
        if any(k in text_l for k in _PERFORMANCE_HINTS):
            return CLAIM_PERFORMANCE
        return CLAIM_STATISTICAL

    return None


def build_finding_object(
    *,
    claim: str,
    claim_type: str | None,
    evidence: dict[str, Any],
    confidence: float,
    verification_method: str | None,
    theme: str | None,
    mechanism: str | None,
) -> dict[str, Any]:
    """Structured finding (fixes.md P2.2)."""
    return {
        "claim": claim,
        "claim_type": claim_type,
        "evidence": str(evidence.get("verdict_reason") or evidence.get("summary") or ""),
        "confidence": round(float(confidence), 4),
        "verification_method": verification_method,
        "theme": theme,
        "mechanism": mechanism,
    }


def extract_mechanism(
    evidence: dict[str, Any],
    *,
    claim_type: str | None,
    hypothesis_text: str = "",
) -> str | None:
    """Heuristic mechanism extraction from confirmed evidence (fixes.md P2.1)."""
    if not claim_type:
        return None
    reason = str(evidence.get("verdict_reason") or "").strip()
    vt = int(evidence.get("verified_true_steps") or 0)
    if vt > 0:
        base = f"Deterministic verification ({vt} independent checks)"
        return f"{base}: {reason}" if reason else base
    p = evidence.get("p_value")
    if p is not None:
        base = f"Statistical support (p={float(p):.4g})"
        return f"{base}: {reason}" if reason else base
    if reason:
        return reason[:600]
    snippet = hypothesis_text.strip()[:200]
    return snippet or None
