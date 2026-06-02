"""
Domain-agnostic research quality controls (fixes.md post network-resilience).

P0: control nodes, evidence reuse
P1: replication tiers, claim strength
P2: theme vectors, mechanism objects
P3: inconclusive diagnosis
P4: canonical finding ledger
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

NODE_ROLE_DISCOVERY = "DISCOVERY"
NODE_ROLE_CONTROL = "CONTROL"

REPLICATION_T1 = "T1"
REPLICATION_T2 = "T2"
REPLICATION_T3 = "T3"

# Claim strength labels (P1.2) — map from legacy CLAIM_* where needed
CLAIM_STATISTICAL = "STATISTICAL"
CLAIM_FINITE_VERIFIED = "FINITE_VERIFIED"
CLAIM_COUNTEREXAMPLE = "COUNTEREXAMPLE"
CLAIM_CONSTRUCTIVE = "CONSTRUCTIVE"
CLAIM_MECHANISTIC = "MECHANISTIC"
CLAIM_SYMBOLIC = "SYMBOLIC"
CLAIM_PERFORMANCE = "PERFORMANCE"

_LEGACY_TO_STRENGTH = {
    "CLAIM_STATISTICAL": CLAIM_STATISTICAL,
    "CLAIM_FINITE_VERIFIED": CLAIM_FINITE_VERIFIED,
    "CLAIM_COUNTEREXAMPLE": CLAIM_COUNTEREXAMPLE,
    "CLAIM_CONSTRUCTIVE_FAMILY": CLAIM_CONSTRUCTIVE,
    "CLAIM_THEOREM": CLAIM_SYMBOLIC,
    "CLAIM_SYMBOLIC": CLAIM_SYMBOLIC,
    "CLAIM_PERFORMANCE": CLAIM_PERFORMANCE,
}

_CONTROL_MARKERS = (
    "null hypothesis",
    "no falsifiable pattern",
    "no statistically significant effect beyond noise",
    "no effect beyond random",
)

_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("spectral", ("spectral gap", "eigenvalue", "laplacian", "adjacency matrix", "algebraic connectivity")),
    ("assortativity", ("assortativity", "degree correlation", "rich-club")),
    ("clustering", ("clustering coefficient", "transitivity", "local clustering")),
    ("centrality", ("betweenness", "centrality", "eigenvector centrality", "degree-based removal")),
    ("degree_structure", ("gini", "degree distribution", "degree variance", "average degree", "k-core")),
    ("scale_free", ("barab", "scale-free", "scale free", "preferential attachment")),
    ("small_world", ("watts-strogatz", "watts strogatz", "small-world", "small world", "rewiring probability")),
    ("random_graph", ("erdős", "erdos", "erdős-rényi", "g(n,p)", "random graph")),
    ("percolation", ("percolation", "giant component", "critical threshold", "pc")),
    ("targeted_removal", ("targeted removal", "targeted attack", "node removal")),
    ("sparse_regime", ("sparse graph", "average degree <", "fragmentation")),
    ("residue_class", (" mod ", " modulo ", "residue", "congruen")),
    ("parametric_family", ("parametric", "family", "closed-form")),
    ("finite_verification", ("exhaust", "scan", "up to n", "for all n", "counterexample")),
    ("unit_fraction", ("unit fraction", "egyptian", "1/n")),
]


def normalize_claim_strength(claim_type: str | None) -> str | None:
    if not claim_type:
        return None
    return _LEGACY_TO_STRENGTH.get(claim_type, claim_type)


def is_control_hypothesis(text: str) -> bool:
    """P0.1 — null / calibration hypotheses are CONTROL, not discoveries."""
    core = re.split(r"\s*\(Question:", (text or "").strip(), maxsplit=1)[0].lower()
    return any(m in core for m in _CONTROL_MARKERS)


def infer_node_role(text: str) -> str:
    return NODE_ROLE_CONTROL if is_control_hypothesis(text) else NODE_ROLE_DISCOVERY


def extract_theme_vector(text: str) -> tuple[str, list[str]]:
    """P2.1 — primary + secondary themes from hypothesis/finding text."""
    t = (text or "").lower()
    matched: list[str] = []
    for theme_id, keywords in _THEME_RULES:
        if any(k in t for k in keywords):
            matched.append(theme_id)
    if not matched:
        return "general", []
    return matched[0], matched[1:]


def compute_evidence_hash(evidence: dict[str, Any]) -> str:
    """P0.2 — stable hash of verification payload."""
    payload = {
        k: evidence.get(k)
        for k in (
            "verified_true_steps",
            "verified_false_steps",
            "n_metric_steps",
            "p_value",
            "effect_size",
            "metric_value",
            "verdict_reason",
        )
        if evidence.get(k) is not None
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_verification_hash(evidence: dict[str, Any]) -> str:
    """Hash of deterministic verification only."""
    payload = {
        "vt": evidence.get("verified_true_steps"),
        "vf": evidence.get("verified_false_steps"),
        "reason": evidence.get("verdict_reason"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def compute_replication_level(
    evidence: dict[str, Any],
    *,
    hypothesis_text: str = "",
    sibling_confirmed: int = 0,
) -> str:
    """P1.1 — T1 single, T2 replicated, T3 cross-context."""
    vt = int(evidence.get("verified_true_steps") or 0)
    vf = int(evidence.get("verified_false_steps") or 0)
    nm = int(evidence.get("n_metric_steps") or 0)
    text_l = (hypothesis_text or "").lower()

    cross_ctx_markers = (
        "barab", "watts", "erdős", "erdos", "configuration model",
        "multiple families", "across synthetic", "mnist", "cifar",
        "dataset a", "dataset b", "family a", "family b",
    )
    n_families = sum(1 for m in cross_ctx_markers if m in text_l)
    _, secondary = extract_theme_vector(hypothesis_text)
    if n_families >= 2 or len(secondary) >= 2 or sibling_confirmed >= 3:
        return REPLICATION_T3
    if vt >= 2 or vf >= 2 or nm >= 2 or sibling_confirmed >= 2:
        return REPLICATION_T2
    if vt >= 1 or vf >= 1 or nm >= 1:
        return REPLICATION_T1
    return REPLICATION_T1


def classify_inconclusive_reason(
    evidence: dict[str, Any],
    *,
    failure_reason: str | None = None,
    verdict_reason: str | None = None,
) -> str:
    """P3.1 — why a node is inconclusive."""
    fr = (failure_reason or "").lower()
    vr = (verdict_reason or evidence.get("verdict_reason") or "").lower()
    blob = f"{fr} {vr}"

    if "duplicate_evidence" in blob or "reused_evidence" in blob:
        return "duplicate_evidence"
    if "control_calibration" in blob or "control node" in blob:
        return "control_calibration"
    if "timeout" in blob or "wall" in blob or "exceeded" in blob:
        return "timeout"
    if "sub-agent failed" in blob or "revoke" in blob:
        return "budget_limit"
    if "counterexample" in blob or "verified_false" in blob:
        return "conflicting_evidence"
    if int(evidence.get("verified_true_steps") or 0) >= 1 and "unreplicated" in blob:
        return "insufficient_samples"
    if "no metric" in blob or "no significance" in blob or "ambiguous" in blob:
        return "verification_failure"
    if "insufficient" in blob or "unreplicated" in blob:
        return "insufficient_samples"
    return "verification_failure"


def classify_claim_strength(
    evidence: dict[str, Any],
    verdict: str,
    *,
    hypothesis_text: str = "",
) -> str | None:
    """P1.2 claim hierarchy from evidence + text."""
    from propab.claim_types import classify_claim_type

    legacy = classify_claim_type(evidence, verdict, hypothesis_text=hypothesis_text)
    strength = normalize_claim_strength(legacy)
    if strength:
        return strength
    text_l = (hypothesis_text or "").lower()
    if verdict == "refuted":
        return CLAIM_COUNTEREXAMPLE
    if "mechanism" in text_l or "mediated by" in text_l or "drives" in text_l:
        return CLAIM_MECHANISTIC
    return None


def build_mechanism_object(
    *,
    claim: str,
    mechanism: str | None,
    evidence: dict[str, Any],
    failure_modes: list[str] | None = None,
) -> dict[str, Any] | None:
    """P2.3 — reusable mechanism, not just correlation text."""
    if not mechanism and not evidence.get("verdict_reason"):
        return None
    modes = list(failure_modes or [])
    text_l = claim.lower()
    if "sparse" in text_l or "fragmentation" in text_l:
        modes.append("sparse_regime_breakdown")
    if "assortativity" in text_l:
        modes.append("extreme_assortativity")
    return {
        "claim": claim[:500],
        "mechanism": (mechanism or str(evidence.get("verdict_reason") or ""))[:600],
        "evidence": str(evidence.get("verdict_reason") or "")[:400],
        "failure_modes": modes[:6],
    }


def build_canonical_finding(
    *,
    claim_id: str,
    claim: str,
    claim_type: str | None,
    replication_level: str,
    confidence: float,
    verification_method: str | None,
    primary_theme: str,
    secondary_themes: list[str],
    mechanism_obj: dict[str, Any] | None,
    evidence_hash: str,
    verification_hash: str,
    node_role: str,
) -> dict[str, Any]:
    """P4.1 canonical finding ledger entry."""
    return {
        "claim_id": claim_id,
        "claim": claim,
        "claim_type": claim_type,
        "replication_level": replication_level,
        "confidence": round(float(confidence), 4),
        "verification_method": verification_method,
        "themes": [primary_theme, *secondary_themes],
        "primary_theme": primary_theme,
        "secondary_themes": secondary_themes,
        "mechanisms": [mechanism_obj] if mechanism_obj else [],
        "evidence_hashes": [evidence_hash],
        "verification_hash": verification_hash,
        "node_role": node_role,
    }


def is_discovery_node(node: Any) -> bool:
    """True if node may expand, count as finding, or appear in paper."""
    role = getattr(node, "node_role", None) or NODE_ROLE_DISCOVERY
    return role != NODE_ROLE_CONTROL


def paper_eligible_finding(finding: dict[str, Any]) -> bool:
    """P5.1 — filter for paper compilation."""
    if finding.get("node_role") == NODE_ROLE_CONTROL:
        return False
    if finding.get("replication_level") == REPLICATION_T1 and float(finding.get("confidence") or 0) < 0.5:
        return False
    return True


def should_retest_inconclusive(
    node: Any,
    *,
    info_gain_min: float = 0.35,
    retry_cost: float = 0.30,
) -> bool:
    """P3.2 — retest only when expected information gain exceeds cost."""
    score = float(getattr(node, "frontier_score", None) or 0.0)
    relevance = float(getattr(node, "question_relevance_score", None) or 0.0)
    ig = max(score, relevance * 0.8)
    return ig >= max(info_gain_min, retry_cost)
