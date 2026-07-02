"""
Domain-agnostic research quality controls (fixes.md post network-resilience + contagion).

P0: verification closure, evidence validity
P1: replication tiers, closure-aware frontier
P2: theme vectors, mechanism objects
P3: inconclusive diagnosis, failure signatures
P4: canonical finding ledger with links
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

NODE_ROLE_DISCOVERY = "DISCOVERY"
NODE_ROLE_CONTROL = "CONTROL"

REPLICATION_T1 = "T1"
REPLICATION_T2 = "T2"
REPLICATION_T3 = "T3"

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

# P0.1 — fine-grained inconclusive subtypes (replaces coarse verification_failure)
INCONCLUSIVE_METRIC_MISSING = "metric_missing"
INCONCLUSIVE_METRIC_AMBIGUOUS = "metric_direction_ambiguous"
INCONCLUSIVE_REPLICATION_FAILED = "replication_failed"
INCONCLUSIVE_CODE_TIMEOUT = "code_timeout"
INCONCLUSIVE_SAMPLE_BUDGET = "sample_budget_exhausted"
INCONCLUSIVE_EXPERIMENT_CONFLICT = "experiment_conflict"
INCONCLUSIVE_DUPLICATE = "duplicate_evidence"
INCONCLUSIVE_CONTROL = "control_calibration"

INCONCLUSIVE_SUBTYPES = frozenset({
    INCONCLUSIVE_METRIC_MISSING,
    INCONCLUSIVE_METRIC_AMBIGUOUS,
    INCONCLUSIVE_REPLICATION_FAILED,
    INCONCLUSIVE_CODE_TIMEOUT,
    INCONCLUSIVE_SAMPLE_BUDGET,
    INCONCLUSIVE_EXPERIMENT_CONFLICT,
    INCONCLUSIVE_DUPLICATE,
    INCONCLUSIVE_CONTROL,
})

# P1.3 — retry policy keys
FAILURE_TIMEOUT = "timeout"
FAILURE_AMBIGUOUS = "ambiguous_metric"
FAILURE_REPLICATION = "replication_conflict"
FAILURE_CONFLICT = "experiment_conflict"
FAILURE_NO_METRIC = "no_metric"
FAILURE_BUDGET = "budget_exhausted"

_CONTROL_MARKERS = (
    "null hypothesis",
    "no falsifiable pattern",
    "no statistically significant effect beyond noise",
    "no effect beyond random",
)

_THEME_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("spectral", ("spectral gap", "eigenvalue", "laplacian", "adjacency matrix", "algebraic connectivity", "λ₂", "lambda_2", "spectral norm")),
    ("diffusion_dynamics", ("contagion", "diffusion", " sis ", " sir ", "transmission", "outbreak", "epidemic", "spreading", "infection")),
    ("normalization", ("pre-normalization", "post-normalization", "normalization", "k_source", "k_target")),
    ("assortativity", ("assortativity", "degree correlation", "rich-club")),
    ("clustering", ("clustering coefficient", "transitivity", "local clustering")),
    ("centrality", ("betweenness", "centrality", "eigenvector centrality", "degree-based removal")),
    ("degree_structure", ("gini", "degree distribution", "degree variance", "average degree", "k-core", "degree heterogeneity")),
    ("scale_free", ("barab", "scale-free", "scale free", "preferential attachment")),
    ("small_world", ("watts-strogatz", "watts strogatz", "small-world", "small world", "rewiring probability")),
    ("random_graph", ("erdős", "erdos", "erdős-rényi", "g(n,p)", "random graph")),
    ("percolation", ("percolation", "giant component", "critical threshold", "pc", "percolation threshold")),
    ("targeted_removal", ("targeted removal", "targeted attack", "node removal", "immunization")),
    ("sparse_regime", ("sparse graph", "average degree <", "fragmentation")),
    ("residue_class", (" mod ", " modulo ", "residue", "congruen")),
    ("parametric_family", ("parametric", "family", "closed-form")),
    ("finite_verification", ("exhaust", "scan", "up to n", "for all n", "counterexample")),
    ("unit_fraction", ("unit fraction", "egyptian", "1/n")),
    ("cache_policy", ("cache", "lru", "miss rate", "replacement policy", "belady")),
    ("scheduling", ("scheduling", "waiting time", "round-robin", "srpt", "queueing", "m/m/1")),
    ("auction", ("auction", "second-price", "bidder", "revenue equivalence")),
    ("collatz", ("collatz", "3n+1", "stopping time")),
    ("prime_gaps", ("prime gap", "cramér", "twin prime")),
    ("thermal_stability", ("t55_raw", "t70_raw", "t75_raw", "t80_raw", "thermal", "thermophilicity", "denaturation", "melting")),
    ("catalytic_geometry", ("triad_best_rmsd", "d1_d2_dist", "d2_d3_dist", "ramachandran", "catalytic triad", "geometry", "yxdd")),
    ("electrostatics", ("mean_pot", "net_charge", "isoelectric", "salt_bridge", "electrostatic", "pocket_hbond")),
    ("fold_similarity", ("foldseek", "tm_score", "lddt", "structural similarity")),
    ("surface_properties", ("camsol", "sasa", "hydrophobic", "surface area")),
    ("motif_structure", ("dgr_motif", "qg_motif", "sp_motif", "motif", "yxdd")),
]

_STRUCTURAL_FALLBACKS: list[tuple[str, tuple[str, ...], float]] = [
    ("diffusion_dynamics", ("network", "graph", "node", "edge"), 0.45),
    ("spectral", ("matrix", "rank", "eigen"), 0.40),
    ("percolation", ("threshold", "component", "redundancy"), 0.38),
]


def normalize_claim_strength(claim_type: str | None) -> str | None:
    if not claim_type:
        return None
    return _LEGACY_TO_STRENGTH.get(claim_type, claim_type)


def is_control_hypothesis(text: str) -> bool:
    core = re.split(r"\s*\(Question:", (text or "").strip(), maxsplit=1)[0].lower()
    return any(m in core for m in _CONTROL_MARKERS)


def infer_node_role(text: str) -> str:
    return NODE_ROLE_CONTROL if is_control_hypothesis(text) else NODE_ROLE_DISCOVERY


def extract_theme_vector(text: str) -> tuple[str, list[str], float]:
    """P4.1 — primary + secondary themes with confidence; shrink general bucket."""
    t = (text or "").lower()
    matched: list[str] = []
    for theme_id, keywords in _THEME_RULES:
        if any(k in t for k in keywords):
            matched.append(theme_id)
    if matched:
        confidence = min(0.95, 0.55 + 0.08 * len(matched))
        return matched[0], matched[1:], round(confidence, 3)

    for theme_id, keywords, conf in _STRUCTURAL_FALLBACKS:
        hits = sum(1 for k in keywords if k in t)
        if hits >= 2:
            return theme_id, [], conf

    if any(k in t for k in ("network", "graph", "contagion", "diffusion")):
        return "diffusion_dynamics", [], 0.35

    return "general", [], 0.20


def is_valid_evidence_for_hash(evidence: dict[str, Any]) -> bool:
    """P0.3 — reject empty/placeholder payloads before hashing."""
    if not evidence:
        return False
    vt = int(evidence.get("verified_true_steps") or 0)
    vf = int(evidence.get("verified_false_steps") or 0)
    if vt > 0 or vf > 0:
        return True
    nm = int(evidence.get("n_metric_steps") or 0)
    mv = evidence.get("metric_value")
    if nm > 0 and mv is not None:
        return True
    if evidence.get("p_value") is not None and mv is not None:
        return True
    reason = str(evidence.get("verdict_reason") or "").strip()
    if reason and "no metric" not in reason.lower():
        if any(k in reason.lower() for k in ("verified", "counterexample", "significance", "replicated")):
            return True
    return False


def compute_evidence_hash(evidence: dict[str, Any]) -> str | None:
    """P0.2/P0.3 — stable hash; None when evidence is empty/invalid."""
    if not is_valid_evidence_for_hash(evidence):
        return None
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
    _, secondary, _ = extract_theme_vector(hypothesis_text)
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
    """P0.1 — concrete inconclusive subtype for every inconclusive node."""
    fr = (failure_reason or "").lower()
    vr = (verdict_reason or evidence.get("verdict_reason") or "").lower()
    blob = f"{fr} {vr}"

    if "duplicate_evidence" in blob or "reused_evidence" in blob:
        return INCONCLUSIVE_DUPLICATE
    if "control_calibration" in blob or "control node" in blob:
        return INCONCLUSIVE_CONTROL

    if any(k in blob for k in (
        "timeout", "wall exceeded", "sub_agent_wall", "timeout_eviction",
        "revoke", "exceeded campaign_sub_agent",
    )):
        return INCONCLUSIVE_CODE_TIMEOUT
    if "budget" in blob or "sample_budget" in blob or "sub-agent failed" in blob:
        return INCONCLUSIVE_SAMPLE_BUDGET

    if "counterexample" in blob and "inconclusive" in blob:
        return INCONCLUSIVE_EXPERIMENT_CONFLICT
    if "conflicting" in blob or "verified_false" in blob:
        return INCONCLUSIVE_EXPERIMENT_CONFLICT

    if "no metric-bearing" in blob or "no metric" in blob:
        return INCONCLUSIVE_METRIC_MISSING
    if "ambiguous" in blob or "metric direction ambiguous" in blob:
        return INCONCLUSIVE_METRIC_AMBIGUOUS
    if "unreplicated" in blob or "need >=" in blob or "insufficient_samples" in blob:
        return INCONCLUSIVE_REPLICATION_FAILED
    if "no significance" in blob or "not decisive" in blob:
        return INCONCLUSIVE_METRIC_MISSING

    if not is_valid_evidence_for_hash(evidence):
        return INCONCLUSIVE_METRIC_MISSING
    return INCONCLUSIVE_METRIC_AMBIGUOUS


def failure_signature_from_reason(inconclusive_reason: str | None, *, verdict_reason: str = "") -> str | None:
    """P1.3 — compact failure signature for retry policy."""
    r = (inconclusive_reason or "").lower()
    vr = (verdict_reason or "").lower()
    if r == INCONCLUSIVE_CODE_TIMEOUT or "timeout" in vr:
        return FAILURE_TIMEOUT
    if r == INCONCLUSIVE_METRIC_AMBIGUOUS or "ambiguous" in vr:
        return FAILURE_AMBIGUOUS
    if r == INCONCLUSIVE_REPLICATION_FAILED or "unreplicated" in vr:
        return FAILURE_REPLICATION
    if r == INCONCLUSIVE_EXPERIMENT_CONFLICT:
        return FAILURE_CONFLICT
    if r == INCONCLUSIVE_METRIC_MISSING:
        return FAILURE_NO_METRIC
    if r == INCONCLUSIVE_SAMPLE_BUDGET:
        return FAILURE_BUDGET
    return None


def retry_policy_for_signature(signature: str | None) -> dict[str, Any]:
    """P1.3 — subtype-specific verification escalation hints for worker dispatch."""
    if signature == FAILURE_TIMEOUT:
        return {"prefer_smaller_experiment": True, "max_steps_scale": 0.7, "sandbox_timeout_scale": 0.85}
    if signature == FAILURE_AMBIGUOUS:
        return {"min_metric_steps_boost": 1, "extra_significance_rounds": 1}
    if signature == FAILURE_REPLICATION:
        return {"min_metric_steps_boost": 2, "require_replication": True}
    if signature == FAILURE_CONFLICT:
        return {"require_alt_method": True, "prefer_symbolic_check": True}
    if signature == FAILURE_NO_METRIC:
        return {"require_metric_step": True, "min_metric_steps_boost": 1}
    if signature == FAILURE_BUDGET:
        return {"prefer_smaller_experiment": True, "max_steps_scale": 0.6}
    return {}


def build_verification_escalation(
    node: Any,
    *,
    parent: Any | None = None,
) -> dict[str, Any]:
    """P0.2/P1.3 — merge node + parent retry hints for sub-agent payload."""
    base_min = 2
    policy: dict[str, Any] = {"min_metric_steps": base_min}

    sig = getattr(node, "failure_signature", None)
    if not sig and parent is not None:
        sig = failure_signature_from_reason(
            getattr(parent, "inconclusive_reason", None),
            verdict_reason=str(getattr(parent, "evidence_summary", "") or "")[:400],
        )
    if getattr(node, "expansion_type", None) == "retest" and parent is not None:
        sig = sig or failure_signature_from_reason(getattr(parent, "inconclusive_reason", None))

    pol = retry_policy_for_signature(sig)
    boost = int(pol.get("min_metric_steps_boost") or 0)
    if boost:
        policy["min_metric_steps"] = base_min + boost
    policy.update({k: v for k, v in pol.items() if k != "min_metric_steps_boost"})
    policy["failure_signature"] = sig
    policy["inconclusive_subtype"] = getattr(node, "inconclusive_reason", None) or getattr(parent, "inconclusive_reason", None)
    return policy


def estimate_closure_probability(
    node: Any,
    *,
    parent: Any | None = None,
    evidence: dict[str, Any] | None = None,
) -> float:
    """P1.2 — likelihood experiment yields decisive (confirmed/refuted) outcome."""
    score = 0.38
    ev = evidence or {}
    repl = getattr(node, "replication_level", None) or ""
    if repl == REPLICATION_T2:
        score += 0.12
    elif repl == REPLICATION_T3:
        score += 0.18

    vt = int(ev.get("verified_true_steps") or 0)
    vf = int(ev.get("verified_false_steps") or 0)
    nm = int(ev.get("n_metric_steps") or 0)
    if vt >= 2 or vf >= 1:
        score += 0.22
    elif vt == 1 or nm >= 2:
        score += 0.12
    elif nm == 1:
        score += 0.05

    rel = float(getattr(node, "question_relevance_score", None) or 0.5)
    score += 0.10 * min(1.0, rel)

    if parent is not None:
        pv = getattr(parent, "verdict", None)
        pr = getattr(parent, "inconclusive_reason", None)
        if pv == "confirmed":
            score += 0.10
        elif pr == INCONCLUSIVE_REPLICATION_FAILED:
            score += 0.15
        elif pr == INCONCLUSIVE_METRIC_AMBIGUOUS:
            score += 0.05
        elif pr in (INCONCLUSIVE_CODE_TIMEOUT, INCONCLUSIVE_SAMPLE_BUDGET):
            score -= 0.12
        elif pr == INCONCLUSIVE_METRIC_MISSING:
            score -= 0.08

    if getattr(node, "expansion_type", None) == "retest":
        score += 0.08

    theme_conf = float(getattr(node, "theme_confidence", None) or 0.5)
    if theme_conf >= 0.5:
        score += 0.05

    return round(max(0.05, min(0.92, score)), 4)


def classify_claim_strength(
    evidence: dict[str, Any],
    verdict: str,
    *,
    hypothesis_text: str = "",
) -> str | None:
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


def _extract_conditions(text: str) -> str:
    t = (text or "").strip()
    for prefix in ("when ", "if ", "in ", "under ", "provided "):
        if t.lower().startswith(prefix):
            return t[:300]
    m = re.search(r"\b(when|if|under|provided that)\b[^.]{10,200}", t, re.I)
    return m.group(0)[:300] if m else ""


def build_mechanism_object(
    *,
    claim: str,
    mechanism: str | None,
    evidence: dict[str, Any],
    failure_modes: list[str] | None = None,
    verdict: str = "confirmed",
) -> dict[str, Any] | None:
    """P2.1 — causal mechanism object, not metadata echo."""
    vr = str(mechanism or evidence.get("verdict_reason") or "").strip()
    if not vr and verdict not in ("confirmed", "refuted"):
        return None

    text_l = claim.lower()
    cause = ""
    effect = ""

    if "λ" in claim or "eigenvalue" in text_l or "spectral" in text_l:
        cause = "Spectral structure of the network adjacency operator"
        effect = vr or "Alters contagion speed or extent relative to baseline"
    elif "percolation" in text_l or "threshold" in text_l:
        cause = "Connectivity regime relative to percolation threshold"
        effect = vr or "Changes final outbreak extent or immunization cost"
    elif "normalization" in text_l:
        cause = "Degree normalization policy during transmission"
        effect = vr or "Shifts epidemic threshold or strain competition outcome"
    elif verdict == "refuted":
        cause = claim[:400]
        effect = vr or "Observed outcome contradicts the stated hypothesis"
    else:
        cause = claim[:400]
        effect = vr[:600] if vr else "Metric direction supports the claim under tested conditions"

    modes = list(failure_modes or [])
    if "sparse" in text_l:
        modes.append("sparse_regime_breakdown")
    if "assortativity" in text_l:
        modes.append("extreme_assortativity")

    evidence_items: list[str] = []
    if evidence.get("p_value") is not None:
        evidence_items.append(f"p={evidence['p_value']}")
    if evidence.get("effect_size") is not None:
        evidence_items.append(f"effect={evidence['effect_size']}")
    if evidence.get("metric_value") is not None:
        evidence_items.append(f"metric={evidence['metric_value']}")
    if vr:
        evidence_items.append(vr[:200])

    return {
        "cause": cause[:500],
        "effect": effect[:600],
        "conditions": _extract_conditions(claim) or "As stated in the tested hypothesis",
        "failure_modes": modes[:6],
        "evidence": evidence_items[:8],
        # legacy compat
        "claim": claim[:500],
        "mechanism": effect[:600],
    }


def build_refutation_mechanism(
    *,
    claim: str,
    evidence: dict[str, Any],
    verdict_reason: str,
) -> dict[str, Any]:
    """P2.2 — why a refuted hypothesis failed."""
    return build_mechanism_object(
        claim=claim,
        mechanism=verdict_reason,
        evidence=evidence,
        failure_modes=["counterexample_found"] if "counterexample" in verdict_reason.lower() else ["hypothesis_not_supported"],
        verdict="refuted",
    ) or {
        "cause": claim[:500],
        "effect": verdict_reason[:600],
        "conditions": _extract_conditions(claim),
        "failure_modes": ["hypothesis_not_supported"],
        "evidence": [verdict_reason[:200]],
    }


def infer_finding_links(
    ledger: list[dict[str, Any]],
    entry: dict[str, Any],
    *,
    parent_id: str | None = None,
) -> dict[str, list[str]]:
    """P3.2 — cross-finding relations."""
    links: dict[str, list[str]] = {
        "supports": [],
        "contradicts": [],
        "extends": [],
        "depends_on": [],
    }
    cid = str(entry.get("claim_id") or "")
    theme = entry.get("primary_theme") or ""
    verdict = entry.get("verdict") or ""

    if parent_id:
        links["depends_on"].append(parent_id)

    for prev in ledger:
        pid = str(prev.get("claim_id") or "")
        if not pid or pid == cid:
            continue
        pt = prev.get("primary_theme") or ""
        pv = prev.get("verdict") or ""
        if theme and pt == theme:
            if pv == "confirmed" and verdict == "confirmed":
                links["extends"].append(pid)
            elif pv == "refuted" and verdict == "confirmed":
                links["contradicts"].append(pid)
            elif pv == "confirmed":
                links["supports"].append(pid)
        sec = set(prev.get("secondary_themes") or [])
        if theme in sec or pt in set(entry.get("secondary_themes") or []):
            links["extends"].append(pid)

    for k in links:
        links[k] = list(dict.fromkeys(links[k]))[:8]
    return links


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
    evidence_hash: str | None,
    verification_hash: str,
    node_role: str,
    verdict: str,
    theme_confidence: float = 0.5,
    links: dict[str, list[str]] | None = None,
    failure_signature: str | None = None,
) -> dict[str, Any]:
    """P3.1/P4.1 canonical finding ledger entry."""
    return {
        "claim_id": claim_id,
        "claim": claim,
        "claim_type": claim_type,
        "verdict": verdict,
        "replication_level": replication_level,
        "confidence": round(float(confidence), 4),
        "verification_method": verification_method,
        "themes": [primary_theme, *secondary_themes],
        "primary_theme": primary_theme,
        "secondary_themes": secondary_themes,
        "theme_confidence": round(float(theme_confidence), 3),
        "mechanisms": [mechanism_obj] if mechanism_obj else [],
        "evidence_hashes": [evidence_hash] if evidence_hash else [],
        "verification_hash": verification_hash,
        "node_role": node_role,
        "supports": (links or {}).get("supports", []),
        "contradicts": (links or {}).get("contradicts", []),
        "extends": (links or {}).get("extends", []),
        "depends_on": (links or {}).get("depends_on", []),
        "failure_signature": failure_signature,
    }


def compute_theme_entropy(theme_histogram: dict[str, int]) -> float:
    """P4.2 — Shannon entropy over theme distribution."""
    total = sum(theme_histogram.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in theme_histogram.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return round(ent, 4)


def compute_theme_concentration(theme_histogram: dict[str, int]) -> float:
    """P4.2 — fraction of nodes in the dominant theme."""
    if not theme_histogram:
        return 0.0
    total = sum(theme_histogram.values())
    if total <= 0:
        return 0.0
    return round(max(theme_histogram.values()) / total, 4)


def is_discovery_node(node: Any) -> bool:
    role = getattr(node, "node_role", None) or NODE_ROLE_DISCOVERY
    return role != NODE_ROLE_CONTROL


def paper_eligible_finding(finding: dict[str, Any]) -> bool:
    if finding.get("node_role") == NODE_ROLE_CONTROL:
        return False
    if finding.get("verdict") not in (None, "confirmed"):
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
    score = float(getattr(node, "frontier_score", None) or 0.0)
    relevance = float(getattr(node, "question_relevance_score", None) or 0.0)
    ig = max(score, relevance * 0.8)
    sig = getattr(node, "failure_signature", None)
    if sig in (FAILURE_REPLICATION, FAILURE_AMBIGUOUS):
        retry_cost *= 0.85
    return ig >= max(info_gain_min, retry_cost)
