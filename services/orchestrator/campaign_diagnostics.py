"""Lightweight campaign diagnostics (fixes.md): logging only, no orchestration changes."""
from __future__ import annotations

import json
import re
from typing import Any

from propab.hypothesis_tree import HypothesisTree


def infer_hypothesis_theme(text: str) -> str:
    """Coarse theme bucket from hypothesis wording (domain-agnostic heuristics)."""
    t = (text or "").lower()
    if any(k in t for k in (" mod ", " modulo ", " residue", " ≡ ", " congruen")):
        return "residue_class"
    if any(k in t for k in ("parametric", "family", "identity", "closed-form", "closed form")):
        return "parametric_family"
    if any(k in t for k in ("exhaust", "search depth", "up to n", "for all n", "counterexample")):
        return "finite_verification"
    if any(k in t for k in ("density", "proportion", "mean number", "count of")):
        return "statistical_density"
    if any(k in t for k in ("unit fraction", "egyptian", "1/x", "1/n")):
        return "unit_fraction"
    return "general"


def classify_verification_method(evidence_summary: str) -> str:
    """Classify how a hypothesis was verified from worker evidence text."""
    raw = evidence_summary or ""
    low = raw.lower()
    if "counterexample" in low or "verified_false" in low or '"verified": false' in low:
        return "counterexample"
    if "verified_true" in low or '"verified": true' in low:
        if any(k in low for k in ("certificate", "identity", "parametric", "identically")):
            return "symbolic_identity"
        if any(k in low for k in ("exhaust", "scan", "range", "up to", "for n in", "for n ≤")):
            return "finite_scan"
        return "symbolic_identity"
    if any(k in low for k in ("p_value", "significance", "effect_size", "metric_value")):
        return "statistical"
    return "unknown"


def frontier_snapshot(tree: HypothesisTree) -> dict[str, Any]:
    """Counts for periodic frontier diagnostics (fixes.md P4.2)."""
    nodes = tree.nodes
    by_verdict: dict[str, int] = {}
    theme_histogram: dict[str, int] = {}
    claim_histogram: dict[str, int] = {}
    for n in nodes.values():
        by_verdict[n.verdict] = by_verdict.get(n.verdict, 0) + 1
        tid = n.primary_theme or n.theme_id or "general"
        theme_histogram[tid] = theme_histogram.get(tid, 0) + 1
        if n.claim_type:
            claim_histogram[n.claim_type] = claim_histogram.get(n.claim_type, 0) + 1
    pending = by_verdict.get("pending", 0)
    executed = sum(v for k, v in by_verdict.items() if k != "pending")
    return {
        "generated": len(nodes),
        "executed": executed,
        "pending": pending,
        "tested": executed,
        "frontier_size": len(tree.frontier),
        "confirmed": len(tree.confirmed),
        "by_verdict": by_verdict,
        "theme_histogram": theme_histogram,
        "claim_histogram": claim_histogram,
        "max_depth": max((n.depth for n in nodes.values()), default=0),
    }


def parse_evidence_obj(evidence_summary: str) -> dict[str, Any]:
    """Best-effort parse of evidence= JSON embedded in evidence_summary."""
    if not evidence_summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", evidence_summary)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}
