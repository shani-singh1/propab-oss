"""Campaign diagnostics (fixes.md P4.2, P5): online metrics + frontier snapshots."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from propab.hypothesis_tree import HypothesisTree
from propab.research_quality import (
    REPLICATION_T1,
    REPLICATION_T2,
    REPLICATION_T3,
    compute_theme_concentration,
    compute_theme_entropy,
    extract_theme_vector,
)


def infer_hypothesis_theme(text: str) -> str:
    """Coarse theme bucket — delegates to extract_theme_vector (P4.1)."""
    primary, _, _ = extract_theme_vector(text)
    return primary


def classify_verification_method(evidence_summary: str) -> str:
    raw = evidence_summary or ""
    low = raw.lower()
    # Decide verified-true/false from the PARSED INTEGER counters, not substrings:
    # the old ``"verified_true" in low`` matched ``"verified_true_steps": 0`` (which
    # actually means NOT verified) and mislabeled refuted/statistical nodes as
    # ``symbolic_identity``. Fall back to the ``"verified": true/false`` literal only
    # when the evidence has no structured counters.
    ev = parse_evidence_obj(raw)
    vt = int(ev.get("verified_true_steps") or 0)
    vf = int(ev.get("verified_false_steps") or 0)
    has_counters = ("verified_true_steps" in ev) or ("verified_false_steps" in ev)

    if vf > 0 or "counterexample" in low or '"verified": false' in low:
        return "counterexample"
    if vt > 0 or (not has_counters and '"verified": true' in low):
        if any(k in low for k in ("certificate", "identity", "parametric", "identically")):
            return "symbolic_identity"
        if any(k in low for k in ("exhaust", "scan", "range", "up to", "for n in", "for n ≤")):
            return "finite_scan"
        return "symbolic_identity"
    if any(k in low for k in ("p_value", "significance", "effect_size", "metric_value")):
        return "statistical"
    return "unknown"


def _theme_lifetime(tree: HypothesisTree) -> dict[str, int]:
    """P4.2 — generations spanned per theme (max gen - min gen per theme)."""
    by_theme: dict[str, list[int]] = {}
    for n in tree.nodes.values():
        tid = n.primary_theme or n.theme_id or "general"
        by_theme.setdefault(tid, []).append(int(n.generation))
    return {
        t: (max(gens) - min(gens) if gens else 0)
        for t, gens in by_theme.items()
    }


def _replication_health(tree: HypothesisTree) -> dict[str, int]:
    """P5 — T1/T2/T3 distribution on tested nodes."""
    counts = {REPLICATION_T1: 0, REPLICATION_T2: 0, REPLICATION_T3: 0}
    for n in tree.nodes.values():
        if n.verdict == "pending":
            continue
        tier = n.replication_level or REPLICATION_T1
        if tier in counts:
            counts[tier] += 1
    return counts


def _knowledge_velocity(tree: HypothesisTree, *, started_at_mono: float | None = None) -> dict[str, float]:
    """P5 — findings and mechanisms per hour (approximate from node counts)."""
    elapsed_h = max(0.01, (time.monotonic() - started_at_mono) / 3600.0) if started_at_mono else 1.0
    confirmed = sum(1 for n in tree.nodes.values() if n.verdict == "confirmed")
    with_mech = sum(1 for n in tree.nodes.values() if n.mechanism or (n.finding and n.finding.get("mechanisms")))
    return {
        "findings_per_hour": round(confirmed / elapsed_h, 3),
        "mechanisms_per_hour": round(with_mech / elapsed_h, 3),
        "elapsed_hours": round(elapsed_h, 3),
    }


def frontier_snapshot(
    tree: HypothesisTree,
    *,
    campaign_started_mono: float | None = None,
    prior_theme_histogram: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Counts + P4.2 theme evolution + P5 campaign analytics."""
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
    decisive = by_verdict.get("confirmed", 0) + by_verdict.get("refuted", 0)

    theme_entropy = compute_theme_entropy(theme_histogram)
    theme_concentration = compute_theme_concentration(theme_histogram)
    general_frac = round(theme_histogram.get("general", 0) / max(1, len(nodes)), 4)

    theme_drift = 0.0
    if prior_theme_histogram:
        keys = set(prior_theme_histogram) | set(theme_histogram)
        drift = sum(
            abs(theme_histogram.get(k, 0) - prior_theme_histogram.get(k, 0))
            for k in keys
        )
        theme_drift = round(drift / max(1, len(nodes)), 4)

    lineages = [n.lineage_length or tree.lineage_length(n.id) for n in nodes.values()]
    avg_lineage = round(sum(lineages) / len(lineages), 2) if lineages else 0.0

    return {
        "generated": len(nodes),
        "executed": executed,
        "pending": pending,
        "tested": executed,
        "frontier_size": len(tree.frontier),
        "confirmed": by_verdict.get("confirmed", 0),
        "by_verdict": by_verdict,
        "theme_histogram": theme_histogram,
        "claim_histogram": claim_histogram,
        "max_depth": max((n.depth for n in nodes.values()), default=0),
        "max_lineage": max(lineages) if lineages else 0,
        "avg_lineage": avg_lineage,
        "generation_histogram": tree.generation_histogram(),
        "theme_entropy": theme_entropy,
        "theme_concentration": theme_concentration,
        "theme_lifetime": _theme_lifetime(tree),
        "general_theme_fraction": general_frac,
        "closure_ratio": round(decisive / executed, 4) if executed else 0.0,
        "replication_health": _replication_health(tree),
        "knowledge_velocity": _knowledge_velocity(tree, started_at_mono=campaign_started_mono),
        "search_coherence": {
            "theme_drift": theme_drift,
            "lineage_drift": avg_lineage,
        },
        "ledger_size": len(tree.finding_ledger),
    }


def parse_evidence_obj(evidence_summary: str) -> dict[str, Any]:
    if not evidence_summary:
        return {}
    raw = evidence_summary.strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass
    m = re.search(r"evidence=(\{.*?\});", evidence_summary)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}
