"""Routing inspector for graph invariant hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.graph_invariants.adapter import (
    KNOWN_INVARIANTS,
    GraphInvariantNotIdentified,
    GraphInvariantSpec,
)
from propab.domain_modules.graph_invariants.verifier import run_graph_invariant_check

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "gi01", "statement": "Spectral gap correlates with clustering coefficient across SNAP network families", "test_methodology": "cross-network LOFO deterministic check"},
    {"id": "gi02", "statement": "Algebraic connectivity increases with average degree in scale-free graphs", "test_methodology": "held-out network family verification"},
    {"id": "gi03", "statement": "Diameter decreases as clustering increases in Watts-Strogatz graphs", "test_methodology": "graph invariant sweep"},
    {"id": "gi04", "statement": "Modularity tracks clustering coefficient across all graph families", "test_methodology": "SNAP subset invariant check"},
    {"id": "gi05", "statement": "Spectral gap predicts diameter on held-out Erdős-Rényi family", "test_methodology": "leave-family-out invariant test"},
    {"id": "gi06", "statement": "Barabási-Albert graphs show positive spectral-clustering correlation under LOFO", "test_methodology": "cross-network holdout"},
    {"id": "gi07", "statement": "Grid lattice graphs violate scale-free spectral-gap trend", "test_methodology": "deterministic invariant verification"},
    {"id": "gi08", "statement": "Average degree monotonically relates to algebraic connectivity across families", "test_methodology": "graph family holdout"},
    {"id": "gi09", "statement": "Clustering coefficient inversely tracks diameter in small-world networks", "test_methodology": "SNAP LOFO invariant"},
    {"id": "gi10", "statement": "Modularity and spectral gap co-vary on held-out barabasi albert family", "test_methodology": "cross-network deterministic check"},
    {"id": "gi11", "statement": "Graph invariant spectral gap exceeds 0.1 for all network families", "test_methodology": "family sweep"},
    {"id": "gi12", "statement": "Cross-family LOFO confirms clustering-spectral relationship", "test_methodology": "invariant correlation holdout"},
    {"id": "gi13", "statement": "Erdos renyi graphs show weaker modularity-clustering link than scale-free", "test_methodology": "SNAP family comparison"},
    {"id": "gi14", "statement": "Diameter invariant fails on held-out watts strogatz family", "test_methodology": "deterministic verification"},
    {"id": "gi15", "statement": "Algebraic connectivity spectral relationship holds all families", "test_methodology": "cross-network invariant"},
    {"id": "gi16", "statement": "Negative correlation between diameter and clustering on grid lattice holdout", "test_methodology": "LOFO graph invariant"},
    {"id": "gi17", "statement": "SNAP repository graphs: modularity predicts clustering under family holdout", "test_methodology": "deterministic invariant test"},
    {"id": "gi18", "statement": "Spectral gap band validation across network categories", "test_methodology": "graph invariant LOFO"},
    {"id": "gi19", "statement": "Scale-free graphs show spectral-gap clustering coupling", "test_methodology": "held-out BA family"},
    {"id": "gi20", "statement": "Graph family holdout refutes universal diameter-spectral identity", "test_methodology": "cross-network check"},
]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = True) -> dict[str, Any]:
    try:
        spec = GraphInvariantSpec.from_hypothesis(hypothesis)
    except GraphInvariantNotIdentified as exc:
        # DOM4: no graph invariant identified — refuse rather than default-route.
        return {
            "domain": "graph_invariants",
            "resolved_verifier": "graph_invariant_lofo",
            "resolved_invariants": [],
            "expected_metric_name": "invariant_correlation",
            "routing_ok": False,
            "error": str(exc),
        }
    invalid = [x for x in (spec.source_invariant, spec.target_invariant) if x not in KNOWN_INVARIANTS]
    routing_ok = not invalid
    out: dict[str, Any] = {
        "domain": "graph_invariants",
        "resolved_verifier": "graph_invariant_lofo",
        "resolved_invariants": [spec.source_invariant, spec.target_invariant],
        "expected_metric_name": "invariant_correlation",
        "routing_ok": routing_ok,
    }
    if dry_run_experiment and routing_ok:
        try:
            result = run_graph_invariant_check(spec)
            out["actual_metric_name"] = result.get("metric_name")
            out["metric_match"] = result.get("metric_name") == "invariant_correlation"
            out["routing_ok"] = routing_ok and bool(out.get("metric_match"))
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
            out["routing_ok"] = False
    else:
        out["actual_metric_name"] = "invariant_correlation"
        out["metric_match"] = True
    return out


def inspect_corpus(hypotheses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    hyps = hypotheses if hypotheses is not None else ROUTING_CORPUS
    rows, mismatches = [], []
    for hyp in hyps:
        result = inspect_routing(hyp)
        row = {"id": hyp.get("id"), "text_preview": str(hyp.get("statement", ""))[:120], **result}
        rows.append(row)
        if not result.get("routing_ok"):
            mismatches.append(row)
    return {
        "total": len(rows),
        "routing_ok": sum(1 for r in rows if r.get("routing_ok")),
        "routing_mismatches": len(mismatches),
        "mismatch_rate": round(len(mismatches) / max(len(rows), 1), 3),
        "mismatches": mismatches,
    }
