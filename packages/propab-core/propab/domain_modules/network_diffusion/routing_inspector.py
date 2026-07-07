"""Routing inspector for network-diffusion hypotheses (spec parsing only, no sim)."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.network_diffusion.adapter import STRUCTURAL_FEATURES, DiffusionSpec

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "nd01", "statement": "Degree heterogeneity (⟨k²⟩/⟨k⟩) increases SIR final outbreak size across real network families", "test_methodology": "cross-topology-family holdout with within-family shuffle null"},
    {"id": "nd02", "statement": "Higher degree Gini raises epidemic outbreak probability on held-out collaboration network", "test_methodology": "leave-network-family-out contagion verification"},
    {"id": "nd03", "statement": "Degree coefficient of variation predicts independent-cascade adoption size", "test_methodology": "cross-network cascade holdout"},
    {"id": "nd04", "statement": "Hub dominance (max-degree ratio) lowers the epidemic threshold in email contagion", "test_methodology": "SIR diffusion simulation on real graphs"},
    {"id": "nd05", "statement": "Clustering coefficient suppresses SIR final size on held-out email family", "test_methodology": "cross-topology-family label-shuffle null"},
    {"id": "nd06", "statement": "Degree assortativity increases contagion outbreak probability across topology families", "test_methodology": "held-out network diffusion holdout"},
    {"id": "nd07", "statement": "Independent-cascade influence spread tracks degree heterogeneity on real networks", "test_methodology": "cross-network cascade replication"},
    {"id": "nd08", "statement": "Degree variance predicts epidemic threshold takeoff across SNAP families", "test_methodology": "outbreak-probability holdout"},
    {"id": "nd09", "statement": "SIR spreading is faster on scale-free-like collaboration subgraphs than email subgraphs", "test_methodology": "cross-topology contagion simulation"},
    {"id": "nd10", "statement": "Inverse relationship between clustering and diffusion final size on held-out family", "test_methodology": "leave-family-out diffusion null test"},
    {"id": "nd11", "statement": "Degree Gini and outbreak probability co-vary under the independent cascade model", "test_methodology": "cross-network cascade holdout"},
    {"id": "nd12", "statement": "Heterogeneity-driven epidemic spreading replicates across email and collaboration networks", "test_methodology": "cross-topology-family SIR verification"},
    {"id": "nd13", "statement": "Max-degree ratio increases contagion final size on held-out collaboration network", "test_methodology": "network diffusion holdout"},
    {"id": "nd14", "statement": "Assortativity decreases outbreak probability in the SIS/SIR contagion model", "test_methodology": "cross-family shuffle-null diffusion test"},
    {"id": "nd15", "statement": "⟨k²⟩/⟨k⟩ governs the epidemic threshold consistently across real topology families", "test_methodology": "cross-topology-family holdout"},
]


def inspect_routing(hypothesis: dict[str, Any]) -> dict[str, Any]:
    spec = DiffusionSpec.from_hypothesis(hypothesis)
    feature_ok = spec.structural_feature in STRUCTURAL_FEATURES
    outcome_ok = spec.outcome in {"final_size", "outbreak_prob"}
    sim_ok = spec.simulator in {"sir", "cascade"}
    routing_ok = feature_ok and outcome_ok and sim_ok
    return {
        "domain": "network_diffusion",
        "resolved_verifier": "cross_topology_family_diffusion",
        "resolved_feature": spec.structural_feature,
        "resolved_outcome": spec.outcome,
        "resolved_simulator": spec.simulator,
        "expected_metric_name": "cross_family_diffusion_correlation",
        "routing_ok": routing_ok,
    }


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
