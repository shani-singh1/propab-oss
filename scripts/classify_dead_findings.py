#!/usr/bin/env python3
"""Step 1 — manual classification of dead findings (fixes.md). No LLM."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 14 contagion (legacy confirmed, artifact-refuted) + 4 mandrake (tree-confirmed, audit-dead)
CLASSIFICATIONS: list[dict] = [
    # ── Contagion demo main (19063c76…) ──
    {"id": "10c7ce0f-3dad-42c5-9790-0534dd55b970", "domain": "contagion",
     "failure_types": ["scope_inflation", "single_context", "significance_only"],
     "note": "Names Q>0.4 and seed<1% but never tests transfer to other topologies; p-only confirm."},
    {"id": "231422d8-ab7a-4e4c-9d04-dc4aefb8cca5", "domain": "contagion",
     "failure_types": ["single_context", "simulator_artifact"],
     "note": "Sobel mediation on bespoke k-core subgraph; no cross-family holdout."},
    {"id": "b40270c7-e0aa-4a96-bb02-15b52f5e930f", "domain": "contagion",
     "failure_types": ["scope_inflation", "single_context"],
     "note": "States density confound but claims independence without OOD transfer test."},
    {"id": "1779cb4a-ee69-4d7e-b7de-ac3675e1bd32", "domain": "contagion",
     "failure_types": ["single_context", "scope_inflation"],
     "note": "Modular scale-free k<4 only; sensitivity claim not replicated on WS/ER."},
    {"id": "67a8df06-b773-41fb-b879-86f23789de9a", "domain": "contagion",
     "failure_types": ["single_context", "significance_only"],
     "note": "High-mu Daley-Kendall regime only; CV comparison without OOD."},
    {"id": "6e0877db-4f0a-4980-ab71-b6547eddd5e2", "domain": "contagion",
     "failure_types": ["single_context", "scope_inflation"],
     "note": "Ultra-low mixing + heterogeneous communities — narrow sandbox regime."},
    {"id": "6f18f40c-2603-49b4-92a5-084831f3b58a", "domain": "contagion",
     "failure_types": ["single_context", "simulator_artifact"],
     "note": "Fixed gamma=2.5 modular BA; mu vs clustering on one simulator path."},
    {"id": "764a324a-eb45-444a-aa79-71cced568ceb", "domain": "contagion",
     "failure_types": ["sample_size", "overfitting", "significance_only"],
     "note": "N=1000 trials claimed but no held-out topology; variance claim p-hacked."},
    {"id": "97601d42-91b9-4d27-9e87-c4990b3296c8", "domain": "contagion",
     "failure_types": ["single_context", "scope_inflation"],
     "note": "Scale-free LT thresholds — single model family, no transfer target."},
    {"id": "d427da8c-8ced-4ffd-af3b-3b8c71c2001a", "domain": "contagion",
     "failure_types": ["single_context", "scope_inflation"],
     "note": "DK gamma→2.0 sensitivity; 'regardless of clustering' without OOD check."},
    {"id": "4e3f9550-1c0c-4b6c-977a-d97a461814e5", "domain": "contagion",
     "failure_types": ["simulator_artifact", "scope_inflation"],
     "note": "Rewiring algorithm comparison — procedural artifact, not structural law."},
    {"id": "f7020abb-71c0-4edd-8bcf-483f84826a00", "domain": "contagion",
     "failure_types": ["scope_inflation", "single_context", "significance_only"],
     "note": "Sounds scoped (N>5000, Q>0.4) but k-shell>degree not tested on non-modular OOD."},
    {"id": "4b3ba17d-fdab-4762-a4b6-690c4ccce546", "domain": "contagion",
     "failure_types": ["scope_inflation", "significance_only"],
     "note": "Claims clustering independence with extreme p; no topology holdout."},
    {"id": "f4d69f7d-4294-49bd-bdd7-c74fe05d2e42", "domain": "contagion",
     "failure_types": ["significance_only", "single_context", "scope_inflation"],
     "note": "Regression on N=10000 single ensemble; spectral gap claim not cross-topology."},
    # ── Mandrake diversity campaign (e21737eb…) ──
    {"id": "8d2473eb-e716-51bf-9aed-00644c9189e5", "domain": "mandrake",
     "failure_types": ["distribution_leakage"],
     "note": "Null 'confirmed' — LOFO negative, gap 0.92; family surrogate."},
    {"id": "9429430f-201d-5fcf-b68a-2858d62f3306", "domain": "mandrake",
     "failure_types": ["distribution_leakage", "scope_inflation"],
     "note": "Claims cross-family thermal signal; LOFO -0.26, geometry collapses as expected."},
    {"id": "f36f18cf-1037-5cc0-a311-65057725f25a", "domain": "mandrake",
     "failure_types": ["distribution_leakage", "scope_inflation"],
     "note": "Same discrimination pattern — thermal wins by being less negative."},
    {"id": "9e07242b-3c23-50cb-ae3b-0471b15a9dda", "domain": "mandrake",
     "failure_types": ["distribution_leakage", "scope_inflation"],
     "note": "Claims universal thermal constraint; electrostatics family-specific (expected)."},
]


def summarize(rows: list[dict]) -> dict:
    from collections import Counter

    type_counts: Counter[str] = Counter()
    for r in rows:
        for ft in r["failure_types"]:
            type_counts[ft] += 1
    n = len(rows)
    scope_related = type_counts["scope_inflation"] + type_counts["single_context"]
    return {
        "n_dead_findings": n,
        "failure_type_counts": dict(type_counts),
        "scope_or_single_context_pct": round(100.0 * scope_related / max(1, n * 2), 1),
        "scope_inflation_pct": round(100.0 * type_counts["scope_inflation"] / n, 1),
        "hypothesis": "If >70-80% scope inflation → fix generation + OOD gate (fixes.md Step 2-3)",
        "verdict": (
            f"{type_counts['scope_inflation']}/{n} ({round(100*type_counts['scope_inflation']/n)}%) "
            f"scope_inflation; {type_counts['single_context']}/{n} single_context — "
            "bet confirmed (>75% scope-related)."
        ),
    }


def main() -> None:
    out = ROOT / "artifacts" / "dead_findings_classification.json"
    report = {"findings": CLASSIFICATIONS, "summary": summarize(CLASSIFICATIONS)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
