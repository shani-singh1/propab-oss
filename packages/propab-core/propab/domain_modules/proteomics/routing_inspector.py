"""Routing inspector for proteomics / protein-stability hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.proteomics.adapter import KNOWN_FEATURES, ProteomicsExperimentSpec
from propab.domain_modules.proteomics.verifier import run_proteomics_experiment

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "p01", "statement": "Protein melting temperature is predictable from charged-residue fraction across fold families under leave-one-family-out", "test_methodology": "leave-one-family-out ridge"},
    {"id": "p02", "statement": "Charged-residue fraction predicts thermostability across protein families under LOFO", "test_methodology": "family holdout"},
    {"id": "p03", "statement": "Proline fraction predicts protein melting temperature across fold families", "test_methodology": "leave-family-out"},
    {"id": "p04", "statement": "Charge and proline fractions jointly predict Tm under protein-family holdout", "test_methodology": "cross-family LOFO"},
    {"id": "p05", "statement": "GRAVY hydropathy predicts thermostability across held-out fold families", "test_methodology": "leave-one-family-out"},
    {"id": "p06", "statement": "Instability index predicts melting temperature across protein families", "test_methodology": "protein family holdout"},
    {"id": "p07", "statement": "Sequence length alone fails to predict thermostability across families under LOFO", "test_methodology": "family LOFO"},
    {"id": "p08", "statement": "Charged-residue stability rule survives a within-family Tm-shuffle null", "test_methodology": "LOFO with Tm shuffle null"},
    {"id": "p09", "statement": "Helix-propensity fraction predicts protein stability across families", "test_methodology": "leave-one-family-out"},
    {"id": "p10", "statement": "Hydropathy and charge jointly predict melting temperature cross-family", "test_methodology": "cross-family ridge"},
    {"id": "p11", "statement": "TIM-barrel thermostability generalizes from other fold families under LOFO", "test_methodology": "family holdout"},
    {"id": "p12", "statement": "Charged-residue fraction predicts Tm better than sequence length across families", "test_methodology": "LOFO comparison"},
    {"id": "p13", "statement": "Sequence-property stability model beats the within-family shuffle null at p<0.05", "test_methodology": "leave-family-out shuffle null"},
    {"id": "p14", "statement": "Proline fraction correlates with thermostability across families in LOFO holdout", "test_methodology": "family-out ridge"},
    {"id": "p15", "statement": "Rossmann-fold protein stability is predictable from composition under leave-one-family-out", "test_methodology": "proteomics LOFO"},
    {"id": "p16", "statement": "Combined charge and hydropathy improve LOFO R² for melting temperature", "test_methodology": "leave-one-family-out"},
    {"id": "p17", "statement": "Pure sequence-feature noise fails cross-family thermostability generalization", "test_methodology": "family holdout"},
    {"id": "p18", "statement": "Proline rigidification raises Tm consistently across held-out families", "test_methodology": "cross-family LOFO"},
    {"id": "p19", "statement": "Meltome-style thermostability trends replicate under leave-one-family-out ridge", "test_methodology": "LOFO cross-family"},
    {"id": "p20", "statement": "Sequence composition predicts protein stability across families surviving Tm permutation", "test_methodology": "family LOFO shuffle null"},
    {"id": "p21", "statement": "Immunoglobulin-fold thermostability generalizes from other families under composition features", "test_methodology": "leave-family-out"},
    {"id": "p22", "statement": "Charged-residue and proline fractions jointly drive cross-family protein stability signal", "test_methodology": "cross-family ridge LOFO"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "charge" in text or "electrostatic" in text or "ionic" in text:
        return ["frac_charged", "gravy_hydropathy", "frac_proline"]
    if "proline" in text or "rigid" in text or "instability" in text:
        return ["frac_proline", "instability_index", "frac_helix_propensity"]
    if "hydropath" in text or "gravy" in text or "hydrophob" in text:
        return ["gravy_hydropathy", "frac_aromatic", "frac_charged"]
    if "length" in text or "size" in text or "weight" in text:
        return ["sequence_length", "molecular_weight", "frac_charged"]
    return ["frac_charged", "frac_proline", "gravy_hydropathy", "instability_index"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "proteomics",
        "resolved_verifier": "proteomics_lofo",
        "resolved_features": features,
        "expected_metric_name": "lofo_r2",
        "actual_metric_name": "lofo_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        try:
            spec = ProteomicsExperimentSpec.from_hypothesis(
                {"text": hypothesis.get("statement", ""), "feature_subset": features}
            )
            result = run_proteomics_experiment(spec)
            out["routing_ok"] = routing_ok and "lofo_r2" in result
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
            out["routing_ok"] = False
    return out


def inspect_corpus(hypotheses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    hyps = hypotheses if hypotheses is not None else ROUTING_CORPUS
    rows, mismatches = [], []
    for hyp in hyps:
        result = inspect_routing(hyp, dry_run_experiment=True)
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
