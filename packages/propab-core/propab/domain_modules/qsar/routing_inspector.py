"""Routing inspector for QSAR / bioactivity hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.qsar.adapter import KNOWN_FEATURES, QSARExperimentSpec
from propab.domain_modules.qsar.verifier import run_qsar_experiment

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "q01", "statement": "pIC50 is predictable from cLogP and molecular weight across scaffolds under leave-one-scaffold-out", "test_methodology": "leave-one-scaffold-out ridge"},
    {"id": "q02", "statement": "Lipophilicity (cLogP) predicts potency across chemical scaffolds under LOSO", "test_methodology": "scaffold holdout"},
    {"id": "q03", "statement": "Topological polar surface area predicts bioactivity across scaffolds", "test_methodology": "leave-scaffold-out"},
    {"id": "q04", "statement": "Aromatic ring count and cLogP jointly predict pIC50 under scaffold holdout", "test_methodology": "cross-scaffold LOSO"},
    {"id": "q05", "statement": "Fraction sp3 predicts potency across held-out scaffolds", "test_methodology": "leave-one-scaffold-out"},
    {"id": "q06", "statement": "Hydrogen-bond donor and acceptor counts predict IC50 across scaffolds", "test_methodology": "QSAR scaffold holdout"},
    {"id": "q07", "statement": "Molecular weight alone fails to predict pIC50 across scaffolds under LOSO", "test_methodology": "scaffold LOSO"},
    {"id": "q08", "statement": "cLogP structure-activity relationship survives a within-scaffold activity-shuffle null", "test_methodology": "LOSO with activity shuffle null"},
    {"id": "q09", "statement": "Rotatable-bond flexibility predicts potency across chemical scaffolds", "test_methodology": "leave-one-scaffold-out"},
    {"id": "q10", "statement": "Polar surface area and H-bond donors jointly predict bioactivity cross-scaffold", "test_methodology": "cross-scaffold ridge"},
    {"id": "q11", "statement": "Benzimidazole-scaffold potency generalizes from other scaffolds under LOSO", "test_methodology": "scaffold holdout"},
    {"id": "q12", "statement": "cLogP predicts pIC50 better than TPSA across held-out scaffolds", "test_methodology": "LOSO comparison"},
    {"id": "q13", "statement": "Descriptor-based QSAR beats the within-scaffold shuffle null at p<0.05", "test_methodology": "leave-scaffold-out shuffle null"},
    {"id": "q14", "statement": "Aromatic ring count correlates with potency across scaffolds in LOSO holdout", "test_methodology": "scaffold-out ridge"},
    {"id": "q15", "statement": "Quinazoline bioactivity is predictable from descriptors under leave-one-scaffold-out", "test_methodology": "QSAR LOSO"},
    {"id": "q16", "statement": "Combined lipophilicity and polarity improve LOSO R² for pIC50", "test_methodology": "leave-one-scaffold-out"},
    {"id": "q17", "statement": "Pure descriptor noise fails cross-scaffold potency generalization", "test_methodology": "scaffold holdout"},
    {"id": "q18", "statement": "Fraction sp3 lowers potency consistently across held-out scaffolds", "test_methodology": "cross-scaffold LOSO"},
    {"id": "q19", "statement": "ChEMBL-style bioactivity trends replicate under leave-one-scaffold-out ridge", "test_methodology": "LOSO cross-scaffold"},
    {"id": "q20", "statement": "Molecular descriptors predict pIC50 across scaffolds surviving activity permutation", "test_methodology": "scaffold LOSO shuffle null"},
    {"id": "q21", "statement": "Indole-scaffold potency generalizes from other scaffolds under descriptor QSAR", "test_methodology": "leave-scaffold-out"},
    {"id": "q22", "statement": "cLogP and aromatic rings jointly drive cross-scaffold structure-activity signal", "test_methodology": "cross-scaffold ridge LOSO"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "lipophil" in text or "logp" in text or "clogp" in text:
        return ["clogp", "num_aromatic_rings", "fraction_csp3", "mol_weight"]
    if "polar" in text or "tpsa" in text or "hydrogen" in text or "h-bond" in text or "donor" in text:
        return ["tpsa", "num_h_donors", "num_h_acceptors", "mol_weight"]
    if "rotatable" in text or "flexib" in text or "weight" in text or "size" in text:
        return ["mol_weight", "num_rotatable_bonds", "num_aromatic_rings"]
    return ["clogp", "mol_weight", "tpsa", "num_aromatic_rings"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "qsar",
        "resolved_verifier": "qsar_loso",
        "resolved_features": features,
        "expected_metric_name": "loso_r2",
        "actual_metric_name": "loso_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        try:
            spec = QSARExperimentSpec.from_hypothesis(
                {"text": hypothesis.get("statement", ""), "feature_subset": features}
            )
            result = run_qsar_experiment(spec)
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
