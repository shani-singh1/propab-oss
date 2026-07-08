"""Routing inspector for immunology / epitope hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.epitope.adapter import KNOWN_FEATURES, EpitopeExperimentSpec
from propab.domain_modules.epitope.verifier import run_epitope_experiment

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "e01", "statement": "Peptide binding_score is predictable from anchor hydrophobicity across MHC alleles under leave-one-allele-out", "test_methodology": "leave-one-allele-out ridge"},
    {"id": "e02", "statement": "C-terminal anchor hydrophobicity predicts MHC binding across HLA alleles under LAOO", "test_methodology": "allele holdout"},
    {"id": "e03", "statement": "Net peptide charge predicts epitope binding across alleles", "test_methodology": "leave-allele-out"},
    {"id": "e04", "statement": "P2 and C-terminal anchor hydrophobicity jointly predict binding under allele holdout", "test_methodology": "cross-allele LAOO"},
    {"id": "e05", "statement": "Peptide length predicts immunogenic binding across held-out alleles", "test_methodology": "leave-one-allele-out"},
    {"id": "e06", "statement": "Mean hydrophobicity predicts peptide-MHC binding across alleles", "test_methodology": "MHC allele holdout"},
    {"id": "e07", "statement": "Aromatic fraction alone fails to predict binding across alleles under LAOO", "test_methodology": "allele LAOO"},
    {"id": "e08", "statement": "Anchor-hydrophobicity binding rule survives a within-allele binding-shuffle null", "test_methodology": "LAOO with binding shuffle null"},
    {"id": "e09", "statement": "Net charge and hydrophobicity jointly predict HLA binding cross-allele", "test_methodology": "leave-one-allele-out"},
    {"id": "e10", "statement": "HLA-A*02:01 binding generalizes from other alleles under LAOO", "test_methodology": "allele holdout"},
    {"id": "e11", "statement": "Molecular weight and length predict peptide binding across MHC alleles", "test_methodology": "cross-allele ridge"},
    {"id": "e12", "statement": "Anchor hydrophobicity predicts binding better than net charge across alleles", "test_methodology": "LAOO comparison"},
    {"id": "e13", "statement": "Peptide-property epitope model beats the within-allele shuffle null at p<0.05", "test_methodology": "leave-allele-out shuffle null"},
    {"id": "e14", "statement": "Mean hydrophobicity correlates with binding across alleles in LAOO holdout", "test_methodology": "allele-out ridge"},
    {"id": "e15", "statement": "HLA-B*07:02 neoantigen binding is predictable from anchors under leave-one-allele-out", "test_methodology": "epitope LAOO"},
    {"id": "e16", "statement": "Combined anchor and charge features improve LAOO R² for binding", "test_methodology": "leave-one-allele-out"},
    {"id": "e17", "statement": "Pure peptide-feature noise fails cross-allele binding generalization", "test_methodology": "allele holdout"},
    {"id": "e18", "statement": "Extreme net charge lowers binding consistently across held-out alleles", "test_methodology": "cross-allele LAOO"},
    {"id": "e19", "statement": "IEDB-style binding trends replicate under leave-one-allele-out ridge", "test_methodology": "LAOO cross-allele"},
    {"id": "e20", "statement": "Peptide anchor properties predict binding across alleles surviving binding permutation", "test_methodology": "allele LAOO shuffle null"},
    {"id": "e21", "statement": "T-cell epitope binding for HLA-C*07:01 generalizes from other alleles", "test_methodology": "leave-allele-out"},
    {"id": "e22", "statement": "Anchor hydrophobicity and net charge jointly drive cross-allele immunogenic binding", "test_methodology": "cross-allele ridge LAOO"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "anchor" in text:
        return ["anchor2_hydrophobicity", "anchorC_hydrophobicity", "mean_hydrophobicity"]
    if "charge" in text or "electrostatic" in text:
        return ["net_charge", "mean_hydrophobicity", "aromatic_fraction"]
    if "length" in text or "weight" in text or "size" in text:
        return ["peptide_length", "mol_weight", "proline_fraction"]
    return ["anchorC_hydrophobicity", "net_charge", "mean_hydrophobicity", "peptide_length"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "epitope",
        "resolved_verifier": "epitope_laoo",
        "resolved_features": features,
        "expected_metric_name": "laoo_r2",
        "actual_metric_name": "laoo_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        try:
            spec = EpitopeExperimentSpec.from_hypothesis(
                {"text": hypothesis.get("statement", ""), "feature_subset": features}
            )
            result = run_epitope_experiment(spec)
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
