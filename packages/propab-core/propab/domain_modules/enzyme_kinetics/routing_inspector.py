"""Routing inspector for enzyme kinetics hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.enzyme_kinetics.adapter import KNOWN_FEATURES
from propab.domain_modules.enzyme_kinetics.verifier import run_enzyme_experiment
from propab.domain_modules.enzyme_kinetics.adapter import EnzymeExperimentSpec

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "ek01", "statement": "log_kcat is predictable from molecular weight and sequence length across EC classes under LOFO", "test_methodology": "leave-one-EC-class-out ridge"},
    {"id": "ek02", "statement": "Oxidoreductases (EC1) show higher kcat than hydrolases under family holdout", "test_methodology": "EC-class LOFO"},
    {"id": "ek03", "statement": "GRAVY hydropathy predicts log_kcat across enzyme families", "test_methodology": "cross-EC LOFO"},
    {"id": "ek04", "statement": "Sequence length alone predicts kcat across EC classes", "test_methodology": "leave-family-out verification"},
    {"id": "ek05", "statement": "Charged-residue fraction and molecular weight jointly predict kcat under LOFO", "test_methodology": "EC LOFO ridge"},
    {"id": "ek06", "statement": "Transferases show cross-EC generalization for kcat prediction", "test_methodology": "leave-one-EC-class-out"},
    {"id": "ek07", "statement": "Hydrolases fail cross-EC kcat generalization under LOFO", "test_methodology": "enzyme family holdout"},
    {"id": "ek08", "statement": "Aromatic-residue fraction correlates with kcat across EC classes in LOFO holdout", "test_methodology": "BRENDA subset LOFO"},
    {"id": "ek09", "statement": "Catalytic turnover kcat exceeds label-shuffle null p95 across EC families", "test_methodology": "LOFO with EC shuffle null"},
    {"id": "ek10", "statement": "EC3 hydrolases show positive LOFO R² for kcat from biophysical features", "test_methodology": "leave-EC-out ridge"},
    {"id": "ek11", "statement": "Molecular weight predicts kcat better than sequence length cross-EC", "test_methodology": "LOFO comparison"},
    {"id": "ek12", "statement": "Hydropathy composition features predict kcat under held-out EC class", "test_methodology": "cross-family LOFO"},
    {"id": "ek13", "statement": "Lyases (EC4) kcat is predictable from composition features under LOFO", "test_methodology": "enzyme LOFO verification"},
    {"id": "ek14", "statement": "Isomerases show no cross-EC kcat signal under family holdout", "test_methodology": "LOFO EC-class"},
    {"id": "ek15", "statement": "Combined molecular weight and hydropathy improve LOFO R² for kcat", "test_methodology": "leave-one-EC-out"},
    {"id": "ek16", "statement": "BRENDA subset enzymes: kcat generalizes across EC1 and EC2 under LOFO", "test_methodology": "ridge LOFO"},
    {"id": "ek17", "statement": "Enzyme kcat prediction survives EC-label permutation at p<0.05", "test_methodology": "LOFO shuffle null"},
    {"id": "ek18", "statement": "log_kcat band holds for oxidoreductases under LOFO", "test_methodology": "EC-class holdout"},
    {"id": "ek19", "statement": "Cross-EC kcat prediction from aromatic fraction alone fails LOFO generalization", "test_methodology": "leave-family-out"},
    {"id": "ek20", "statement": "UniProt-scale kcat trends replicate under leave-one-EC-class-out ridge", "test_methodology": "LOFO cross-EC"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "aromatic" in text or "hydrophob" in text or "composition" in text:
        return ["frac_aromatic", "frac_hydrophobic", "frac_charged", "molecular_weight"]
    if "hydropath" in text or "gravy" in text or "thermal" in text or "temperature" in text:
        return ["gravy_hydropathy", "frac_charged", "molecular_weight"]
    if "sequence" in text or "length" in text:
        return ["sequence_length", "molecular_weight", "frac_charged"]
    return ["molecular_weight", "sequence_length", "frac_charged", "frac_aromatic"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "enzyme_kinetics",
        "resolved_verifier": "enzyme_lofo",
        "resolved_features": features,
        "expected_metric_name": "lofo_r2",
        "actual_metric_name": "lofo_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        try:
            spec = EnzymeExperimentSpec.from_hypothesis({"text": hypothesis.get("statement", ""), "feature_subset": features})
            result = run_enzyme_experiment(spec)
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
