"""Routing inspector for genomics hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.genomics.adapter import KNOWN_FEATURES
from propab.domain_modules.genomics.verifier import run_genomics_experiment
from propab.domain_modules.genomics.adapter import GenomicsExperimentSpec

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "g01", "statement": "Genes with low tissue specificity tau index show cross-tissue LOFO R² above 0.15", "test_methodology": "leave-tissue-out ridge regression on expression variance and mean expression"},
    {"id": "g02", "statement": "Housekeeping genes (high mean expression, low CV) generalize across held-out tissues", "test_methodology": "cross-tissue LOFO with tissue label shuffle null"},
    {"id": "g03", "statement": "Expression variance predicts mean expression across tissues under LOFO", "test_methodology": "leave-one-tissue-out"},
    {"id": "g04", "statement": "Tissue specificity tau index is predictable from expression variance in LOFO holdout", "test_methodology": "leave-tissue-out ridge"},
    {"id": "g05", "statement": "Cross-tissue conserved genes have CV across tissues below 0.5", "test_methodology": "LOFO verification with cv_across_tissues feature"},
    {"id": "g06", "statement": "Genes with high expression variance fail cross-tissue generalization", "test_methodology": "leave-tissue-out LOFO"},
    {"id": "g07", "statement": "Mean expression alone predicts tissue specificity tau under held-out tissue", "test_methodology": "cross-tissue LOFO"},
    {"id": "g08", "statement": "Combined expression variance and tau index improve LOFO R² over either alone", "test_methodology": "leave-one-tissue-out"},
    {"id": "g09", "statement": "Stress-response genes show partial cross-tissue conservation in GTEx subset", "test_methodology": "LOFO with tissue label shuffle"},
    {"id": "g10", "statement": "Metabolic genes retain expression correlation across tissues in LOFO", "test_methodology": "leave-tissue-out verification"},
    {"id": "g11", "statement": "Non-housekeeping genes with tau below 0.3 show LOFO R² above 0.1", "test_methodology": "cross-tissue holdout"},
    {"id": "g12", "statement": "Expression variance and CV across tissues jointly predict cross-tissue mean", "test_methodology": "LOFO ridge regression"},
    {"id": "g13", "statement": "Tissue-enriched genes (tau > 0.8) fail leave-tissue-out generalization", "test_methodology": "leave-one-tissue-out"},
    {"id": "g14", "statement": "Low tau index genes survive tissue-label permutation test at p<0.05", "test_methodology": "tissue label shuffle null LOFO"},
    {"id": "g15", "statement": "Cross-tissue gene expression patterns are not explained by tissue metadata alone", "test_methodology": "LOFO with label shuffle"},
    {"id": "g16", "statement": "Gene-level expression variance correlates with cross-tissue LOFO performance", "test_methodology": "leave-tissue-out"},
    {"id": "g17", "statement": "Constitutively expressed genes show label-shuffle null p below 0.05", "test_methodology": "cross-tissue LOFO tissue shuffle"},
    {"id": "g18", "statement": "CV across tissues predicts held-out tissue mean expression", "test_methodology": "leave-one-tissue-out ridge"},
    {"id": "g19", "statement": "GTEx subset genes with high mean expression generalize under LOFO", "test_methodology": "leave-tissue-out verification"},
    {"id": "g20", "statement": "Partial cross-tissue conservation is predicted by expression variance not tau alone", "test_methodology": "LOFO cross-tissue holdout"},
    {"id": "g21", "statement": "Cross-tissue eQTL-like patterns require LOFO R² above label-shuffle p95", "test_methodology": "leave-tissue-out with permutation null"},
    {"id": "g22", "statement": "Housekeeping function genes have tissue specificity tau below 0.2", "test_methodology": "cross-tissue LOFO on tau index target"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "tau" in text or "specificity" in text:
        return ["tissue_specificity_tau", "expression_variance"]
    if "housekeeping" in text or "constitutive" in text:
        return ["mean_expression", "cv_across_tissues"]
    if "cv" in text:
        return ["cv_across_tissues", "expression_variance"]
    return ["expression_variance", "mean_expression"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    claim = str(hypothesis.get("statement") or hypothesis.get("text") or "")
    methodology = str(hypothesis.get("test_methodology") or "")
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "genomics",
        "resolved_verifier": "genomics_lofo",
        "resolved_features": features,
        "invalid_features": invalid,
        "expected_metric_name": "lofo_r2",
        "actual_metric_name": "lofo_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
        "methodology": methodology,
    }
    if dry_run_experiment:
        try:
            spec = GenomicsExperimentSpec.from_hypothesis({"text": claim, "feature_subset": features})
            result = run_genomics_experiment(spec)
            out["dry_run_result_keys"] = list(result.keys())
            out["routing_ok"] = routing_ok and "lofo_r2" in result
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)
            out["routing_ok"] = False
    return out


def inspect_corpus(hypotheses: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    hyps = hypotheses if hypotheses is not None else ROUTING_CORPUS
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for hyp in hyps:
        result = inspect_routing(hyp)
        row = {"id": hyp.get("id"), "text_preview": (hyp.get("statement") or "")[:120], **result}
        rows.append(row)
        if not result.get("routing_ok"):
            mismatches.append(row)
    return {
        "total": len(rows),
        "routing_ok": sum(1 for r in rows if r.get("routing_ok")),
        "routing_mismatches": len(mismatches),
        "mismatch_rate": round(len(mismatches) / max(len(rows), 1), 3),
        "rows": rows,
        "mismatches": mismatches,
    }
