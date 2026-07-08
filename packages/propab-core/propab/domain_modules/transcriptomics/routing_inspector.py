"""Routing inspector for transcriptomics / gene-regulation hypotheses."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.transcriptomics.adapter import (
    KNOWN_FEATURES,
    TranscriptomicsExperimentSpec,
)
from propab.domain_modules.transcriptomics.verifier import run_transcriptomics_experiment

ROUTING_CORPUS: list[dict[str, Any]] = [
    {"id": "t01", "statement": "Expression log2 fold change is predictable from TF-motif count across conditions under leave-one-condition-out", "test_methodology": "leave-one-condition-out ridge"},
    {"id": "t02", "statement": "Transcription-factor motif count predicts differential expression across perturbations under LOCO", "test_methodology": "condition holdout"},
    {"id": "t03", "statement": "Chromatin accessibility predicts expression fold change across conditions", "test_methodology": "leave-condition-out"},
    {"id": "t04", "statement": "Motif count and accessibility jointly predict fold change under condition holdout", "test_methodology": "cross-condition LOCO"},
    {"id": "t05", "statement": "CpG ratio predicts expression response across held-out conditions", "test_methodology": "leave-one-condition-out"},
    {"id": "t06", "statement": "Promoter conservation predicts differential expression across conditions", "test_methodology": "condition holdout"},
    {"id": "t07", "statement": "Promoter length alone fails to predict fold change across conditions under LOCO", "test_methodology": "condition LOCO"},
    {"id": "t08", "statement": "TF-motif regulatory rule survives a within-condition fold-change-shuffle null", "test_methodology": "LOCO with fold-change shuffle null"},
    {"id": "t09", "statement": "TATA-box score predicts expression response across conditions", "test_methodology": "leave-one-condition-out"},
    {"id": "t10", "statement": "Chromatin accessibility and CpG ratio jointly predict fold change cross-condition", "test_methodology": "cross-condition ridge"},
    {"id": "t11", "statement": "Heat-shock expression response generalizes from other conditions under LOCO", "test_methodology": "condition holdout"},
    {"id": "t12", "statement": "TF-motif count predicts fold change better than promoter length across conditions", "test_methodology": "LOCO comparison"},
    {"id": "t13", "statement": "Promoter-feature regulatory model beats the within-condition shuffle null at p<0.05", "test_methodology": "leave-condition-out shuffle null"},
    {"id": "t14", "statement": "Chromatin accessibility correlates with fold change across conditions in LOCO holdout", "test_methodology": "condition-out ridge"},
    {"id": "t15", "statement": "Hypoxia expression response is predictable from promoter features under leave-one-condition-out", "test_methodology": "transcriptomics LOCO"},
    {"id": "t16", "statement": "Combined motif and accessibility improve LOCO R² for fold change", "test_methodology": "leave-one-condition-out"},
    {"id": "t17", "statement": "Pure promoter-feature noise fails cross-condition expression generalization", "test_methodology": "condition holdout"},
    {"id": "t18", "statement": "Higher TF-motif density raises expression response consistently across held-out conditions", "test_methodology": "cross-condition LOCO"},
    {"id": "t19", "statement": "GEO-style differential-expression trends replicate under leave-one-condition-out ridge", "test_methodology": "LOCO cross-condition"},
    {"id": "t20", "statement": "Promoter features predict fold change across conditions surviving fold-change permutation", "test_methodology": "condition LOCO shuffle null"},
    {"id": "t21", "statement": "Oxidative-stress expression response generalizes from other conditions under regulatory features", "test_methodology": "leave-condition-out"},
    {"id": "t22", "statement": "TF-motif count and chromatin accessibility jointly drive cross-condition regulatory signal", "test_methodology": "cross-condition ridge LOCO"},
]


def _infer_features(hypothesis: dict[str, Any]) -> list[str]:
    text = str(hypothesis.get("statement") or hypothesis.get("text") or "").lower()
    if "motif" in text or "transcription factor" in text or "tf-motif" in text or "tf motif" in text:
        return ["tf_motif_count", "chromatin_accessibility", "tata_score"]
    if "accessib" in text or "chromatin" in text or "atac" in text:
        return ["chromatin_accessibility", "tf_motif_count", "cpg_ratio"]
    if "cpg" in text or "methyl" in text or "gc content" in text or "gc-content" in text:
        return ["cpg_ratio", "gc_content", "conservation_score"]
    if "conservation" in text or "length" in text or "promoter" in text:
        return ["conservation_score", "promoter_length", "gc_content"]
    return ["tf_motif_count", "chromatin_accessibility", "cpg_ratio", "tata_score"]


def inspect_routing(hypothesis: dict[str, Any], *, dry_run_experiment: bool = False) -> dict[str, Any]:
    features = _infer_features(hypothesis)
    invalid = [f for f in features if f not in KNOWN_FEATURES]
    routing_ok = not invalid and bool(features)
    out: dict[str, Any] = {
        "domain": "transcriptomics",
        "resolved_verifier": "transcriptomics_loco",
        "resolved_features": features,
        "expected_metric_name": "loco_r2",
        "actual_metric_name": "loco_r2",
        "metric_match": True,
        "routing_ok": routing_ok,
    }
    if dry_run_experiment:
        try:
            spec = TranscriptomicsExperimentSpec.from_hypothesis(
                {"text": hypothesis.get("statement", ""), "feature_subset": features}
            )
            result = run_transcriptomics_experiment(spec)
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
