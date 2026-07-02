"""Composition tests for run_verdict_pipeline (fixes.md Task 0)."""
from __future__ import annotations

from propab.verdict_pipeline import run_verdict_pipeline


def test_ml_evidence_confirms():
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "verified_true_steps": 3,
        "verified_false_steps": 0,
        "metric_direction": "higher_is_better",
        "verdict_reason": "significance gate passed",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
    assert confidence >= 0.85


def test_deterministic_math_proof_confirms():
    evidence = {
        "verified_true_steps": 2,
        "verified_false_steps": 0,
        "deterministic": True,
        "verification_method": "symbolic_proof",
        "verdict_reason": "proof verified",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
    assert confidence >= 0.90


def test_lofo_evidence_with_null_stats_confirms():
    evidence = {
        "lofo_r2": 0.15,
        "label_shuffle_null_p95": 0.11,
        "label_shuffle_permutation_p": 0.02,
        "lofo_gap": 0.45,
        "family_leakage_confirmed": False,
        "verified_true_steps": 2,
        "p_value": 0.03,
        "metric_value": 0.15,
        "n_samples": 120,
        "n_families": 7,
        "methodology": "LOFO",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"


def test_generic_significance_only_goes_inconclusive_not_refuted():
    evidence = {
        "metric_value": 0.88,
        "p_value": 0.03,
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "n_metric_steps": 3,
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"
    assert "replication" in reason.lower() or "holdout" in reason.lower()


def test_statistical_confirmed_downgrades_without_holdout():
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"
    assert "holdout" in reason.lower()


def test_worker_path_integration_composition():
    """End-to-end through pipeline with worker-like campaign context."""
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, confidence, reason = run_verdict_pipeline(
        evidence,
        hypothesis={"text": "MNIST accuracy improves with deeper CNN"},
        campaign_context={
            "hyp_text": "MNIST accuracy improves with deeper CNN",
            "domain_bucket": "ml",
            "min_metric_steps": 2,
        },
    )
    assert verdict == "inconclusive"
    assert verdict != "refuted"
