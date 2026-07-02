"""Tests for finding audit and competing mechanisms (fixes.md)."""
from __future__ import annotations

from propab.anomaly_engine.anomaly_objects import AnomalyObject
from propab.anomaly_engine.competing_mechanisms import build_competing_sets
from propab.finding_audit import (
    DISPOSITION_ARCHIVE,
    DISPOSITION_GOLD,
    DISPOSITION_REJECT,
    audit_confirmed_findings,
    classify_finding,
)


def test_null_group_specific_rejected():
    a = classify_finding(
        hypothesis_id="1",
        text="Null hypothesis: No falsifiable pattern...",
        evidence_summary=(
            'evidence={"metric_value": -0.44, "effect_size": 0.96, "verdict_reason": "null: group-specific"}; '
            "features=['sp_motif_found', 'native_net_charge']"
        ),
    )
    assert a.disposition == DISPOSITION_REJECT
    assert a.is_null


def test_thermal_discrimination_archived():
    a = classify_finding(
        hypothesis_id="2",
        text="Thermal stability features (t70_raw, t75_raw) provide a more robust cross-family signal than geometry",
        evidence_summary=(
            'evidence={"metric_value": -0.25, "effect_size": 0.93}; '
            "features=['t70_raw', 't75_raw', 'triad_best_rmsd']"
        ),
    )
    assert a.primary_family == "mixed"
    assert a.disposition == DISPOSITION_ARCHIVE


def test_positive_lofo_is_gold():
    a = classify_finding(
        hypothesis_id="3",
        text="Thermal beats foldseek under LOFO",
        evidence_summary=(
            'evidence={"metric_value": 0.063, "effect_size": 0.54}; '
            "features=['t70_raw', 't75_raw', 'foldseek_best_TM']"
        ),
    )
    assert a.disposition == DISPOSITION_GOLD


def test_competing_sets_from_anomalies():
    anomalies = [
        AnomalyObject(
            feature_subset=["t70_raw", "t75_raw"],
            metric_name="leave_one_family_out_r2",
            expected_score=0.2,
            observed_score=-0.12,
            surprise_score=0.05,
            anomaly_type="family_violation",
            affected_families=[],
            neighboring_subsets=[],
            metadata={"bucket": "survivor", "lofo_gap": 0.3},
        ),
        AnomalyObject(
            feature_subset=["foldseek_best_TM"],
            metric_name="leave_one_family_out_r2",
            expected_score=0.2,
            observed_score=-0.45,
            surprise_score=-0.5,
            anomaly_type="prediction_failure",
            affected_families=[],
            neighboring_subsets=[],
            metadata={"bucket": "collapse", "lofo_gap": 0.9, "within_family_r2": 0.6},
        ),
    ]
    sets = build_competing_sets(anomalies, max_sets=2)
    assert len(sets) == 2
    assert len(sets[0].mechanisms) >= 2
    assert sets[0].mechanisms[0].explanation != sets[0].mechanisms[1].explanation


def test_audit_summary_fake_diversity():
    rows = [
        {"id": "1", "text": "Thermal only t70_raw", "evidence_summary": 'evidence={"metric_value": -0.1}; features=[\'t70_raw\']'},
        {"id": "2", "text": "Thermal t75_raw cross-group", "evidence_summary": 'evidence={"metric_value": -0.2}; features=[\'t75_raw\']'},
    ]
    report = audit_confirmed_findings(rows)
    assert report["fake_diversity"] is True
