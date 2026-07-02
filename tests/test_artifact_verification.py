"""Tests for artifact-aware verification (fixes.md P0–P6)."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    ARTIFACT_TOPOLOGY_DEPENDENCE,
    EvidenceContext,
    apply_artifact_gate_override,
    apply_two_stage_gate,
    audit_confirmed_rows,
    generate_artifact_models,
    merge_artifact_into_evidence,
    rank_artifact_models,
    run_adversarial_test,
    run_artifact_gate,
)


def test_generate_family_leakage_for_lofo():
    ctx = EvidenceContext(
        hypothesis_text="Thermal features predict RT activity across evolutionary families",
        evidence_generation_method="LOFO",
        n_samples=56,
        n_groups=7,
        group_column="rt_family",
    )
    models = generate_artifact_models(ctx)
    ids = {m.artifact_id for m in models}
    assert ARTIFACT_FAMILY_LEAKAGE in ids


def test_significance_only_for_contagion():
    ctx = EvidenceContext(
        hypothesis_text="k-shell index correlates with SIS outbreak in modular networks",
        evidence_generation_method="statistical_significance",
        n_samples=40,
        p_value=0.001,
        domain_bucket="graphs",
    )
    ranked = rank_artifact_models(generate_artifact_models(ctx), top_k=3)
    assert ranked[0].artifact_id in {
        ARTIFACT_FAMILY_LEAKAGE,
        ARTIFACT_TOPOLOGY_DEPENDENCE,
        ARTIFACT_SIGNIFICANCE_ONLY,
    }


def test_label_shuffle_survives_when_lofo_above_null():
    ctx = EvidenceContext(
        hypothesis_text="Cross-family thermal signal",
        evidence_generation_method="LOFO",
        n_samples=120,
        n_groups=7,
        lofo_r2=0.12,
        lofo_gap=0.2,
    )
    exp = {
        "mean_r2": 0.12,
        "lofo_gap": 0.2,
        "label_shuffle_null_p95": 0.05,
        "label_shuffle_permutation_p": 0.02,
    }
    gate = run_artifact_gate(ctx, exp)
    assert gate.top_artifact_survived is True
    assert gate.verdict == "confirmed"


def test_label_shuffle_refutes_negative_lofo():
    ctx = EvidenceContext(
        hypothesis_text="Geometry predicts activity across families",
        evidence_generation_method="LOFO",
        n_samples=56,
        n_groups=7,
        lofo_r2=-0.45,
        lofo_gap=1.1,
    )
    exp = {
        "mean_r2": -0.45,
        "lofo_gap": 1.1,
        "label_shuffle_null_p95": 0.25,
        "label_shuffle_permutation_p": 0.99,
    }
    gate = run_artifact_gate(ctx, exp)
    assert gate.top_artifact_survived is False
    assert gate.verdict == "refuted"


def test_significance_only_downgrades_confirmed():
    ctx = EvidenceContext(
        hypothesis_text="Clustering coefficient drives contagion speed on scale-free networks",
        p_value=0.003,
        n_samples=35,
        domain_bucket="graphs",
    )
    verdict, reason, conf, gate = apply_artifact_gate_override(
        "confirmed", "significance gate passed", 0.9, ctx,
    )
    assert gate is not None
    assert verdict in {"refuted", "inconclusive"}
    assert verdict != "confirmed"


def test_merge_artifact_into_evidence():
    ctx = EvidenceContext(hypothesis_text="test", p_value=0.5, n_samples=10)
    gate = run_artifact_gate(ctx)
    ev = merge_artifact_into_evidence({"metric_value": 0.5}, gate)
    assert "artifact_gate" in ev
    assert ev.get("top_artifact") is not None


def test_batch_audit_contagion_shape():
    rows = [
        {
            "id": "a",
            "text": "In modular networks k-shell correlates with SIS outbreak extent",
            "evidence_summary": (
                'evidence={"p_value": 0.001, "metric_value": 0.33, "n_metric_steps": 2}; '
                "significance={}"
            ),
        }
    ]
    report = audit_confirmed_rows(rows)
    assert report["n_audited"] == 1
    assert "artifact_failure_distribution" in report
    assert report["n_confirmed_under_artifact_gate"] == 0


def test_network_artifact_vocabulary_owned_by_plugin():
    """Core must not hardcode network keywords — they live on the domain plugin."""
    from propab.domain_modules.registry import get_domain_plugin

    plugin = get_domain_plugin("network_diffusion")
    assert plugin is not None
    markers = set(plugin.artifact_question_markers)
    assert {"contagion", "topology", "modular", "k-shell"} <= markers


def test_topology_artifact_detected_via_plugin_markers_without_domain_bucket():
    """Marker-based network detection (no domain_bucket) still yields the topology artifact."""
    ctx = EvidenceContext(
        hypothesis_text="k-shell index drives SIS contagion extent on scale-free networks",
        evidence_generation_method="statistical_significance",
        n_samples=40,
        p_value=0.001,
    )
    ids = {m.artifact_id for m in generate_artifact_models(ctx)}
    assert ARTIFACT_TOPOLOGY_DEPENDENCE in ids
