"""Tests for fixes.md post-contagion campaign quality layer."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.research_quality import (
    INCONCLUSIVE_CODE_TIMEOUT,
    INCONCLUSIVE_METRIC_AMBIGUOUS,
    INCONCLUSIVE_METRIC_MISSING,
    INCONCLUSIVE_REPLICATION_FAILED,
    build_canonical_finding,
    build_mechanism_object,
    build_refutation_mechanism,
    build_verification_escalation,
    classify_inconclusive_reason,
    compute_evidence_hash,
    compute_theme_entropy,
    estimate_closure_probability,
    extract_theme_vector,
    failure_signature_from_reason,
    infer_finding_links,
    is_valid_evidence_for_hash,
    retry_policy_for_signature,
)
from services.orchestrator.campaign_diagnostics import frontier_snapshot


def test_inconclusive_subtype_taxonomy():
    assert classify_inconclusive_reason({"verdict_reason": "no metric-bearing steps executed"}) == INCONCLUSIVE_METRIC_MISSING
    assert classify_inconclusive_reason({"verdict_reason": "significance gate passed but metric direction ambiguous"}) == INCONCLUSIVE_METRIC_AMBIGUOUS
    assert classify_inconclusive_reason(
        {"verdict_reason": "unreplicated (1 metric step; need >= 2 to confirm)", "n_metric_steps": 1}
    ) == INCONCLUSIVE_REPLICATION_FAILED
    assert classify_inconclusive_reason({}, failure_reason="sub_agent_wall_exceeded") == INCONCLUSIVE_CODE_TIMEOUT


def test_empty_evidence_not_hashed():
    assert not is_valid_evidence_for_hash({})
    assert compute_evidence_hash({}) is None
    assert compute_evidence_hash({"n_metric_steps": 0}) is None
    h = compute_evidence_hash({"n_metric_steps": 2, "metric_value": 0.5, "p_value": 0.01})
    assert h is not None


def test_empty_hashes_do_not_collide_in_dedup():
    tree = HypothesisTree()
    assert tree.register_evidence_hash(compute_evidence_hash({}) or "") is True
    assert tree.register_evidence_hash(compute_evidence_hash({}) or "") is True
    ev = {"n_metric_steps": 3, "metric_value": 0.9, "p_value": 0.001}
    h = compute_evidence_hash(ev)
    assert tree.register_evidence_hash(h) is True
    assert tree.register_evidence_hash(h) is False


def test_theme_vector_shrinks_general():
    primary, secondary, conf = extract_theme_vector(
        "SIS contagion speed on scale-free networks depends on spectral gap"
    )
    assert primary != "general"
    assert conf >= 0.5
    assert "spectral" in [primary, *secondary]


def test_mechanism_object_has_causal_fields():
    mobj = build_mechanism_object(
        claim="Low λ₂/λ₁ ratio collapses inter-community infection variance",
        mechanism=None,
        evidence={"p_value": 0.003, "metric_value": 1.0, "verdict_reason": "significance gate passed"},
    )
    assert mobj is not None
    assert mobj.get("cause")
    assert mobj.get("effect")
    assert mobj.get("conditions")
    assert isinstance(mobj.get("evidence"), list)


def test_refutation_mechanism():
    rm = build_refutation_mechanism(
        claim="Spectral norm does not predict spike timing",
        evidence={"verified_false_steps": 1},
        verdict_reason="deterministic counterexample found (verified=false)",
    )
    assert rm.get("effect")
    assert "counterexample" in str(rm.get("failure_modes")).lower() or rm.get("failure_modes")


def test_ledger_verdict_and_links():
    ledger = [
        build_canonical_finding(
            claim_id="a",
            claim="percolation extent grows log",
            claim_type="STATISTICAL",
            replication_level="T2",
            confidence=0.9,
            verification_method="statistical",
            primary_theme="percolation",
            secondary_themes=[],
            mechanism_obj=None,
            evidence_hash="abc",
            verification_hash="def",
            node_role="DISCOVERY",
            verdict="confirmed",
        )
    ]
    entry = build_canonical_finding(
        claim_id="b",
        claim="spectral gap predicts T50",
        claim_type="SYMBOLIC",
        replication_level="T2",
        confidence=0.95,
        verification_method="symbolic",
        primary_theme="spectral",
        secondary_themes=["percolation"],
        mechanism_obj=None,
        evidence_hash="ghi",
        verification_hash="jkl",
        node_role="DISCOVERY",
        verdict="confirmed",
        links=infer_finding_links(ledger, {"claim_id": "b", "primary_theme": "spectral", "verdict": "confirmed", "secondary_themes": ["percolation"]}),
    )
    assert entry["verdict"] == "confirmed"
    assert entry["verdict"] is not None
    assert entry.get("extends") or entry.get("supports")


def test_closure_probability_modulates_frontier():
    tree = HypothesisTree()
    parent = HypothesisNode(
        id="p", text="x", parent_id=None, depth=0, verdict="inconclusive",
        inconclusive_reason=INCONCLUSIVE_REPLICATION_FAILED,
    )
    child = HypothesisNode(
        id="c", text="spectral gap and percolation threshold in SIS contagion",
        parent_id="p", depth=1, verdict="pending", expansion_type="retest",
        question_relevance_score=0.7,
    )
    tree.nodes["p"] = parent
    tree.nodes["c"] = child
    tree.frontier = ["c"]
    score = tree._information_gain_score(child)
    closure = estimate_closure_probability(child, parent=parent)
    assert closure > 0.4
    assert score > 0


def test_failure_signature_retry_policy():
    sig = failure_signature_from_reason(INCONCLUSIVE_CODE_TIMEOUT)
    pol = retry_policy_for_signature(sig)
    assert pol.get("prefer_smaller_experiment")
    esc = build_verification_escalation(
        HypothesisNode(id="n", text="t", parent_id="p", depth=1, verdict="pending", expansion_type="retest"),
        parent=HypothesisNode(id="p", text="p", parent_id=None, depth=0, verdict="inconclusive", inconclusive_reason=INCONCLUSIVE_REPLICATION_FAILED),
    )
    assert esc.get("min_metric_steps", 2) >= 3


def test_frontier_snapshot_theme_entropy_and_closure():
    tree = HypothesisTree()
    for i, theme in enumerate(["spectral", "percolation", "spectral", "diffusion_dynamics"]):
        tree.nodes[f"n{i}"] = HypothesisNode(
            id=f"n{i}", text=f"hyp {theme}", parent_id=None, depth=0,
            verdict="confirmed" if i < 2 else "inconclusive",
            primary_theme=theme, replication_level="T2",
        )
    snap = frontier_snapshot(tree)
    assert snap["theme_entropy"] > 0
    assert snap["closure_ratio"] == 0.5
    assert "replication_health" in snap
    assert snap["general_theme_fraction"] == 0.0


def test_lineage_length():
    tree = HypothesisTree()
    tree.nodes["r"] = HypothesisNode(id="r", text="root", parent_id=None, depth=0)
    tree.nodes["c"] = HypothesisNode(id="c", text="child", parent_id="r", depth=1)
    assert tree.lineage_length("c") == 2
