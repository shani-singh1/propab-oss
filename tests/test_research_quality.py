"""Tests for post network-resilience research quality controls (fixes.md)."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.research_quality import (
    NODE_ROLE_CONTROL,
    NODE_ROLE_DISCOVERY,
    build_canonical_finding,
    compute_evidence_hash,
    extract_theme_vector,
    infer_node_role,
    is_control_hypothesis,
    is_discovery_node,
    paper_eligible_finding,
    classify_inconclusive_reason,
    should_retest_inconclusive,
)


def test_control_hypothesis_detection():
    assert is_control_hypothesis("Null hypothesis: no effect beyond noise")
    assert infer_node_role("Null hypothesis: no effect") == NODE_ROLE_CONTROL


def test_theme_vector_not_all_general():
    primary, secondary, conf = extract_theme_vector(
        "Targeted removal of highest-degree nodes increases spectral gap in Barabási–Albert graphs"
    )
    assert primary != "general"
    assert conf >= 0.5
    assert "targeted_removal" in [primary, *secondary] or "scale_free" in [primary, *secondary]


def test_evidence_hash_dedup():
    ev = {"verified_true_steps": 3, "p_value": 0.01, "verdict_reason": "pattern holds"}
    h = compute_evidence_hash(ev)
    tree = HypothesisTree()
    assert tree.register_evidence_hash(h) is True
    assert tree.register_evidence_hash(h) is False


def test_control_confirm_becomes_inconclusive():
    tree = HypothesisTree()
    node = HypothesisNode(
        id="c1",
        text="Null hypothesis: no falsifiable pattern in random graphs",
        parent_id=None,
        depth=0,
        node_role=NODE_ROLE_CONTROL,
    )
    tree.nodes[node.id] = node
    tree.update_node(node.id, "confirmed", 0.95, "evidence={};")
    assert tree.nodes[node.id].verdict == "inconclusive"
    assert node.id not in tree.confirmed


def test_discovery_node_expansion_blocked_for_control():
    tree = HypothesisTree()
    node = HypothesisNode(
        id="c2",
        text="Null hypothesis: no statistically significant effect beyond noise",
        parent_id=None,
        depth=0,
        verdict="confirmed",
        node_role=NODE_ROLE_CONTROL,
    )
    tree.nodes[node.id] = node
    assert tree.build_expand_prompt(node.id) is None


def test_paper_eligible_filters_control():
    finding = build_canonical_finding(
        claim_id="x",
        claim="test",
        claim_type="STATISTICAL",
        replication_level="T1",
        confidence=0.9,
        verification_method="statistical",
        primary_theme="general",
        secondary_themes=[],
        mechanism_obj=None,
        evidence_hash="abc",
        verification_hash="def",
        node_role=NODE_ROLE_CONTROL,
        verdict="confirmed",
    )
    assert not paper_eligible_finding(finding)


def test_inconclusive_reason_populated():
    reason = classify_inconclusive_reason({"verdict_reason": "timeout after 120s"})
    assert reason == "code_timeout"


def test_retest_gate_requires_information_gain():
    low = HypothesisNode(id="n", text="x", parent_id=None, depth=0, frontier_score=0.1)
    high = HypothesisNode(id="m", text="y", parent_id=None, depth=0, frontier_score=0.8, question_relevance_score=0.7)
    assert not should_retest_inconclusive(low)
    assert should_retest_inconclusive(high)


def test_is_discovery_default():
    n = HypothesisNode(id="d", text="Spectral gap predicts fragmentation", parent_id=None, depth=0)
    assert is_discovery_node(n)
    assert infer_node_role(n.text) == NODE_ROLE_DISCOVERY
