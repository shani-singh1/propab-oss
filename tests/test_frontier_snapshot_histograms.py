"""Tests for frontier snapshot theme/claim histograms (fixes.md P4.2)."""

from propab.claim_types import CLAIM_FINITE_VERIFIED
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from services.orchestrator.campaign_diagnostics import frontier_snapshot


def test_frontier_snapshot_includes_histograms() -> None:
    tree = HypothesisTree()
    tree.nodes["a"] = HypothesisNode(
        id="a", text="mod 4 scan", parent_id=None, depth=0, verdict="confirmed",
        theme_id="residue_class", claim_type=CLAIM_FINITE_VERIFIED,
    )
    tree.nodes["b"] = HypothesisNode(
        id="b", text="mod 4 again", parent_id=None, depth=0, verdict="pending",
        theme_id="residue_class",
    )
    tree.frontier = ["b"]
    snap = frontier_snapshot(tree)
    assert snap["theme_histogram"]["residue_class"] == 2
    assert snap["claim_histogram"][CLAIM_FINITE_VERIFIED] == 1
    assert snap["tested"] == 1
    assert snap["pending"] == 1
