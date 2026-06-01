"""Tests for frontier information-gain scoring and expansion merit gate (fixes.md P1)."""

from propab.hypothesis_tree import HypothesisNode, HypothesisTree


def test_information_gain_prefers_higher_relevance() -> None:
    tree = HypothesisTree()
    tree.set_scoring_context("Egyptian fractions for odd n", ["unit fraction representations"])
    low = HypothesisNode(
        id="a", text="generic intervention baseline", parent_id=None, depth=0,
        verdict="pending", theme_id="general", question_relevance_score=0.1,
    )
    high = HypothesisNode(
        id="b", text="odd n unit fraction egyptian decomposition mod 4", parent_id=None, depth=0,
        verdict="pending", theme_id="unit_fraction", question_relevance_score=0.8,
    )
    tree.nodes[low.id] = low
    tree.nodes[high.id] = high
    tree.frontier = [low.id, high.id]
    batch = tree.next_batch(1)
    assert batch[0].id == "b"
    assert high.frontier_score is not None
    assert high.frontier_score > (low.frontier_score or 0)


def test_theme_saturation_penalizes_oversaturated_theme() -> None:
    tree = HypothesisTree()
    tree.set_scoring_context("residue classes mod 4", [], theme_saturation_penalty=0.2)
    for i in range(5):
        nid = f"s{i}"
        tree.nodes[nid] = HypothesisNode(
            id=nid, text=f"mod 4 variant {i}", parent_id=None, depth=1,
            verdict="confirmed", theme_id="residue_class", question_relevance_score=0.6,
        )
    fresh = HypothesisNode(
        id="new", text="parametric family identity", parent_id=None, depth=0,
        verdict="pending", theme_id="parametric_family", question_relevance_score=0.6,
    )
    saturated = HypothesisNode(
        id="sat", text="another mod 4 claim", parent_id=None, depth=0,
        verdict="pending", theme_id="residue_class", question_relevance_score=0.6,
    )
    tree.nodes[fresh.id] = fresh
    tree.nodes[saturated.id] = saturated
    assert tree._information_gain_score(fresh) > tree._information_gain_score(saturated)


def test_expansion_merit_gate() -> None:
    tree = HypothesisTree()
    node = HypothesisNode(
        id="x", text="off topic", parent_id=None, depth=0, verdict="confirmed",
        question_relevance_score=0.05,
    )
    tree.nodes[node.id] = node
    ok, reason = tree.expansion_passes_merit_gate("x", novelty_min=0.25, info_gain_min=0.99)
    assert not ok
    assert "merit_gate_failed" in reason
