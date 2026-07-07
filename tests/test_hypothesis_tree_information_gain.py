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


def test_convergence_prefers_deepening_confirmed_over_inconclusive_breadth() -> None:
    """A scope-narrowing child of a CONFIRMED finding must outrank a child of an
    inconclusive parent (convergence/exploit over breadth) — the fix for
    'generation increases, depth does not' (investigation report §3.3)."""
    tree = HypothesisTree()
    tree.set_scoring_context("crossing threshold for greedy ratio", [])
    confirmed_parent = HypothesisNode(
        id="pc", text="ratio crosses 0.60 near n=35000", parent_id=None, depth=0,
        verdict="confirmed", theme_id="threshold", question_relevance_score=0.6,
    )
    inconclusive_parent = HypothesisNode(
        id="pi", text="unrelated variance probe", parent_id=None, depth=0,
        verdict="inconclusive", theme_id="variance", question_relevance_score=0.6,
    )
    tree.nodes[confirmed_parent.id] = confirmed_parent
    tree.nodes[inconclusive_parent.id] = inconclusive_parent
    # Narrowing child of the confirmed finding (boundary refinement, scope delta).
    deepen = HypothesisNode(
        id="cc", text="does the crossing hold in [34000,36000] specifically", parent_id="pc",
        depth=1, verdict="pending", theme_id="threshold", question_relevance_score=0.6,
        expansion_type="boundary", scope_delta={"narrowed": "n range"},
    )
    # Child of the inconclusive parent (breadth).
    breadth = HypothesisNode(
        id="ci", text="another variance angle", parent_id="pi",
        depth=1, verdict="pending", theme_id="variance", question_relevance_score=0.6,
    )
    tree.nodes[deepen.id] = deepen
    tree.nodes[breadth.id] = breadth
    assert tree._information_gain_score(deepen) > tree._information_gain_score(breadth)


def test_confirmed_lineage_depth_metric() -> None:
    tree = HypothesisTree()
    assert tree.confirmed_lineage_depth() == 0.0  # no confirmed nodes yet
    root = HypothesisNode(id="r", text="root finding", parent_id=None, depth=0, verdict="confirmed")
    child = HypothesisNode(id="c", text="narrower finding", parent_id="r", depth=1, verdict="confirmed")
    tree.nodes[root.id] = root
    tree.nodes[child.id] = child
    # root has confirmed-ancestry depth 1, child has 2 → mean 1.5.
    assert tree.confirmed_lineage_depth() == 1.5


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
