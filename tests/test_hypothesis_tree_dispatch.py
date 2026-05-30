"""Frontier dispatch selection for pipelined campaigns."""

from propab.hypothesis_tree import HypothesisNode, HypothesisTree


def _tree_with_frontier() -> HypothesisTree:
    t = HypothesisTree()
    a = HypothesisNode(id="a", text="hypothesis a " * 5, parent_id=None, depth=0, verdict="pending")
    b = HypothesisNode(id="b", text="hypothesis b " * 8, parent_id=None, depth=0, verdict="pending")
    t.nodes[a.id] = a
    t.nodes[b.id] = b
    t.frontier = ["a", "b"]
    return t


def test_next_dispatch_candidate_excludes_inflight() -> None:
    t = _tree_with_frontier()
    first = t.next_dispatch_candidate(frozenset())
    assert first is not None
    second = t.next_dispatch_candidate(frozenset({first.id}))
    assert second is not None
    assert second.id != first.id
    assert t.next_dispatch_candidate(frozenset({first.id, second.id})) is None


def test_next_dispatch_candidate_empty_frontier() -> None:
    t = HypothesisTree()
    assert t.next_dispatch_candidate(frozenset()) is None


def test_reseeding_does_not_clobber_confirmed_nodes() -> None:
    """Re-seeding with the same LLM ids must not overwrite an earlier confirmed node."""
    t = HypothesisTree()
    gen1 = t.add_seeds([{"id": "h4", "text": "hypothesis four " * 4}], generation=1)
    nid = gen1[0].id
    t.update_node(nid, "confirmed", 0.9, "beat baseline")
    assert t.summary()["confirmed_count"] == 1

    # Frontier empties → regenerate seeds; the LLM reuses id "h4".
    gen2 = t.add_seeds([{"id": "h4", "text": "a different hypothesis " * 4}], generation=2)
    # New seed must be a distinct node, leaving the confirmed one intact.
    assert gen2[0].id != nid
    assert t.nodes[nid].verdict == "confirmed"
    assert t.summary()["confirmed_count"] == 1
    assert len(t.nodes) == 2


def test_confirmed_list_consistent_after_reevaluation() -> None:
    """A confirmed node re-evaluated to inconclusive leaves the confirmed list."""
    t = HypothesisTree()
    node = t.add_seeds([{"id": "x", "text": "hypothesis x " * 4}], generation=1)[0]
    t.update_node(node.id, "confirmed", 0.9)
    assert t.summary()["confirmed_count"] == 1
    t.frontier.append(node.id)  # simulate a re-dispatch path
    t.update_node(node.id, "inconclusive", 0.0)
    assert node.id not in t.confirmed
    assert t.summary()["confirmed_count"] == 0
