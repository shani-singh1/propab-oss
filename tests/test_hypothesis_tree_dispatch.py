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
