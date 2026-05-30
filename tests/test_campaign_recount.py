"""Campaign outcome counts must reflect distinct evaluated tree nodes.

Regression for the paper/summary mismatch: re-dispatched nodes used to inflate
``total_confirmed`` via per-result ``+= 1`` increments, so the campaign summary
(e.g. "13 confirmed") disagreed with the hypothesis tree and the DB-derived paper
counts (e.g. "7 supported"). ``recount_from_tree`` makes them consistent.
"""
from __future__ import annotations

from propab.campaign import BreakthroughCriteria, ResearchCampaign


def _campaign() -> ResearchCampaign:
    return ResearchCampaign(
        id="11111111-1111-1111-1111-111111111111",
        question="Test question",
        breakthrough_criteria=BreakthroughCriteria.default_accuracy(),
    )


def test_recount_counts_distinct_nodes_not_results() -> None:
    c = _campaign()
    # Node ids are tree-owned (UUIDs); capture them from the returned nodes.
    n1, n2, n3 = c.hypothesis_tree.add_seeds(
        [{"id": "H1", "text": "a"}, {"id": "H2", "text": "b"}, {"id": "H3", "text": "c"}]
    )

    c.hypothesis_tree.update_node(n1.id, "confirmed", 0.9, "ev")
    c.hypothesis_tree.update_node(n2.id, "confirmed", 0.9, "ev")
    c.hypothesis_tree.update_node(n3.id, "refuted", 0.9, "ev")
    c.recount_from_tree()

    assert c.total_hypotheses == 3
    assert c.total_confirmed == 2


def test_recount_does_not_double_count_re_evaluated_node() -> None:
    c = _campaign()
    n1 = c.hypothesis_tree.add_seeds([{"id": "H1", "text": "a"}])[0]

    # Same node evaluated multiple times (re-dispatch) must count once.
    for _ in range(5):
        c.hypothesis_tree.update_node(n1.id, "confirmed", 0.9, "ev")
        c.recount_from_tree()

    assert c.total_hypotheses == 1
    assert c.total_confirmed == 1


def test_recount_ignores_pending_nodes() -> None:
    c = _campaign()
    n1, _n2 = c.hypothesis_tree.add_seeds([{"id": "H1", "text": "a"}, {"id": "H2", "text": "b"}])
    c.hypothesis_tree.update_node(n1.id, "inconclusive", 0.5, "ev")
    c.recount_from_tree()

    # The second seed is still pending → not counted as an evaluated hypothesis.
    assert c.total_hypotheses == 1
    assert c.total_confirmed == 0
