"""O3: honest "finalized with zero confirmed findings" signal.

A campaign that confirms nothing still finalizes with a normal-looking stop reason
and writes a paper, so a broken run presents as a completed one. The finalize path
layers an additive, queryable signal on top of the stop reason. These tests pin the
load-bearing decision — the confirmed-findings count source and the boolean derived
from it — without needing the full integration harness (DB/LLM/emitter).

The source of truth is ``campaign.total_confirmed`` (distinct confirmed *discovery*
nodes, controls excluded, kept fresh by ``recount_from_tree``) — the same number the
paper abstract and DB present, exposed via ``confirmed_findings_count``.
"""
from __future__ import annotations

from propab.campaign import BreakthroughCriteria, ResearchCampaign
from services.orchestrator.campaign_loop import confirmed_findings_count


def _campaign() -> ResearchCampaign:
    return ResearchCampaign(
        id="22222222-2222-2222-2222-222222222222",
        question="Test question",
        breakthrough_criteria=BreakthroughCriteria.default_accuracy(),
    )


def test_zero_confirmed_flags_finalized_without_findings() -> None:
    c = _campaign()
    n1, n2 = c.hypothesis_tree.add_seeds(
        [{"id": "H1", "text": "a"}, {"id": "H2", "text": "b"}]
    )
    # Nothing confirmed: one refuted, one inconclusive.
    c.hypothesis_tree.update_node(n1.id, "refuted", 0.9, "ev")
    c.hypothesis_tree.update_node(n2.id, "inconclusive", 0.4, "ev")
    c.recount_from_tree()

    count = confirmed_findings_count(c)
    finalized_without_findings = count == 0
    assert count == 0
    assert finalized_without_findings is True


def test_confirmed_finding_does_not_flag() -> None:
    c = _campaign()
    n1, n2 = c.hypothesis_tree.add_seeds(
        [{"id": "H1", "text": "a"}, {"id": "H2", "text": "b"}]
    )
    c.hypothesis_tree.update_node(n1.id, "confirmed", 0.9, "ev")
    c.hypothesis_tree.update_node(n2.id, "refuted", 0.9, "ev")
    c.recount_from_tree()

    count = confirmed_findings_count(c)
    assert count == 1
    assert (count == 0) is False


def test_all_inconclusive_flags_zero_findings() -> None:
    """Every node tested but nothing confirmed (all inconclusive) — the classic
    'infrastructure produced nothing' case — must flag zero findings even though
    the campaign will finalize with a normal stop reason and write a paper."""
    c = _campaign()
    n1, n2, n3 = c.hypothesis_tree.add_seeds(
        [{"id": "H1", "text": "a"}, {"id": "H2", "text": "b"}, {"id": "H3", "text": "c"}]
    )
    for nid in (n1.id, n2.id, n3.id):
        c.hypothesis_tree.update_node(nid, "inconclusive", 0.3, "weak signal")
    c.recount_from_tree()

    count = confirmed_findings_count(c)
    assert c.total_hypotheses == 3  # work happened
    assert count == 0              # ...but nothing was confirmed
    assert (count == 0) is True
