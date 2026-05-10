"""Contract tests: campaign synthesis payload matches hypothesis tree ledger (paper abstract sync)."""

from __future__ import annotations

import asyncio

from propab.campaign import ResearchCampaign
from propab.hypothesis_tree import HypothesisNode, HypothesisTree
from propab.paper_sections import generate_prose_sections
from services.orchestrator.campaign_loop import build_campaign_synthesis_payload


def _tree_with_counts(*, n_confirmed: int, n_refuted: int, n_inconclusive: int) -> HypothesisTree:
    tree = HypothesisTree()
    for i in range(n_confirmed):
        nid = f"c{i}"
        tree.nodes[nid] = HypothesisNode(id=nid, text=f"confirmed {i}", parent_id=None, depth=0, verdict="confirmed")
        tree.confirmed.append(nid)
    for j in range(n_refuted):
        nid = f"r{j}"
        tree.nodes[nid] = HypothesisNode(id=nid, text=f"refuted {j}", parent_id=None, depth=0, verdict="refuted")
    for k in range(n_inconclusive):
        nid = f"i{k}"
        tree.nodes[nid] = HypothesisNode(id=nid, text=f"inc {k}", parent_id=None, depth=0, verdict="inconclusive")
    return tree


def test_build_campaign_synthesis_matches_tree_verdicts() -> None:
    tree = _tree_with_counts(n_confirmed=22, n_refuted=2, n_inconclusive=3)
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-000000000099",
        question="Synthetic campaign for ledger contract.",
        hypothesis_tree=tree,
        total_hypotheses=27,
    )
    syn = build_campaign_synthesis_payload(campaign)
    assert syn["ledger"]["total_confirmed"] == 22
    assert syn["ledger"]["total_refuted"] == 2
    assert syn["ledger"]["total_inconclusive"] == 3
    assert syn["total_confirmed"] == 22
    assert syn["total_refuted"] == 2
    assert len(syn["ledger"]["confirmed"]) == 22


def test_campaign_synthesis_prose_abstract_matches_ledger_counts() -> None:
    tree = _tree_with_counts(n_confirmed=22, n_refuted=2, n_inconclusive=0)
    campaign = ResearchCampaign(
        id="00000000-0000-0000-0000-0000000000aa",
        question="Contract: abstract must list same confirmed count as ledger.",
        hypothesis_tree=tree,
    )
    syn = build_campaign_synthesis_payload(campaign)

    async def _run() -> None:
        out = await generate_prose_sections(
            llm=None,
            session_id=campaign.id,
            question=campaign.question,
            prior={"key_papers": []},
            synthesis=syn,
        )
        assert "confirmed=22" in out["abstract"]
        assert "refuted=2" in out["abstract"]

    asyncio.run(_run())
