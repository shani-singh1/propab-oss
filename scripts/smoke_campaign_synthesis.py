#!/usr/bin/env python3
"""Smoke-test campaign synthesis pass with mock LLM (no API key)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.belief_state import CampaignBeliefState
from propab.campaign_synthesis import run_campaign_synthesis_pass
from propab.hypothesis_tree import HypothesisTree


class _MockLLM:
    async def call(self, *, prompt: str, purpose: str, session_id: str) -> str:
        return json.dumps({
            "beliefs": [{
                "statement": "k-shell claims fail without cross-topology holdout",
                "confidence": "weak",
                "status": "active",
                "supporting_nodes": [],
            }],
            "critical_experiment": {
                "title": "WS/ER holdout",
                "description": "Test k-shell vs degree on held-out topologies",
                "discriminates_between": ["topology dependence", "metric noise"],
            },
            "frontier_candidates": [{
                "id": "holdout_ws",
                "text": (
                    "k-shell beats degree only on modular graphs with WS holdout.\n"
                    "Population: modular N>1000\nDistribution: BA+SBM train\n"
                    "Claimed generalization: Watts-Strogatz holdout\n"
                    "Expected failure modes: single-family overfit\nOOD test: leave-one topology family"
                ),
                "test_methodology": "statistical_significance",
                "expansion_type": "diagnostic",
                "implements_critical_experiment": True,
            }],
            "recent_activity_summary": "First synthesis round after refuted seed",
            "direction_exhausted": False,
        })


async def main() -> None:
    tree = HypothesisTree()
    node = tree.add_seeds([{
        "text": (
            "k-shell index predicts SIS spread.\nPopulation: modular Q>0.4\n"
            "Distribution: SBM\nClaimed generalization: any modular\n"
            "Expected failure modes: degree-only baseline\nOOD test: ER holdout"
        ),
        "test_methodology": "sandbox_simulation",
    }], generation=0)[0]
    tree.update_node(node.id, "refuted", 0.85, 'evidence={"verdict_reason": "replication failed on WS"};')
    node.inconclusive_reason = "replication_failed"
    node.failure_signature = "no_cross_topology"

    state = CampaignBeliefState()
    added, metrics = await run_campaign_synthesis_pass(
        campaign_id="smoke-test",
        question="Which network metrics predict contagion speed?",
        tree=tree,
        belief_state=state,
        llm=_MockLLM(),
        generation=1,
    )
    print(json.dumps({
        "n_added": len(added),
        "metrics": metrics,
        "active_beliefs": [b.statement for b in state.active_beliefs],
        "frontier_size": len(tree.frontier),
        "dispatchable": tree.next_dispatch_candidate(frozenset()) is not None,
    }, indent=2))
    assert state.active_beliefs
    assert tree.next_dispatch_candidate(frozenset()) is not None
    print("OK — synthesis smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
