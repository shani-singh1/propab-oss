#!/usr/bin/env python3
"""Ensure graphs:3h has a CANDIDATE with V2 entropy-dynamics predictions before residual batch."""
from __future__ import annotations

import json
import sys

from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_record import PolicyStatus
from propab.policy_store import PolicyStore
from services.orchestrator.policy_analyst import propose_policy_narrative_sync
from propab.knowledge_graph import KnowledgeGraph
from propab.meta_science import MetaScienceLedger


def main() -> int:
    store = PolicyStore.load()
    graph = KnowledgeGraph.load()
    meta = MetaScienceLedger.load()
    fitness = PolicyFitnessLedger.load()
    bb, db = "3h", "graphs"

    parent = store.accepted_policy(domain_bucket=db, budget_bucket=bb)
    latest = store.latest_candidate(domain_bucket=db, budget_bucket=bb)

    if latest and latest.predicted_effects.uses_entropy_dynamics():
        print(json.dumps({
            "action": "none",
            "candidate_id": latest.id,
            "predicted": latest.predicted_effects.to_dict(),
        }, indent=2))
        return 0

    rationale, predicted, fals, params = propose_policy_narrative_sync(
        parent=parent,
        graph=graph,
        meta=meta,
        budget_bucket=bb,
        domain_bucket=db,
        campaign_metrics={"closure_ratio": 0.1},
        fitness=fitness,
    )
    if latest and latest.status == PolicyStatus.CANDIDATE:
        latest.predicted_effects = predicted
        latest.rationale = rationale[:2000]
        latest.falsification_conditions = fals
        latest.boosts = dict(params.get("boosts") or latest.boosts)
        latest.penalties = dict(params.get("penalties") or latest.penalties)
        latest.blocked_failures = list(params.get("blocked_failures") or latest.blocked_failures)
        store.save()
        print(json.dumps({
            "action": "upgraded_existing",
            "candidate_id": latest.id,
            "predicted": predicted.to_dict(),
        }, indent=2))
        return 0

    cand = store.add_candidate(
        parent=parent,
        params=params,
        rationale=rationale,
        predicted=predicted,
        falsification=fals,
    )
    store.save()
    print(json.dumps({
        "action": "created",
        "candidate_id": cand.id,
        "predicted": predicted.to_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
