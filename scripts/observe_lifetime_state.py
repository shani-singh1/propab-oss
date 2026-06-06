#!/usr/bin/env python3
"""Print lifetime knowledge, search policy, and meta-science learning curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from propab.knowledge_graph import KnowledgeGraph, knowledge_store_path
from propab.meta_science import MetaScienceLedger, meta_store_path
from propab.policy_fitness_ledger import PolicyFitnessLedger, fitness_ledger_path
from propab.policy_store import PolicyStore, policy_store_path
from propab.search_policy import SearchPolicy, policy_store_path as legacy_policy_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    graph = KnowledgeGraph.load()
    store = PolicyStore.load()
    meta = MetaScienceLedger.load()
    fitness = PolicyFitnessLedger.load()
    legacy = SearchPolicy.load()

    accepted = {
        k: store.get_policy(pid).to_dict() if store.get_policy(pid) else None
        for k, pid in store.accepted.items()
    }
    candidates = [
        p.to_dict() for p in store.policies.values() if p.status.value == "CANDIDATE"
    ][-3:]
    rejected = [
        store.get_policy(pid).to_dict()
        for pid in store.rejected_ids[-5:]
        if store.get_policy(pid)
    ]

    summary = {
        "store_paths": {
            "knowledge": str(knowledge_store_path()),
            "policy_store": str(policy_store_path()),
            "legacy_policy": str(legacy_policy_path()),
            "meta": str(meta_store_path()),
            "fitness": str(fitness_ledger_path()),
        },
        "campaigns_ingested": graph.campaign_ids,
        "claims": len(graph.claims),
        "failures": len(graph.failures),
        "theories": {k: v.name for k, v in graph.theories.items()},
        "accepted_policies": accepted,
        "recent_candidates": candidates,
        "rejected_history": rejected,
        "fitness_records": len(fitness.records),
        "learning_curve_graphs_3h": meta.learning_curve(
            budget_bucket="3h", domain_bucket="graphs",
        ),
        "observations": [o.to_dict() for o in meta.observations[-5:]],
        "legacy_policy": legacy.to_dict(),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("=== Lifetime knowledge ===")
    print(f"Store: {knowledge_store_path()}")
    print(f"Campaigns: {len(graph.campaign_ids)}")
    print(f"Claims: {len(graph.claims)}  Failures: {len(graph.failures)}  Theories: {len(graph.theories)}")
    print("\n=== Policy store ===")
    print(f"Accepted buckets: {list(store.accepted.keys())}")
    print(f"Candidates: {len(candidates)}  Rejected: {len(store.rejected_ids)}")
    print(f"Fitness evaluations: {len(fitness.records)}")
    for key, pid in store.accepted.items():
        rec = store.get_policy(pid)
        if rec:
            print(f"  [{key}] gen={rec.generation} boost={rec.boosts}")
    if meta.observations:
        print("\n=== Meta-science (last 3 campaigns) ===")
        for o in meta.observations[-3:]:
            rate = o.confirmed / max(1, o.tested)
            print(
                f"  {o.campaign_id[:8]}… closure={o.closure_ratio:.2%} "
                f"confirmed_rate={rate:.2%} entropy={o.theme_entropy:.2f}"
            )
        curve = meta.learning_curve()
        if len(curve["closure_ratio"]) >= 2:
            print(f"Closure trend: {curve['closure_ratio']}")
    else:
        print("\nNo campaign observations recorded yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
