#!/usr/bin/env python3
"""Print lifetime knowledge, search policy, and meta-science learning curve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from propab.knowledge_graph import KnowledgeGraph, knowledge_store_path
from propab.meta_science import MetaScienceLedger, meta_store_path
from propab.search_policy import SearchPolicy, policy_store_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    graph = KnowledgeGraph.load()
    policy = SearchPolicy.load()
    meta = MetaScienceLedger.load()

    summary = {
        "store_paths": {
            "knowledge": str(knowledge_store_path()),
            "policy": str(policy_store_path()),
            "meta": str(meta_store_path()),
        },
        "campaigns_ingested": graph.campaign_ids,
        "claims": len(graph.claims),
        "failures": len(graph.failures),
        "theories": {k: v.name for k, v in graph.theories.items()},
        "theme_success_rates": graph.theme_success_rates(),
        "policy": policy.to_dict(),
        "learning_curve": meta.learning_curve(),
        "observations": [o.to_dict() for o in meta.observations[-5:]],
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print("=== Lifetime knowledge ===")
    print(f"Store: {knowledge_store_path()}")
    print(f"Campaigns: {len(graph.campaign_ids)}")
    print(f"Claims: {len(graph.claims)}  Failures: {len(graph.failures)}  Theories: {len(graph.theories)}")
    if graph.theme_success_rates():
        print("Theme success rates:", graph.theme_success_rates())
    print("\n=== Search policy ===")
    print(f"Generation: {policy.generation}")
    print(f"Boost: {policy.theme_boost}")
    print(f"Penalty: {policy.theme_penalty}")
    print(f"Saturated: {policy.saturated_themes}")
    print(f"Blocked signatures: {policy.blocked_failure_signatures[:6]}")
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
