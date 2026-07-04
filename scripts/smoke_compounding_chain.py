#!/usr/bin/env python3
"""Smoke test compounding chain fixes (agent1 step 3) — no campaign required."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.campaign_synthesis import apply_synthesis_to_frontier
from propab.hypothesis_tree import HypothesisTree
from propab.belief_state import CampaignBeliefState
from propab.numerical_seeds import classify_hypothesis_bucket, format_seeds_for_question
from propab.knowledge_graph import KnowledgeGraph
from propab.prompt_composer import compose_synthesis_prompt
from propab.synthesis_diversity import tree_problem_counts_from_nodes


def _sidon_with_boilerplate() -> str:
    return (
        "Population: Greedy Sidon for n in {20000, 30000, 40000}. "
        "Claim: F(n)/sqrt(n) falls below 0.62 at n=30000.\n"
        "Population: Integers {1,...,n} or vector space F_3^n\n"
        "Distribution: All admissible combinatorial structures in the domain"
    )


def main() -> int:
    failures: list[str] = []

    b = classify_hypothesis_bucket(_sidon_with_boilerplate(), "greedy Sidon sweep")
    if b["problem_type"] != "sidon":
        failures.append(f"bucket: expected sidon got {b['problem_type']}")

    tree = HypothesisTree()
    for i in range(8):
        tree.add_seeds(
            [{
                "text": _sidon_with_boilerplate().replace("30000", str(10000 + i * 5000)),
                "test_methodology": "greedy Sidon threshold sweep",
            }],
            generation=i,
        )
    cap = tree.add_seeds(
        [{"text": "Cap set CLP ratio in F_3^6", "test_methodology": "cap-set lookup"}],
        generation=9,
    )
    for n in cap:
        tree.update_node(n.id, "refuted", 0.9, '{"metric_name":"cap_set_clp_ratio"}')

    counts = tree_problem_counts_from_nodes(
        {nid: nd.to_dict() for nid, nd in tree.nodes.items()}
    )
    if counts.get("sidon", 0) < 8 or counts.get("cap_set", 0) != 1:
        failures.append(f"tree counts wrong: {counts}")

    state = CampaignBeliefState()
    parsed = {
        "beliefs": [],
        "frontier_candidates": [
            {
                "id": "struct-bad",
                "text": "Population: Greedy Sidon. Claim: ratio is strictly monotonic decreasing structurally",
                "test_methodology": "variance analysis",
            },
            {
                "id": "numeric-good",
                "text": "For n=45000, greedy Sidon F(n)/sqrt(n) falls below 0.58",
                "test_methodology": "greedy Sidon threshold crossing sweep",
            },
        ],
    }
    added, metrics = apply_synthesis_to_frontier(
        tree,
        state,
        parsed,
        question="[domain_profile:math_combinatorics] where does F(n)/sqrt(n) fall below 0.60?",
        generation=10,
        relevance_threshold=0.0,
        domain_id="math_combinatorics",
    )
    if metrics.get("n_rejected_unimplementable", 0) < 1:
        failures.append("expected structural claim rejected for math_combinatorics")
    if not added:
        failures.append("expected numeric sidon candidate added")

    prompt = compose_synthesis_prompt(
        question="Q1: 0.60 crossing",
        belief_state=state,
        tree=tree,
        lifetime_context=format_seeds_for_question([
            {
                "finding_type": "threshold_crossing",
                "claim": "F(n)/sqrt(n) first drops below 0.70 at n=10000",
                "next_hypotheses": ["Where below 0.60?"],
            }
        ]),
    )
    if "prior campaigns" not in prompt.lower() and "0.70" not in prompt:
        failures.append("lifetime seeds not in synthesis prompt")

    report = {
        "bucket": b,
        "tree_counts": counts,
        "synthesis_metrics": metrics,
        "added": len(added),
        "prompt_has_seeds": "0.70" in prompt,
        "failures": failures,
        "ok": not failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
