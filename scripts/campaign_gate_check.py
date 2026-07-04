#!/usr/bin/env python3
"""30-min gate + health check for active math_combinatorics campaign."""
from __future__ import annotations

import json
import sys
from collections import Counter
from urllib.request import urlopen

from propab.numerical_seeds import classify_hypothesis_bucket
from propab.synthesis_diversity import tree_problem_counts_from_nodes

CID = sys.argv[1] if len(sys.argv) > 1 else "29297598-da18-4a1e-97da-d5bd21f60c58"
API = "http://localhost:8000"


def event_payload(ev: dict) -> dict:
    raw = ev.get("payload_json", ev.get("payload"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def main() -> int:
    with urlopen(f"{API}/campaigns/{CID}", timeout=120) as resp:
        raw = json.loads(resp.read())
    c = raw.get("campaign") or raw
    summary = raw.get("summary") or {}
    nodes = (c.get("hypothesis_tree") or {}).get("nodes") or {}
    bs = c.get("belief_state") or {}

    by_verdict = Counter()
    for n in nodes.values():
        if isinstance(n, dict):
            by_verdict[n.get("verdict") or "pending"] += 1

    node_dicts = {nid: n for nid, n in nodes.items() if isinstance(n, dict)}
    buckets = tree_problem_counts_from_nodes(node_dicts)
    total_typed = sum(buckets.values()) or 1
    dominant = max(buckets, key=buckets.get) if buckets else "none"
    same_type_fraction = (buckets.get(dominant, 0) / total_typed) if buckets else 0.0

    active = bs.get("active_beliefs") or []
    supporting = [len(b.get("supporting_nodes") or []) for b in active if isinstance(b, dict)]
    max_supporting = max(supporting) if supporting else 0

    with urlopen(f"{API}/sessions/{CID}/events?limit=400", timeout=60) as resp:
        evdata = json.loads(resp.read())
    events = evdata if isinstance(evdata, list) else list(evdata.get("events") or [])

    seeds_loaded = 0
    synth_metrics: list[dict] = []
    for ev in events:
        step = ev.get("step") or ""
        p = event_payload(ev)
        if step == "lifetime.knowledge_loaded":
            seeds_loaded = max(seeds_loaded, int(p.get("numerical_seeds_loaded") or 0))
        if step == "campaign.synthesis":
            m = p.get("metrics") or p
            if isinstance(m, dict):
                synth_metrics.append(m)

    trend_promotions = sum(
        int((m or {}).get("beliefs_promoted_by_trend") or 0) for m in synth_metrics
    )
    tested = int(summary.get("total_hypotheses") or 0)
    confirmed = int(summary.get("total_confirmed") or 0)
    inconclusive = by_verdict.get("inconclusive", 0)
    resolved = confirmed + by_verdict.get("refuted", 0) + inconclusive
    inconclusive_rate = (inconclusive / resolved) if resolved else 0.0

    gate = {
        "campaign_id": CID,
        "status": summary.get("status") or c.get("status"),
        "elapsed_sec": summary.get("elapsed_sec"),
        "remaining_sec": summary.get("remaining_sec"),
        "tree_nodes": len(nodes),
        "verdicts": dict(by_verdict),
        "problem_buckets": buckets,
        "same_type_fraction": round(same_type_fraction, 3),
        "dominant_type": dominant,
        "active_beliefs": len(active),
        "max_supporting_nodes": max_supporting,
        "numerical_seeds_loaded": seeds_loaded,
        "beliefs_promoted_by_trend": trend_promotions,
        "inconclusive_rate": round(inconclusive_rate, 3),
        "hypotheses_tested": tested,
        "confirmed": confirmed,
        "gates": {
            "same_type_fraction_lt_60pct": same_type_fraction < 0.60,
            "supporting_nodes_gt_5": max_supporting > 5,
            "numerical_seeds_loaded_gt_0": seeds_loaded > 0,
            "beliefs_promoted_by_trend_gt_0": trend_promotions > 0,
            "inconclusive_rate_lt_60pct": inconclusive_rate < 0.60,
        },
        "all_pass": all([
            same_type_fraction < 0.60,
            max_supporting > 5,
            seeds_loaded > 0,
        ]),
    }
    print(json.dumps(gate, indent=2))
    return 0 if gate["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
