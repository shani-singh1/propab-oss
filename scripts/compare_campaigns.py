#!/usr/bin/env python3
"""Compare two campaigns side-by-side (fixes.md D1)."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))


def fetch(api: str, cid: str) -> dict:
    with urllib.request.urlopen(f"{api.rstrip('/')}/campaigns/{cid}", timeout=60) as r:
        return json.loads(r.read())


def metrics_for(campaign: dict, summary: dict, replay: dict | None = None) -> dict:
    tree = (campaign.get("hypothesis_tree") or {}).get("nodes") or {}
    verdicts = {}
    for n in tree.values():
        v = n.get("verdict") or "pending"
        verdicts[v] = verdicts.get(v, 0) + 1
    total = sum(verdicts.values()) or 1
    bs = campaign.get("belief_state") or {}
    return {
        "confirmed": verdicts.get("confirmed", 0),
        "refuted": verdicts.get("refuted", 0),
        "inconclusive": verdicts.get("inconclusive", 0),
        "inconclusive_pct": round(100 * verdicts.get("inconclusive", 0) / total, 1),
        "false_confirm_rate": (replay or {}).get("false_positive_rate_before"),
        "beliefs_active": len(bs.get("active_beliefs") or []),
        "beliefs_ungrounded": len(bs.get("proposed_ungrounded_beliefs") or []),
        "elapsed_min": round((summary.get("elapsed_sec") or 0) / 60, 1),
        "budget_used_pct": round(
            100 * (summary.get("elapsed_sec") or 0) / max(campaign.get("compute_budget_seconds") or 1, 1),
            1,
        ),
        "stop_reason": campaign.get("stop_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-a", required=True)
    parser.add_argument("--campaign-b", required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--replay-a", default=None, help="Deep analysis JSON for A")
    parser.add_argument("--replay-b", default=None, help="Deep analysis JSON for B")
    args = parser.parse_args()

    raw_a = fetch(args.api, args.campaign_a)
    raw_b = fetch(args.api, args.campaign_b)
    ca, cb = raw_a.get("campaign") or {}, raw_b.get("campaign") or {}
    replay_a = json.loads(Path(args.replay_a).read_text()) if args.replay_a else None
    replay_b = json.loads(Path(args.replay_b).read_text()) if args.replay_b else None
    ma = metrics_for(ca, raw_a.get("summary") or {}, (replay_a or {}).get("replay_with_fixed_verifier"))
    mb = metrics_for(cb, raw_b.get("summary") or {}, (replay_b or {}).get("replay_with_fixed_verifier"))

    keys = sorted(set(ma) | set(mb))
    print(f"{'metric':<22} {'A':>12} {'B':>12} {'delta':>12}")
    print("-" * 62)
    for k in keys:
        a, b = ma.get(k), mb.get(k)
        delta = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d = b - a
            delta = f"{d:+.1f}" if isinstance(a, float) else f"{d:+d}"
            if k == "inconclusive_pct" and d < 0:
                delta += " (better)"
            if k == "beliefs_active" and d > 0:
                delta += " (better)"
        print(f"{k:<22} {str(a):>12} {str(b):>12} {delta:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
