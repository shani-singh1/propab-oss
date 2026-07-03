#!/usr/bin/env python3
"""Verifier feature coverage per campaign (fixes.md D3)."""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fetch(api: str, cid: str) -> dict:
    with urllib.request.urlopen(f"{api.rstrip('/')}/campaigns/{cid}", timeout=60) as r:
        return json.loads(r.read())


def infer_metric(node: dict) -> str:
    for key in ("finding", "evidence_summary"):
        block = node.get(key) or {}
        if isinstance(block, dict) and block.get("metric_name"):
            return str(block["metric_name"])
    text = str(node.get("text") or "").lower()
    if "cap set" in text or "f_3" in text:
        return "cap_set_clp_ratio"
    if "ap-free" in text:
        return "ap_free_density"
    if "sumset" in text:
        return "sumset_growth"
    if "bose" in text:
        return "bose_chowla_vs_greedy_ratio"
    return "sidon_ratio_to_sqrt_n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    raw = fetch(args.api, args.campaign_id)
    nodes = ((raw.get("campaign") or {}).get("hypothesis_tree") or {}).get("nodes") or {}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for n in nodes.values():
        if not isinstance(n, dict):
            continue
        v = n.get("verdict") or "pending"
        if v == "pending":
            continue
        feat = infer_metric(n)
        stats[feat]["called"] += 1
        stats[feat][v] += 1

    rows = []
    print(f"{'Feature':<28} {'Called':>8} {'Confirmed':>10} {'Refuted':>10} {'Inconclusive':>12}")
    for feat in sorted(stats):
        s = stats[feat]
        print(
            f"{feat:<28} {s['called']:>8} {s.get('confirmed', 0):>10} "
            f"{s.get('refuted', 0):>10} {s.get('inconclusive', 0):>12}"
        )
        rows.append({"feature": feat, **dict(s)})
    report = {"campaign_id": args.campaign_id, "features": rows}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
