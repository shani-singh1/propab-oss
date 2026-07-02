#!/usr/bin/env python3
"""Compare duplicate pairs before vs after resume (fixes.md step 4)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.campaign_synthesis import text_similarity  # noqa: E402


def _get_json(url: str):
    with urlopen(url, timeout=120) as r:
        return json.load(r)


def _pair_keys(tree: dict, threshold: float = 0.85) -> set[tuple[str, str]]:
    nodes = tree.get("nodes") or {}
    ids = list(nodes.keys())
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(ids):
        ta = nodes[a].get("text") or ""
        for b in ids[i + 1:]:
            tb = nodes[b].get("text") or ""
            if text_similarity(ta, tb) >= threshold:
                pairs.add(tuple(sorted((a, b))))
    return pairs

DEFAULT_CID = "faaf394b-7f95-4778-9136-e922f2401e7f"
BASELINE = ROOT / "artifacts" / "mandrake_duplicate_pair_analysis.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=DEFAULT_CID)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "mandrake_post_resume_duplicate_check.json")
    args = parser.parse_args()

    base = args.api.rstrip("/")
    camp_resp = _get_json(f"{base}/campaigns/{args.campaign_id}")
    camp = camp_resp.get("campaign") or camp_resp
    tree = camp.get("hypothesis_tree") or {}

    current_keys = _pair_keys(tree)
    current_n = len(current_keys)

    baseline_pairs = set()
    baseline_n = 0
    if args.baseline.exists():
        bl = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_n = bl.get("n_near_duplicate_pairs", 0)
        for p in bl.get("pairs") or bl.get("pairs_enriched") or []:
            a, b = p.get("node_a"), p.get("node_b")
            if a and b:
                baseline_pairs.add(tuple(sorted((a, b))))

    new_pairs = current_keys - baseline_pairs
    report = {
        "campaign_id": args.campaign_id,
        "baseline_duplicate_pairs": baseline_n,
        "current_duplicate_pairs": current_n,
        "new_duplicate_pairs_since_baseline": len(new_pairs),
        "new_pair_node_ids": [list(k) for k in sorted(new_pairs)[:20]],
        "fix_held": len(new_pairs) == 0,
        "current_node_count": len(tree.get("nodes") or {}),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["fix_held"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
