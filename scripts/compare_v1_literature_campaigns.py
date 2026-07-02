#!/usr/bin/env python3
"""
Compare hypotheses: campaign without literature prior vs campaign with prior.

Example (after both campaigns have run):
  python scripts/compare_v1_literature_campaigns.py \\
    --baseline d7afdf9c-0e2c-4528-b795-f2c4afb9877b \\
    --with-prior <campaign2_id>

Or use saved snapshots:
  python scripts/compare_v1_literature_campaigns.py \\
    --baseline-snapshot artifacts/v1_literature_prior/hypotheses_d7afdf9c.json \\
    --with-prior-snapshot artifacts/v1_literature_prior/hypotheses_abc12345.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v1_literature_prior.compare import (  # noqa: E402
    compare_hypothesis_sets,
    fetch_campaign_hypotheses,
    fetch_hypotheses_from_events,
    load_hypothesis_snapshot,
    snapshot_campaign,
)


def _texts_from_campaign(cid: str, api: str) -> list[str]:
    nodes = fetch_campaign_hypotheses(cid, api=api)
    texts = [n["text"] for n in nodes]
    if not texts:
        texts = fetch_hypotheses_from_events(cid, api=api)
    return texts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", help="Campaign ID without literature prior")
    p.add_argument("--with-prior", help="Campaign ID launched with literature prior")
    p.add_argument("--baseline-snapshot", help="JSON snapshot from snapshot_v1_campaign_hypotheses.py")
    p.add_argument("--with-prior-snapshot")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--out", default=str(ROOT / "artifacts" / "v1_literature_prior" / "campaign_comparison.json"))
    args = p.parse_args()

    if args.baseline_snapshot:
        baseline = load_hypothesis_snapshot(Path(args.baseline_snapshot))
    elif args.baseline:
        baseline = _texts_from_campaign(args.baseline, args.api)
    else:
        print("Provide --baseline or --baseline-snapshot", file=sys.stderr)
        return 1

    if args.with_prior_snapshot:
        with_prior = load_hypothesis_snapshot(Path(args.with_prior_snapshot))
    elif args.with_prior:
        with_prior = _texts_from_campaign(args.with_prior, args.api)
    else:
        print("Provide --with-prior or --with-prior-snapshot", file=sys.stderr)
        return 1

    result = compare_hypothesis_sets(baseline, with_prior)
    result["baseline_source"] = args.baseline or args.baseline_snapshot
    result["with_prior_source"] = args.with_prior or args.with_prior_snapshot

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "baseline_count": result["baseline_count"],
        "with_prior_count": result["with_prior_count"],
        "overlap_ratio": result["overlap_ratio"],
        "novel_with_prior": len(result["novel_with_prior"]),
        "novel_term_hints": result["novel_term_hints"][:10],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
