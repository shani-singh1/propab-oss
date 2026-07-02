#!/usr/bin/env python3
"""Snapshot root hypotheses from a running or finished V1 campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v1_literature_prior.compare import snapshot_campaign  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument(
        "--out",
        default=None,
        help="Default: artifacts/v1_literature_prior/hypotheses_<campaign_id[:8]>.json",
    )
    args = p.parse_args()

    out = Path(args.out or ROOT / "artifacts" / "v1_literature_prior" / f"hypotheses_{args.campaign_id[:8]}.json")
    payload = snapshot_campaign(args.campaign_id, out, api=args.api)
    print(json.dumps({
        "campaign_id": args.campaign_id,
        "hypothesis_count": payload["hypothesis_count"],
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
