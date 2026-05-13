#!/usr/bin/env python3
"""
Start Campaign v2 — same research question as v1, for apples-to-apples comparison.

Writes artifacts/campaign_v2_latest.json with campaign_id for monitor_campaign.py.

Example (Windows PowerShell):
  $env:PROPAB_PROFILE = 'campaign'
  python scripts/start_campaign_v2.py --api http://localhost:8000 --hours 4

Example (bash):
  export PROPAB_PROFILE=campaign
  python scripts/start_campaign_v2.py --api http://localhost:8000 --hours 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Identical question string to scripts/start_campaign_v1.py (Campaign v1 reference).
CAMPAIGN_V2_QUESTION = (
    "Find the optimal MLP architecture for MNIST under a 50,000 parameter budget "
    "that maximizes test accuracy. Baseline: 784-60-10 single hidden layer. "
    "Breakthrough threshold: +5% accuracy improvement over baseline."
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="POST /campaigns for Campaign v2 (same question as v1).")
    parser.add_argument("--api", default="http://localhost:8000", help="Propab API base URL")
    parser.add_argument(
        "--hours",
        type=float,
        default=4.0,
        help="Compute budget in wall-clock hours (default 4)",
    )
    parser.add_argument(
        "--out",
        default=str(root / "artifacts" / "campaign_v2_latest.json"),
        help="Write campaign_id + metadata here",
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")

    body = json.dumps(
        {
            "question": CAMPAIGN_V2_QUESTION,
            "compute_budget_hours": args.hours,
            "breakthrough_criteria": {
                "metric_name": "val_accuracy",
                "improvement_threshold": 0.05,
                "direction": "higher_is_better",
                "min_confidence": 0.85,
                "min_replications": 3,
            },
        }
    ).encode("utf-8")

    req = Request(
        f"{api}/campaigns",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Cannot reach API at {api}: {exc}", file=sys.stderr)
        print("Start the stack: docker compose up -d", file=sys.stderr)
        sys.exit(1)

    cid = data["campaign_id"]
    record = {
        "campaign_id": cid,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
        "events_url": f"{api}/sessions/{cid}/events",
        "question": CAMPAIGN_V2_QUESTION,
        "compute_budget_hours": args.hours,
        "api": api,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print("Campaign v2 started.")
    print(json.dumps(record, indent=2))
    print(f"\nMonitor:  python scripts/monitor_campaign.py --state-file {out_path}")
    print(f"Or once:  python scripts/monitor_campaign.py --state-file {out_path} --once")


if __name__ == "__main__":
    main()
