#!/usr/bin/env python3
"""
Run a 3h policy calibration campaign (fixes.md P2).

Same domain and question family as the 3h contagion baseline; uses the latest
CANDIDATE policy for evaluation — not discovery-first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.orchestrator.seed_validation import PHASE2_CONTAGION_QUESTION

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "calibration_campaign_latest.json"),
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    body = json.dumps(
        {
            "question": PHASE2_CONTAGION_QUESTION,
            "compute_budget_hours": args.hours,
            "policy_mode": "candidate",
            "breakthrough_criteria": {
                "metric_name": "final_outbreak_fraction",
                "improvement_threshold": 0.05,
                "direction": "higher_is_better",
                "min_confidence": 0.85,
                "min_replications": 2,
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
        with urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:2000]}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Cannot reach {api}: {exc}", file=sys.stderr)
        sys.exit(1)

    cid = data["campaign_id"]
    record = {
        "campaign_id": cid,
        "question": PHASE2_CONTAGION_QUESTION,
        "compute_budget_hours": args.hours,
        "policy_mode": "candidate",
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
        "success_target": "closure_ratio >= 0.306 * 0.85 with theme_entropy < 2.0",
        "note": "Policy evaluation run — check lifetime.ingested evaluation field.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
