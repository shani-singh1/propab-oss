#!/usr/bin/env python3
"""Start Phase 2 contagion-network campaign (fixes.md) after seed suite passes."""
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
    parser.add_argument("--hours", type=float, default=7.0, help="Compute budget (6–8 recommended)")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "contagion_campaign_latest.json"),
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    body = json.dumps(
        {
            "question": PHASE2_CONTAGION_QUESTION,
            "compute_budget_hours": args.hours,
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
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
        "note": "Run only after scripts/run_seed_validation_suite.py passes.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
