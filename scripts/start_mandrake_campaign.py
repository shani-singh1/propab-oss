#!/usr/bin/env python3
"""
Phase 8 — Mandrake anomaly-driven campaign with real LOFO verification.

Prerequisite: python scripts/run_anomaly_pipeline.py
Use PROPAB_PROFILE=campaign for depth (see docker-compose.mount-dev.yml).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.mandrake.domain import CAMPAIGN_QUESTION


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Mandrake anomaly-seeded campaign (LOFO verification)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=0.5, help="Campaign budget (fixes.md: 30 min)")
    parser.add_argument("--max-hypotheses", type=int, default=20, help="Mechanism diversity over R² (fixes.md)")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "mandrake_campaign_latest.json"),
    )
    args = parser.parse_args()

    mech = (ROOT / args.artifacts_dir / "mechanism_objects.json").resolve()
    if not mech.is_file():
        print("Run pipeline first: python scripts/run_anomaly_pipeline.py", file=sys.stderr)
        return 1

    api = args.api.rstrip("/")
    body = json.dumps({
        "question": CAMPAIGN_QUESTION,
        "compute_budget_hours": args.hours,
        "seed_source": "anomaly",
        "anomaly_artifacts_dir": args.artifacts_dir,
        "max_hypotheses": args.max_hypotheses,
        "policy_mode": "accepted",
        "breakthrough_criteria": {
            "metric_name": "lofo_r2",
            "improvement_threshold": 0.05,
            "direction": "higher_is_better",
            "min_confidence": 0.85,
            "min_replications": 1,
        },
    }).encode("utf-8")

    req = Request(
        f"{api}/campaigns",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1

    record = {
        "campaign_id": data["campaign_id"],
        "question": CAMPAIGN_QUESTION,
        "seed_source": "anomaly",
        "artifacts_dir": args.artifacts_dir,
        "compute_budget_hours": args.hours,
        "max_hypotheses": args.max_hypotheses,
        "propab_profile": os.environ.get("PROPAB_PROFILE", "campaign"),
        "note": "Verification uses mandrake_verification (real LOFO). Poll: python scripts/monitor_campaign.py",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
