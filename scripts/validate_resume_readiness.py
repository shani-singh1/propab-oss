#!/usr/bin/env python3
"""CLI for validate_resume_readiness without requiring the new API route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "propab-core"))

from propab.campaign_resume import validate_resume_readiness  # noqa: E402


def _get(url: str):
    with urlopen(url, timeout=120) as r:
        return json.load(r)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_id", nargs="?", default="faaf394b-7f95-4778-9136-e922f2401e7f")
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    base = args.api.rstrip("/")
    camp_resp = _get(f"{base}/campaigns/{args.campaign_id}")
    camp = camp_resp.get("campaign") or camp_resp
    events = _get(f"{base}/sessions/{args.campaign_id}/events?limit=2000")
    if not isinstance(events, list):
        events = events.get("events", [])

    launch_meta = None
    launch_path = ROOT / "artifacts" / "mandrake_campaign_latest.json"
    if launch_path.exists():
        blob = json.loads(launch_path.read_text(encoding="utf-8"))
        if blob.get("campaign_id") == args.campaign_id:
            launch_meta = blob

    report = validate_resume_readiness(camp, events=events, launch_meta=launch_meta)
    out = ROOT / "artifacts" / f"resume_readiness_{args.campaign_id[:8]}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    ok = report.get("resume_ready") or report.get("belief_backfill_available")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
