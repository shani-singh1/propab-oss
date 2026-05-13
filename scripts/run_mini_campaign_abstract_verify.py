#!/usr/bin/env python3
"""
Start a 30-minute compute-budget campaign and poll until the paper is ready, then assert
the paper abstract matches ``summary.total_confirmed`` (via ``poll_campaign_for_review``).

Requires a running stack (API + worker + Postgres + Redis). Set ``PROPAB_PROFILE=campaign``
on the **worker** (and API if settings are read there) for production-like caps.

Usage (repo root, PowerShell)::

    $env:PROPAB_PROFILE = 'campaign'
    python scripts/run_mini_campaign_abstract_verify.py --api http://localhost:8000

Artifacts: ``artifacts/mini_campaign_verify_latest.json`` (campaign_id for manual follow-up).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "mini_campaign_verify_latest.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--hours", type=float, default=0.5, help="Compute budget hours (default 0.5 = 30 min)")
    p.add_argument("--poll-max-seconds", type=int, default=1860, help="Wall poll budget (~31 min)")
    args = p.parse_args()
    api = args.api.rstrip("/")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    start = [
        sys.executable,
        str(ROOT / "scripts" / "start_campaign_v2.py"),
        "--api",
        api,
        "--hours",
        str(args.hours),
        "--out",
        str(OUT),
    ]
    print("Running:", " ".join(start), flush=True)
    r0 = subprocess.run(start, cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    if r0.returncode != 0:
        print(r0.stdout, file=sys.stderr)
        print(r0.stderr, file=sys.stderr)
        return r0.returncode

    poll = [
        sys.executable,
        str(ROOT / "scripts" / "poll_campaign_for_review.py"),
        "--api",
        api,
        "--state-file",
        str(OUT),
        "--max-seconds",
        str(int(args.poll_max_seconds)),
        "--interval",
        "45",
    ]
    print("Running:", " ".join(poll), flush=True)
    r1 = subprocess.run(poll, cwd=str(ROOT), text=True, timeout=int(args.poll_max_seconds) + 120)
    return int(r1.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
