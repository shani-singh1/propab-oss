#!/usr/bin/env python3
"""Start a 2h number-theory campaign (Sierpiński / 5/n Egyptian fractions)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SIERPINSKI_QUESTION = (
    "Investigate the Sierpiński conjecture computationally using exact integer arithmetic: "
    "for every odd integer n > 1, determine whether 5/n can be written as a sum of three "
    "positive unit fractions, 5/n = 1/x + 1/y + 1/z with positive integers x, y, z. "
    "Characterize which residue classes modulo small moduli admit closed-form parametric "
    "solution families versus those requiring search; discover and rigorously verify explicit "
    "parametric families that cover infinite classes of odd n; and exhaustively verify instances "
    "up to n = 1,000,000 with exact arithmetic certificates or counterexamples. "
    "Every claim must be checked in integer arithmetic (no floating point)."
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument(
        "--out",
        default=str(root / "artifacts" / "sierpinski_campaign_latest.json"),
    )
    args = p.parse_args()
    api = args.api.rstrip("/")

    body = json.dumps(
        {
            "question": SIERPINSKI_QUESTION,
            "compute_budget_hours": args.hours,
            "breakthrough_criteria": {
                "metric_name": "verified_claims",
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
        print(exc.read().decode("utf-8", errors="replace")[:2000], file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Cannot reach API at {api}: {exc}", file=sys.stderr)
        return 1

    record = {
        "campaign_id": data["campaign_id"],
        "stream_url": f"{api}/stream/{data['campaign_id']}",
        "state_url": f"{api}/campaigns/{data['campaign_id']}",
        "events_url": f"{api}/sessions/{data['campaign_id']}/events",
        "question": SIERPINSKI_QUESTION,
        "compute_budget_hours": args.hours,
        "api": api,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nMonitor: python scripts/monitor_campaign.py --state-file {out} --log artifacts/sierpinski_campaign_live.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
