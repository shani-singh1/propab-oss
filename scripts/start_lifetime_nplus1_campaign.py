#!/usr/bin/env python3
"""
Start a focused Campaign N+1 run after lifetime knowledge is populated.

Uses a narrower question on high-yield contagion themes (spectral, k-shell,
modularity) so policy boosts and dead-end injection are observable within ~1h.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

# Follow-up: replicate and discriminate among themes that confirmed in prior runs.
LIFETIME_NPLUS1_QUESTION = (
    "Building on prior campaigns: which structural determinants—spectral gap "
    "(lambda_2/lambda_1), k-shell/coreness, or modularity/community structure—"
    "most causally predict final outbreak fraction and contagion speed under SIS "
    "versus independent-cascade diffusion on scale-free and configuration-model "
    "networks? Prioritize T2+ replication; avoid repeating saturated general-theme "
    "hypotheses and known dead ends from lifetime knowledge."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=1.0, help="Compute budget (1h default)")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "lifetime_nplus1_campaign_latest.json"),
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    body = json.dumps(
        {
            "question": LIFETIME_NPLUS1_QUESTION,
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
        "question": LIFETIME_NPLUS1_QUESTION,
        "compute_budget_hours": args.hours,
        "api": api,
        "stream_url": f"{api}/stream/{cid}",
        "state_url": f"{api}/campaigns/{cid}",
        "note": "Run after scripts/ingest_campaign_lifetime.py on prior campaign(s).",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
