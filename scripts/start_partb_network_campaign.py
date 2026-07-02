#!/usr/bin/env python3
"""Start Part B validation campaign (fixes.md) — network resilience domain."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

QUESTION = (
    "Investigate whether simple graph-topological features can reliably predict "
    "resilience of complex networks to targeted node removal. Discover predictive "
    "metrics, verify on synthetic graph families, and characterize failure regimes."
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B live campaign — network resilience")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "partb_live_campaign_latest.json"),
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    body = json.dumps(
        {
            "question": QUESTION,
            "compute_budget_hours": args.hours,
            "max_hypotheses": 40,
            "breakthrough_criteria": {
                "metric_name": "resilience_prediction_score",
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
        print(exc.read().decode("utf-8", errors="replace")[:2000], file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Cannot reach {api}: {exc}", file=sys.stderr)
        sys.exit(1)

    record = {
        "campaign_id": data["campaign_id"],
        "stream_url": f"{api}/stream/{data['campaign_id']}",
        "state_url": f"{api}/campaigns/{data['campaign_id']}",
        "events_url": f"{api}/sessions/{data['campaign_id']}/events",
        "question": QUESTION,
        "compute_budget_hours": args.hours,
        "max_hypotheses": 40,
        "part_b_interpreter": True,
        "config_note": "campaign_expand_use_interpreter=True (default)",
        "domain": "network_resilience",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "audit_plan": "Post-run: artifact gate + permutation null on confirmed findings",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
