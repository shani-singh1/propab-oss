#!/usr/bin/env python3
"""Start scope-gate integrity campaign (fixes.md) — one experiment on Propab itself."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.orchestrator.seed_validation import PHASE2_CONTAGION_QUESTION

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scope gate integrity campaign (fixes.md)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--hours", type=float, default=1.0, help="Budget (~20 hypotheses)")
    parser.add_argument("--max-hypotheses", type=int, default=20)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "scope_gate_campaign_latest.json"))
    args = parser.parse_args()

    api = args.api.rstrip("/")
    body = json.dumps({
        "question": PHASE2_CONTAGION_QUESTION,
        "compute_budget_hours": args.hours,
        "max_hypotheses": args.max_hypotheses,
        "breakthrough_criteria": {
            "metric_name": "final_outbreak_fraction",
            "improvement_threshold": 0.05,
            "direction": "higher_is_better",
            "min_confidence": 0.85,
            "min_replications": 2,
        },
        "note": "scope_gate_integrity_audit — no architecture changes",
    }).encode("utf-8")

    req = Request(f"{api}/campaigns", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1

    record = {
        "campaign_id": data["campaign_id"],
        "question": PHASE2_CONTAGION_QUESTION,
        "compute_budget_hours": args.hours,
        "max_hypotheses": args.max_hypotheses,
        "api": api,
        "poll": f"python scripts/finalize_scope_campaign.py --campaign-id {data['campaign_id']}",
        "note": "Stop after one campaign. Output is information, not discoveries.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
