#!/usr/bin/env python3
"""Resume a checkpointed campaign via POST /campaigns/{id}/resume."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_launch(path: Path, campaign_id: str) -> dict | None:
    if not path.exists():
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    if blob.get("campaign_id") == campaign_id:
        return blob
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_id", nargs="?", default="faaf394b-7f95-4778-9136-e922f2401e7f")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--launch-config",
        type=Path,
        default=None,
        help="Launch JSON (default: mandrake_contrarian_campaign.json if present)",
    )
    parser.add_argument(
        "--contrarian",
        action="store_true",
        help="Contrarian reframing: belief reset, no hypothesis cap, 2h budget",
    )
    parser.add_argument(
        "--max-hypotheses",
        type=int,
        default=None,
        help="Absolute cap override (omit with --contrarian for no cap)",
    )
    parser.add_argument(
        "--additional-hypotheses",
        type=int,
        default=30,
        help="When resuming without --contrarian: cap = tested + this many (default 30)",
    )
    parser.add_argument("--compute-budget-hours", type=float, default=None)
    args = parser.parse_args()

    launch_path = args.launch_config
    if launch_path is None:
        contrarian_path = ROOT / "artifacts" / "mandrake_contrarian_campaign.json"
        latest_path = ROOT / "artifacts" / "mandrake_campaign_latest.json"
        launch_path = contrarian_path if (args.contrarian or contrarian_path.exists()) else latest_path

    launch_blob = _load_launch(launch_path, args.campaign_id) if launch_path else None
    contrarian = args.contrarian or (launch_blob or {}).get("belief_reset") == "contrarian"

    body: dict = {}

    if contrarian:
        body["belief_reset"] = "contrarian"
        body["clear_hypothesis_cap"] = bool((launch_blob or {}).get("clear_hypothesis_cap", True))
        if launch_blob and launch_blob.get("question"):
            body["question"] = launch_blob["question"]
    elif args.max_hypotheses is not None:
        body["max_hypotheses_cap"] = int(args.max_hypotheses)
    else:
        try:
            camp = json.loads(
                urllib.request.urlopen(
                    f"{args.api.rstrip('/')}/campaigns/{args.campaign_id}", timeout=30
                ).read()
            )
            tested = int((camp.get("summary") or {}).get("total_hypotheses") or 0)
            body["max_hypotheses_cap"] = tested + max(1, int(args.additional_hypotheses))
        except Exception:
            cap = launch_blob.get("max_hypotheses") if launch_blob else 50
            if cap is not None:
                body["max_hypotheses_cap"] = int(cap)

    budget = args.compute_budget_hours
    if budget is None and launch_blob:
        budget = launch_blob.get("compute_budget_hours")
    if budget is None:
        budget = 2.0
    body["compute_budget_hours"] = float(budget)

    url = f"{args.api.rstrip('/')}/campaigns/{args.campaign_id}/resume"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"Resume failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**data, "resume_body": body, "launch_config": str(launch_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
