#!/usr/bin/env python3
"""
Launch N parallel graphs×3h campaigns for Level 1 residual collection (fixes.md).

Each campaign is an independent API instance — no orchestrator changes required.
Residuals are collected from Postgres events (see export_policy_residuals.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.orchestrator.seed_validation import PHASE2_CONTAGION_QUESTION

ROOT = Path(__file__).resolve().parents[1]


def _start_one(
    *,
    api: str,
    hours: float,
    policy_mode: str,
    index: int,
) -> dict:
    body = json.dumps(
        {
            "question": PHASE2_CONTAGION_QUESTION,
            "compute_budget_hours": hours,
            "policy_mode": policy_mode,
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
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return {
        "index": index,
        "campaign_id": data["campaign_id"],
        "stream_url": data.get("stream_url"),
        "policy_mode": policy_mode,
        "compute_budget_hours": hours,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch parallel graphs×3h residual batch.")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=5, help="Campaigns to launch (5–10 recommended)")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument(
        "--policy-mode",
        default="candidate",
        choices=("candidate", "accepted"),
        help="candidate = calibration evaluation; accepted = discovery only",
    )
    parser.add_argument("--workers", type=int, default=5, help="Parallel HTTP launch threads")
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "residual_batch_latest.json"),
    )
    args = parser.parse_args()
    api = args.api.rstrip("/")
    workers = max(1, min(args.workers, args.count))

    print(f"Launching {args.count} campaigns ({args.hours}h, {args.policy_mode}) ...", flush=True)
    results: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _start_one,
                api=api,
                hours=args.hours,
                policy_mode=args.policy_mode,
                index=i,
            )
            for i in range(args.count)
        ]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
                print(f"  started {results[-1]['campaign_id']}", flush=True)
            except (HTTPError, URLError, OSError) as exc:
                errors.append(str(exc))
                print(f"  FAILED: {exc}", file=sys.stderr)

    policy_context: dict = {}
    try:
        from propab.policy_store import PolicyStore
        from propab.policy_record import PolicyStatus

        store = PolicyStore.load()
        acc_id = store.accepted.get("graphs:3h")
        acc = store.get_policy(acc_id) if acc_id else None
        cands = [
            p for p in store.policies.values()
            if p.status == PolicyStatus.CANDIDATE
            and p.budget_bucket == "3h"
            and p.domain_bucket == "graphs"
        ]
        latest = max(cands, key=lambda p: p.generation) if cands else None
        policy_context = {
            "accepted_policy_id": acc_id,
            "accepted_generation": acc.generation if acc else None,
            "accepted_boosts": acc.boosts if acc else None,
            "candidate_policy_id": latest.id if latest else None,
            "candidate_parent": latest.parent_policy_id if latest else None,
            "note": (
                "All candidate-mode runs bind to the same latest graphs:3h CANDIDATE at launch; "
                "evaluation deltas use the pinned baseline observation from batch start."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        policy_context = {"policy_lookup_error": str(exc)}

    record = {
        "api": api,
        "batch_version": "v2",
        "count_requested": args.count,
        "count_started": len(results),
        "policy_mode": args.policy_mode,
        "compute_budget_hours": args.hours,
        "policy_context": policy_context,
        "campaigns": sorted(results, key=lambda x: x["index"]),
        "errors": errors,
        "note": (
            "Parallel orchestration is supported; Celery workers are shared (concurrency=16). "
            "Export residuals with scripts/export_policy_residuals.py after completion."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if results and not errors else (1 if not results else 0)


if __name__ == "__main__":
    raise SystemExit(main())
