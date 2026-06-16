#!/usr/bin/env python3
"""
P5 — Demo main runs (2–4 h). Parallel launches, no architecture changes.

Collects: paper, tree, traces, lineage, metrics via build_demo_assets after completion.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from demo.benchmark.domain import DEMO_DOMAIN


def _start_one(*, api: str, hours: float, index: int) -> dict:
    body = json.dumps(DEMO_DOMAIN.campaign_body(
        compute_budget_hours=hours,
        policy_mode="accepted",
    )).encode("utf-8")
    req = Request(
        f"{api.rstrip('/')}/campaigns",
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
        "state_url": f"{api.rstrip('/')}/campaigns/{data['campaign_id']}",
        "compute_budget_hours": hours,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo main runs (2–4 h, parallel)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=1, help="Parallel campaigns")
    parser.add_argument("--hours", type=float, default=3.0, help="Budget per campaign (2–4)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "demo" / "main_latest.json"),
    )
    args = parser.parse_args()

    if not (2.0 <= args.hours <= 4.0):
        print("Warning: fixes.md recommends 2–4 h; proceeding anyway.", flush=True)

    api = args.api.rstrip("/")
    workers = max(1, min(args.workers, args.count))
    print(
        f"Main demo: {args.count} × {args.hours}h | domain={DEMO_DOMAIN.domain_id}",
        flush=True,
    )

    results: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_start_one, api=api, hours=args.hours, index=i)
            for i in range(args.count)
        ]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
                print(f"  started {results[-1]['campaign_id']}", flush=True)
            except (HTTPError, URLError) as exc:
                errors.append(str(exc))

    payload = {
        "domain": DEMO_DOMAIN.to_dict(),
        "mode": "main",
        "campaigns": sorted(results, key=lambda r: r["index"]),
        "errors": errors,
        "note": (
            "Poll with monitor_campaign.py; then run build_demo_assets.py "
            "after campaigns complete."
        ),
    }
    if len(results) == 1:
        payload["campaign_id"] = results[0]["campaign_id"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if errors and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
