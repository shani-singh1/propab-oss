#!/usr/bin/env python3
"""
P4 — Demo pilot run (10–20 min). Purpose: find bugs, not discoveries.

Architecture freeze: no code changes once pilots stabilize.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from demo.benchmark.domain import DEMO_DOMAIN
from demo.benchmark.gold import load_baseline_metrics
from demo.benchmark.load import load_campaign_metrics, load_tree_summary_and_findings
from demo.benchmark.metric import metrics_from_api_summary
from demo.benchmark.report import build_campaign_asset, write_report, build_demo_report
from demo.benchmark.verifier import verify_pilot


def _post_campaign(api: str, hours: float) -> dict:
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
        return json.loads(resp.read())


def _get_campaign(api: str, campaign_id: str) -> dict:
    with urlopen(f"{api.rstrip('/')}/campaigns/{campaign_id}", timeout=60) as resp:
        return json.loads(resp.read())


def _poll_until_done(
    api: str,
    campaign_id: str,
    *,
    timeout_sec: float,
    interval_sec: float,
) -> dict:
    deadline = time.time() + timeout_sec
    last: dict = {}
    while time.time() < deadline:
        last = _get_campaign(api, campaign_id)
        status = (last.get("summary") or last).get("status") or ""
        if status in ("breakthrough", "budget_exhausted", "completed"):
            return last
        time.sleep(interval_sec)
    return last


async def _finalize(campaign_id: str, api_blob: dict, out_dir: Path) -> dict:
    baseline = await load_baseline_metrics()
    db_metrics = await load_campaign_metrics(campaign_id)
    metrics = db_metrics or metrics_from_api_summary(
        campaign_id, api_blob.get("summary") and api_blob or {"summary": api_blob},
    )
    tree_summary, findings, paper_url = await load_tree_summary_and_findings(campaign_id)
    verification = verify_pilot(metrics)
    asset = build_campaign_asset(
        metrics, verification, baseline,
        tree_summary=tree_summary, top_findings=findings, paper_url=paper_url,
    )
    report = build_demo_report([asset], gold_corpus_size=7, archive_size=48)
    paths = write_report(report, out_dir)
    return {
        "campaign_id": campaign_id,
        "verification": verification.to_dict(),
        "metrics": metrics.to_dict(),
        "outputs": {k: str(v) for k, v in paths.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo pilot run (10–20 min)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--minutes", type=float, default=15.0, help="Compute budget in minutes")
    parser.add_argument("--poll-timeout", type=float, default=1200.0, help="Max wait after launch (sec)")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--out-dir", default=str(ROOT / "artifacts" / "demo" / "pilot"))
    parser.add_argument("--launch-only", action="store_true")
    args = parser.parse_args()

    hours = max(0.1, args.minutes / 60.0)
    print(f"Pilot: {args.minutes:.0f} min | domain={DEMO_DOMAIN.domain_id}", flush=True)

    try:
        launched = _post_campaign(args.api, hours)
    except (HTTPError, URLError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        print("Start stack: docker compose up -d", file=sys.stderr)
        return 1

    cid = launched["campaign_id"]
    state_path = Path(args.out_dir) / "pilot_latest.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "campaign_id": cid,
        "domain": DEMO_DOMAIN.domain_id,
        "compute_budget_hours": hours,
        "api": args.api,
        "mode": "pilot",
    }, indent=2), encoding="utf-8")
    print(f"Launched {cid}", flush=True)

    if args.launch_only:
        return 0

    print(f"Polling up to {args.poll_timeout}s...", flush=True)
    final = _poll_until_done(
        args.api, cid, timeout_sec=args.poll_timeout, interval_sec=args.interval,
    )
    result = asyncio.run(_finalize(cid, final, Path(args.out_dir)))
    print(json.dumps(result, indent=2))
    return 0 if result["verification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
