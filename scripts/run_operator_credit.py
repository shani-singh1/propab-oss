#!/usr/bin/env python3
"""
Operator credit assignment cycle (fixes.md operator_credit_assignment).

Experience → Replay → Counterfactuals → Operator Credits → Priors → Bench

Usage:
  python scripts/run_operator_credit.py
  python scripts/run_operator_credit.py --trajectories artifacts/entropy_trajectories.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.operator_credit.credit_cycle import run_operator_credit_cycle


async def _all_db_campaign_ids() -> list[str]:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    "SELECT id::text FROM research_campaigns "
                    "WHERE hypothesis_tree_json IS NOT NULL ORDER BY id"
                ),
            )
        ).fetchall()
    await engine.dispose()
    return [str(r[0]) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator credit assignment")
    parser.add_argument(
        "--trajectories",
        default=str(ROOT / "artifacts" / "entropy_trajectories.json"),
    )
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--all-db",
        action="store_true",
        help="Use all campaigns with hypothesis trees in Postgres",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip Postgres; use offline trajectory/snapshot approximations only",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "operator_credit_report.json"),
    )
    args = parser.parse_args()

    traj = Path(args.trajectories)
    if not traj.is_file():
        print(f"Trajectories not found: {traj}", file=sys.stderr)
        return 1

    campaign_ids = None
    if args.all_db:
        campaign_ids = asyncio.run(_all_db_campaign_ids())
        print(f"Loading {len(campaign_ids)} campaigns from Postgres...", flush=True)

    report, traces, credits = run_operator_credit_cycle(
        trajectory_path=traj,
        campaign_ids=campaign_ids,
        persist=not args.no_persist,
        use_db=not args.no_db,
    )
    payload = {
        "layer": "operator_credit",
        "trajectories": str(traj),
        "report": report.to_dict(),
        "sample_traces": [t.to_dict() for t in traces.traces[:3]],
        "top_credits": sorted(
            [c.to_dict() for c in credits.credits],
            key=lambda c: c["contribution"],
            reverse=True,
        )[:5],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
