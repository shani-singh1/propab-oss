#!/usr/bin/env python3
"""Harvest DB-backed operator traces from Postgres (fixes.md #1)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.operator_credit.campaign_corpus import harvest_from_bundles
from propab.operator_credit.db_trace_loader import (
    campaign_ids_from_trajectory,
    load_bundles_from_db,
)


async def _all_campaign_ids_with_trees() -> list[str]:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT id::text FROM research_campaigns
                    WHERE hypothesis_tree_json IS NOT NULL
                    ORDER BY id
                    """
                ),
            )
        ).fetchall()
    await engine.dispose()
    return [str(r[0]) for r in rows]


async def _harvest(campaign_ids: list[str]) -> dict:
    bundles = await load_bundles_from_db(campaign_ids)
    corpus, ledger = harvest_from_bundles(bundles)
    n_db = sum(1 for t in ledger.traces if t.source == "db")
    n_tools = sum(1 for t in ledger.traces if t.tool_calls)
    return {
        "n_campaigns": len(bundles),
        "n_traces": len(ledger.traces),
        "n_traces_from_db": n_db,
        "n_traces_with_tool_calls": n_tools,
        "campaigns": [b.to_dict() for b in bundles],
        "coverage": corpus.coverage.to_dict() if corpus.coverage else {},
        "sample_traces": [t.to_dict() for t in ledger.traces[:3]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest DB-backed operator traces")
    parser.add_argument(
        "--trajectories",
        default=str(ROOT / "artifacts" / "entropy_trajectories.json"),
        help="Trajectory file listing campaign IDs to harvest",
    )
    parser.add_argument(
        "--campaign-id",
        action="append",
        dest="campaign_ids",
        help="Explicit campaign ID (repeatable)",
    )
    parser.add_argument(
        "--all-db",
        action="store_true",
        help="Harvest all campaigns with hypothesis trees in Postgres",
    )
    args = parser.parse_args()

    ids = list(args.campaign_ids or [])
    if args.all_db:
        ids = asyncio.run(_all_campaign_ids_with_trees())
    traj = Path(args.trajectories)
    if not ids and traj.is_file():
        ids = campaign_ids_from_trajectory(traj)
    if not ids:
        print("No campaign IDs provided.", file=sys.stderr)
        return 1

    report = asyncio.run(_harvest(ids))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
