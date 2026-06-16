#!/usr/bin/env python3
"""Partition campaigns into architecture eras and select gold corpus (fixes.md)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "orchestrator"))

from propab.operator_credit.campaign_era import (
    ERA_LABELS,
    CampaignEraPartition,
    build_era_definitions,
    build_experience_archive,
    compute_cross_era_comparisons,
    compute_era_local_statistics,
    current_git_commit,
    select_gold_corpus,
)
from propab.operator_credit.credit_cycle import run_operator_credit_cycle
from propab.operator_credit.db_trace_loader import load_bundles_from_db
from propab.operator_credit.campaign_corpus import harvest_from_bundles
from propab.operator_credit.difference_rewards import build_difference_rewards
from propab.operator_credit.era_loader import load_campaign_era_metadata
from propab.policy_store import PolicyStore


async def _all_campaign_ids() -> list[str]:
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
                    "WHERE hypothesis_tree_json IS NOT NULL ORDER BY started_at ASC"
                ),
            )
        ).fetchall()
    await engine.dispose()
    return [str(r[0]) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Campaign era partitioning")
    parser.add_argument("--max-gold", type=int, default=10)
    parser.add_argument("--min-era", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=1.0)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "campaign_era_partition.json"),
    )
    parser.add_argument(
        "--with-credits",
        action="store_true",
        help="Run operator credit to compute era-local statistics",
    )
    args = parser.parse_args()

    ids = asyncio.run(_all_campaign_ids())
    print(f"Partitioning {len(ids)} campaigns...", flush=True)

    campaigns = asyncio.run(load_campaign_era_metadata(ids))
    gold = select_gold_corpus(
        campaigns,
        min_era=args.min_era,
        max_size=args.max_gold,
        min_quality=args.min_quality,
    )
    archive = build_experience_archive(campaigns)

    partition = CampaignEraPartition(
        git_commit=current_git_commit(),
        eras=build_era_definitions(),
        campaigns=campaigns,
        gold_corpus=gold,
        archive=archive,
    )

    if args.with_credits:
        print("Computing era-local operator statistics...", flush=True)
        store = PolicyStore.load()
        candidate = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
        bundles = asyncio.run(load_bundles_from_db(ids))
        _, trace_ledger = harvest_from_bundles(bundles)
        snapshots = {b.campaign_id: b.snapshots for b in bundles}
        trees = {b.campaign_id: b.tree for b in bundles if b.tree}
        credits = build_difference_rewards(
            traces=trace_ledger,
            snapshots_by_campaign=snapshots,
            trees=trees,
            candidate_policy=candidate,
            baseline_policy=candidate,
        )
        era_stats = compute_era_local_statistics(
            partition=partition,
            traces=trace_ledger,
            credits=credits,
        )
        partition.era_local_stats = era_stats
        partition.cross_era_comparisons = compute_cross_era_comparisons(era_stats)

    partition.save()
    gold.save()

    payload = partition.to_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = payload["summary"]
    print(json.dumps(summary, indent=2))
    print(f"\nGold corpus ({len(gold.campaign_ids)} campaigns):")
    for entry in gold.entries:
        print(
            f"  {entry.campaign_id[:8]}  era={ERA_LABELS.get(entry.era_id)}  "
            f"quality={entry.quality_score}  trust={entry.trust_weight}  "
            f"confirmed={entry.total_confirmed}  paper={entry.has_paper}",
        )
    print(f"\nWrote {out}")
    print(f"Persisted {partition.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
