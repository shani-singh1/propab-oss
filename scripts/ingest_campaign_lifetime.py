#!/usr/bin/env python3
"""Backfill lifetime knowledge store from a completed campaign in Postgres."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from propab.config import settings
from propab.db import create_engine, create_session_factory
from services.orchestrator.campaign_loop import db_load_campaign
from services.orchestrator.lifetime_knowledge import ingest_campaign


async def _run(campaign_id: str) -> int:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        campaign = await db_load_campaign(campaign_id, session_factory)
        if campaign is None:
            print(f"Campaign not found: {campaign_id}", file=sys.stderr)
            return 1
        report = ingest_campaign(campaign)
        print(json.dumps(report, indent=2))
        return 0
    finally:
        await engine.dispose()


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest campaign outcomes into lifetime knowledge.")
    p.add_argument("campaign_id")
    args = p.parse_args()
    return asyncio.run(_run(args.campaign_id))


if __name__ == "__main__":
    raise SystemExit(main())
