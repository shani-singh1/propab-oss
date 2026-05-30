#!/usr/bin/env python3
"""Generate paper for a campaign that stopped before the paper phase (e.g. LLM timeout)."""
from __future__ import annotations

import argparse
import asyncio
import sys

from propab.config import settings
from propab.db import create_engine, create_redis, create_session_factory
from propab.events import EventEmitter
from propab.llm import LLMClient
from services.orchestrator.campaign_loop import (
    build_campaign_synthesis_payload,
    db_load_campaign,
    db_mark_research_session_completed,
    db_set_research_session_stage,
)
from services.orchestrator.paper import write_paper_minimal


async def _run(campaign_id: str) -> int:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    redis = await create_redis(settings.redis_url)
    emitter = EventEmitter(source="salvage_paper", redis=redis, session_factory=session_factory)
    llm = LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_secret,
        emitter=emitter,
        session_factory=session_factory,
    )
    try:
        campaign = await db_load_campaign(campaign_id, session_factory)
        if campaign is None:
            print(f"Campaign not found: {campaign_id}", file=sys.stderr)
            return 1
        async with session_factory() as db:
            from sqlalchemy import text

            row = (
                await db.execute(
                    text("SELECT prior_json FROM research_sessions WHERE id = :id"),
                    {"id": campaign_id},
                )
            ).scalar_one_or_none()
        prior = row if isinstance(row, dict) else {}
        await db_set_research_session_stage(campaign_id, session_factory, stage="campaign.paper")
        synthesis = build_campaign_synthesis_payload(campaign)
        await write_paper_minimal(
            session_id=campaign_id,
            session_factory=session_factory,
            emitter=emitter,
            llm=llm,
            question=campaign.question,
            prior=prior,
            synthesis=synthesis,
        )
        await db_mark_research_session_completed(campaign_id, session_factory)
        print(f"Paper written for campaign {campaign_id}")
        return 0
    finally:
        await redis.close()
        await engine.dispose()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("campaign_id")
    args = p.parse_args()
    return asyncio.run(_run(args.campaign_id))


if __name__ == "__main__":
    raise SystemExit(main())
