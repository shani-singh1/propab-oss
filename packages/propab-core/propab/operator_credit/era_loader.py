"""Load campaign metadata from Postgres for era partitioning."""
from __future__ import annotations

from typing import Any

from propab.operator_credit.campaign_era import (
    CampaignEraMetadata,
    build_campaign_metadata,
    current_git_commit,
)


async def load_campaign_metadata_rows(
    campaign_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load metadata rows for era classification."""
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    rows_out: list[dict[str, Any]] = []

    id_filter = ""
    params: dict[str, Any] = {}
    if campaign_ids:
        id_filter = "AND rc.id = ANY(CAST(:ids AS uuid[]))"
        params["ids"] = campaign_ids

    query = f"""
        SELECT
            rc.id::text AS campaign_id,
            rc.started_at,
            rc.total_confirmed,
            rc.total_hypotheses,
            rc.compute_seconds_used,
            rc.status,
            rc.question,
            li.payload_json->>'active_policy_id' AS policy_id,
            li.payload_json->>'budget_bucket' AS budget_bucket,
            li.payload_json->>'domain_bucket' AS domain_bucket,
            li.payload_json->'evaluation'->>'entropy_eval_mode' AS entropy_eval_mode,
            li.payload_json->'evaluation'->>'policy_generation' AS policy_generation,
            (SELECT COUNT(*) FROM events e
             WHERE e.session_id = rc.id AND e.event_type = 'paper.ready') AS paper_count,
            (SELECT COUNT(*) FROM events e
             WHERE e.session_id = rc.id AND e.step = 'campaign.frontier_snapshot') AS n_snapshots,
            (SELECT COUNT(*) FROM tool_calls tc
             JOIN hypotheses h ON h.id = tc.hypothesis_id
             WHERE h.session_id = rc.id) AS n_tool_calls,
            (SELECT MAX((e.payload_json->>'closure_ratio')::float)
             FROM events e
             WHERE e.session_id = rc.id AND e.step = 'campaign.frontier_snapshot'
               AND e.payload_json->>'closure_ratio' IS NOT NULL) AS max_closure_ratio
        FROM research_campaigns rc
        LEFT JOIN LATERAL (
            SELECT payload_json FROM events
            WHERE session_id = rc.id AND step = 'lifetime.ingested'
            ORDER BY created_at DESC LIMIT 1
        ) li ON true
        WHERE rc.hypothesis_tree_json IS NOT NULL
        {id_filter}
        ORDER BY rc.started_at ASC NULLS LAST
    """

    async with session_factory() as db:
        rows = (await db.execute(text(query), params)).mappings().all()
        commit = current_git_commit()
        for row in rows:
            d = dict(row)
            d["git_commit"] = commit
            d["simulator_version"] = "sim_v2"
            rows_out.append(d)

    await engine.dispose()
    return rows_out


async def load_campaign_era_metadata(
    campaign_ids: list[str] | None = None,
) -> list[CampaignEraMetadata]:
    rows = await load_campaign_metadata_rows(campaign_ids)
    return [build_campaign_metadata(r) for r in rows]
