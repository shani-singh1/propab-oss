"""
Campaign persistence — the single source of truth for reading/writing a
``ResearchCampaign`` to Postgres.

This lives in ``propab-core`` (not in a service package) so that the API, the
worker, and the orchestrator can all persist and reload campaign state without
importing each other. It contains **only** persistence: no campaign-loop
business logic, no LLM calls, no service-specific code.

The full campaign lifecycle is resumable from the ``research_campaigns`` row
alone — ``db_load_campaign(campaign_id)`` reconstructs a complete
``ResearchCampaign`` (belief state, tree, budget, stop reason) from Postgres.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.campaign import ResearchCampaign


def _db_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a float from a DB row without treating 0.0 as missing (unlike ``x or 0.0``)."""
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_started_at(iso_s: str) -> datetime:
    """asyncpg expects datetime for TIMESTAMPTZ, not an ISO string."""
    if not iso_s:
        return datetime.now(tz=UTC)
    s = iso_s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _campaign_meta_blob(campaign: ResearchCampaign) -> dict[str, Any]:
    """Extra campaign fields stored inside breakthrough_criteria_json._campaign_meta."""
    return {
        "belief_state": campaign.belief_state.to_dict(),
        "max_hypotheses_cap": campaign.max_hypotheses_cap,
        "stop_reason": campaign.stop_reason,
        "seed_source": campaign.seed_source,
        "anomaly_artifacts_dir": campaign.anomaly_artifacts_dir,
        "policy_mode": campaign.policy_mode,
        "checkpoint_every": campaign.checkpoint_every,
    }


def _breakthrough_criteria_for_db(campaign: ResearchCampaign) -> dict[str, Any]:
    blob = campaign.breakthrough_criteria.to_dict()
    blob["_campaign_meta"] = _campaign_meta_blob(campaign)
    return blob


def _apply_campaign_meta_from_db(data: dict[str, Any], bc: dict[str, Any]) -> None:
    meta = bc.pop("_campaign_meta", None) if isinstance(bc, dict) else None
    if not isinstance(meta, dict):
        return
    if meta.get("belief_state") is not None:
        data["belief_state"] = meta["belief_state"]
    for key in (
        "max_hypotheses_cap",
        "stop_reason",
        "seed_source",
        "anomaly_artifacts_dir",
        "policy_mode",
        "checkpoint_every",
    ):
        if key in meta:
            data[key] = meta[key]


async def db_save_campaign(campaign: ResearchCampaign, session_factory: async_sessionmaker) -> None:
    """Upsert the full campaign state to research_campaigns.

    ``stop_reason`` is persisted both inside ``breakthrough_criteria_json._campaign_meta``
    (for full reconstruction) and as a top-level column (so it is queryable/visible
    directly in ``research_campaigns`` — Checklist 4 observability contract).
    """
    now = datetime.now(tz=UTC).isoformat()
    campaign.last_checkpoint = now
    async with session_factory() as db:
        await db.execute(
            text("""
                INSERT INTO research_campaigns (
                    id, question, status, stop_reason, breakthrough_criteria_json,
                    hypothesis_tree_json, baseline_metric, best_metric,
                    improvement_pct, best_finding_json, total_hypotheses,
                    total_confirmed, compute_seconds_used, compute_budget_seconds,
                    started_at, last_checkpoint_at
                ) VALUES (
                    :id, :question, :status, :stop_reason,
                    CAST(:breakthrough_criteria_json AS jsonb),
                    CAST(:hypothesis_tree_json AS jsonb),
                    :baseline_metric, :best_metric, :improvement_pct,
                    CAST(:best_finding_json AS jsonb),
                    :total_hypotheses, :total_confirmed,
                    :compute_seconds_used, :compute_budget_seconds,
                    :started_at, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    question = EXCLUDED.question,
                    status = EXCLUDED.status,
                    stop_reason = EXCLUDED.stop_reason,
                    breakthrough_criteria_json = EXCLUDED.breakthrough_criteria_json,
                    hypothesis_tree_json = EXCLUDED.hypothesis_tree_json,
                    baseline_metric = EXCLUDED.baseline_metric,
                    best_metric = EXCLUDED.best_metric,
                    improvement_pct = EXCLUDED.improvement_pct,
                    best_finding_json = EXCLUDED.best_finding_json,
                    total_hypotheses = EXCLUDED.total_hypotheses,
                    total_confirmed = EXCLUDED.total_confirmed,
                    compute_seconds_used = EXCLUDED.compute_seconds_used,
                    compute_budget_seconds = EXCLUDED.compute_budget_seconds,
                    started_at = EXCLUDED.started_at,
                    last_checkpoint_at = NOW()
            """),
            {
                "id": campaign.id,
                "question": campaign.question,
                "status": campaign.status,
                "stop_reason": campaign.stop_reason,
                "breakthrough_criteria_json": json.dumps(_breakthrough_criteria_for_db(campaign)),
                "hypothesis_tree_json": json.dumps(campaign.hypothesis_tree.to_dict()),
                "baseline_metric": campaign.baseline_metric,
                "best_metric": campaign.best_metric,
                "improvement_pct": campaign.improvement_pct,
                "best_finding_json": json.dumps(campaign.best_finding or {}),
                "total_hypotheses": campaign.total_hypotheses,
                "total_confirmed": campaign.total_confirmed,
                "compute_seconds_used": int(campaign.elapsed_seconds()),
                "compute_budget_seconds": campaign.compute_budget_seconds,
                "started_at": _parse_started_at(campaign.started_at),
            },
        )
        await db.commit()


async def db_load_session_events_tail(
    session_id: str,
    session_factory: async_sessionmaker,
    *,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    async with session_factory() as db:
        rows = (
            await db.execute(
                text("""
                    SELECT step, event_type, payload_json, created_at
                    FROM events
                    WHERE session_id = CAST(:id AS uuid)
                    ORDER BY created_at ASC
                    LIMIT :lim
                """),
                {"id": session_id, "lim": limit},
            )
        ).mappings().all()
    return [dict(r) for r in rows]


async def db_load_campaign(campaign_id: str, session_factory: async_sessionmaker) -> ResearchCampaign | None:
    """Load a campaign from the DB. Returns None if not found."""
    async with session_factory() as db:
        row = (await db.execute(
            text("SELECT * FROM research_campaigns WHERE id = CAST(:id AS uuid)"),
            {"id": campaign_id},
        )).mappings().one_or_none()
    if row is None:
        return None
    bc_raw = row["breakthrough_criteria_json"] or {}
    bc = dict(bc_raw) if isinstance(bc_raw, dict) else {}
    data = {
        "id": str(row["id"]),
        "question": row["question"],
        "status": row["status"],
        "breakthrough_criteria": {k: v for k, v in bc.items() if k != "_campaign_meta"},
        "hypothesis_tree": row["hypothesis_tree_json"] or {},
        "baseline_metric": _db_float(row, "baseline_metric", 0.0),
        "best_metric": _db_float(row, "best_metric", 0.0),
        "improvement_pct": _db_float(row, "improvement_pct", 0.0),
        "best_finding": row.get("best_finding_json") or None,
        "total_hypotheses": row.get("total_hypotheses") or 0,
        "total_confirmed": row.get("total_confirmed") or 0,
        "compute_budget_seconds": row.get("compute_budget_seconds") or 14400,
        "compute_seconds_used": row.get("compute_seconds_used") or 0,
        "started_at": str(row.get("started_at") or ""),
        "last_checkpoint": str(row.get("last_checkpoint_at") or ""),
    }
    _apply_campaign_meta_from_db(data, bc)
    # The top-level column is authoritative when present (it is what external
    # queries/dashboards read); fall back to the meta value otherwise.
    col_stop_reason = row.get("stop_reason")
    if col_stop_reason:
        data["stop_reason"] = col_stop_reason
    return ResearchCampaign.from_dict(data)
