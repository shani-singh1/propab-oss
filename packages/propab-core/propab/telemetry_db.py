"""
Telemetry persistence — save/load :class:`HypothesisTrajectory` records.

Reuses the same ``async_sessionmaker`` engine as ``campaign_db``. Writes are an
**idempotent upsert keyed by (campaign_id, hypothesis_id)** so re-running the hook
at every checkpoint simply refreshes each row with the latest derived trajectory.

The scalar columns mirror the dataclass so the corpus is queryable directly
(``SELECT verdict, failure_reason … FROM hypothesis_trajectories``); the full
record is also stored verbatim in ``record_json`` so nothing is lost if the
dataclass gains fields ahead of the columns.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from propab.telemetry import HypothesisTrajectory

# Scalar columns persisted alongside record_json (order == INSERT column order).
_SCALAR_COLUMNS = (
    "campaign_id",
    "hypothesis_id",
    "parent_id",
    "generation",
    "round",
    "domain",
    "verdict",
    "confidence",
    "discovery_worthy",
    "was_novel",
    "literature_predicted",
    "llm_calls",
    "tool_calls",
    "code_runs",
    "tokens_in",
    "tokens_out",
    "duration_sec",
    "generator_strategy",
    "reasoning_strategy",
    "expansion_type",
    "experiment_informative",
    "failure_reason",
    "verifier_that_exposed_failure",
    "branch_outcome",
)


def _trajectory_to_params(t: HypothesisTrajectory) -> dict[str, Any]:
    d = t.to_dict()
    params = {col: d.get(col) for col in _SCALAR_COLUMNS}
    params["record_json"] = json.dumps(d)
    return params


def _row_to_trajectory(row: dict[str, Any]) -> HypothesisTrajectory:
    """Reconstruct from a DB row. Prefers the full ``record_json`` blob; falls
    back to the scalar columns when it is absent."""
    raw = row.get("record_json")
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, dict) and raw:
        return HypothesisTrajectory.from_dict(raw)
    return HypothesisTrajectory.from_dict({col: row.get(col) for col in _SCALAR_COLUMNS})


_UPSERT_SQL = text(
    """
    INSERT INTO hypothesis_trajectories (
        campaign_id, hypothesis_id, parent_id, generation, round, domain,
        verdict, confidence, discovery_worthy, was_novel, literature_predicted,
        llm_calls, tool_calls, code_runs, tokens_in, tokens_out, duration_sec,
        generator_strategy, reasoning_strategy, expansion_type,
        experiment_informative, failure_reason, verifier_that_exposed_failure,
        branch_outcome, record_json, updated_at
    ) VALUES (
        CAST(:campaign_id AS uuid), :hypothesis_id, :parent_id, :generation, :round, :domain,
        :verdict, :confidence, :discovery_worthy, :was_novel, :literature_predicted,
        :llm_calls, :tool_calls, :code_runs, :tokens_in, :tokens_out, :duration_sec,
        :generator_strategy, :reasoning_strategy, :expansion_type,
        :experiment_informative, :failure_reason, :verifier_that_exposed_failure,
        :branch_outcome, CAST(:record_json AS jsonb), NOW()
    )
    ON CONFLICT (campaign_id, hypothesis_id) DO UPDATE SET
        parent_id = EXCLUDED.parent_id,
        generation = EXCLUDED.generation,
        round = EXCLUDED.round,
        domain = EXCLUDED.domain,
        verdict = EXCLUDED.verdict,
        confidence = EXCLUDED.confidence,
        discovery_worthy = EXCLUDED.discovery_worthy,
        was_novel = EXCLUDED.was_novel,
        literature_predicted = EXCLUDED.literature_predicted,
        llm_calls = EXCLUDED.llm_calls,
        tool_calls = EXCLUDED.tool_calls,
        code_runs = EXCLUDED.code_runs,
        tokens_in = EXCLUDED.tokens_in,
        tokens_out = EXCLUDED.tokens_out,
        duration_sec = EXCLUDED.duration_sec,
        generator_strategy = EXCLUDED.generator_strategy,
        reasoning_strategy = EXCLUDED.reasoning_strategy,
        expansion_type = EXCLUDED.expansion_type,
        experiment_informative = EXCLUDED.experiment_informative,
        failure_reason = EXCLUDED.failure_reason,
        verifier_that_exposed_failure = EXCLUDED.verifier_that_exposed_failure,
        branch_outcome = EXCLUDED.branch_outcome,
        record_json = EXCLUDED.record_json,
        updated_at = NOW()
    """
)


async def save_trajectories(
    trajectories: list[HypothesisTrajectory],
    session_factory: async_sessionmaker,
) -> int:
    """Idempotently upsert trajectory records. Returns the number written."""
    if not trajectories:
        return 0
    async with session_factory() as db:
        for t in trajectories:
            await db.execute(_UPSERT_SQL, _trajectory_to_params(t))
        await db.commit()
    return len(trajectories)


async def db_load_campaign_events(
    campaign_id: str,
    session_factory: async_sessionmaker,
    *,
    limit: int = 20000,
) -> list[dict[str, Any]]:
    """Load a campaign's event stream for trajectory derivation.

    Unlike ``campaign_db.db_load_session_events_tail`` this SELECTs the top-level
    ``hypothesis_id`` — needed to attribute cost/verifier events (llm.response,
    tool.called, code.*) to their hypothesis, since those payloads do not embed it.
    """
    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT event_type, step, hypothesis_id, payload_json, created_at
                    FROM events
                    WHERE session_id = CAST(:id AS uuid)
                    ORDER BY created_at ASC
                    LIMIT :lim
                    """
                ),
                {"id": campaign_id, "lim": limit},
            )
        ).mappings().all()
    out: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        if d.get("hypothesis_id") is not None:
            d["hypothesis_id"] = str(d["hypothesis_id"])
        out.append(d)
    return out


async def load_trajectories(
    session_factory: async_sessionmaker,
    campaign_id: str | None = None,
) -> list[HypothesisTrajectory]:
    """Load trajectory records — all campaigns, or one when ``campaign_id`` is given."""
    async with session_factory() as db:
        if campaign_id is None:
            rows = (
                await db.execute(
                    text(
                        "SELECT * FROM hypothesis_trajectories "
                        "ORDER BY campaign_id, hypothesis_id"
                    )
                )
            ).mappings().all()
        else:
            rows = (
                await db.execute(
                    text(
                        "SELECT * FROM hypothesis_trajectories "
                        "WHERE campaign_id = CAST(:cid AS uuid) "
                        "ORDER BY hypothesis_id"
                    ),
                    {"cid": campaign_id},
                )
            ).mappings().all()
    return [_row_to_trajectory(dict(r)) for r in rows]
