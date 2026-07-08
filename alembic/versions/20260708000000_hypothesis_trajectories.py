"""Hypothesis-trajectory telemetry table (the per-hypothesis research corpus).

Persists one structured research-trajectory record per hypothesis across every
campaign so the accumulated corpus (thousands of future campaigns on the FIXED
engine) can answer meta-questions like "which generators produce confirmed
hypotheses?", "which failures repeat?", "where is compute wasted?".

Idempotent upsert key: (campaign_id, hypothesis_id). Scalar columns mirror the
``propab.telemetry.HypothesisTrajectory`` dataclass for direct SQL querying;
``record_json`` stores the full record verbatim.

Revision ID: 20260708000000
Revises: 20260704140000
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260708000000"
down_revision: Union[str, None] = "20260704140000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_UPGRADE = [
    """
    CREATE TABLE IF NOT EXISTS hypothesis_trajectories (
        id                             BIGSERIAL PRIMARY KEY,
        campaign_id                    UUID NOT NULL,
        hypothesis_id                  TEXT NOT NULL,
        parent_id                      TEXT,
        generation                     INTEGER,
        round                          INTEGER,
        domain                         TEXT,
        verdict                        TEXT,
        confidence                     DOUBLE PRECISION,
        discovery_worthy               BOOLEAN,
        was_novel                      BOOLEAN,
        literature_predicted           BOOLEAN,
        llm_calls                      INTEGER,
        tool_calls                     INTEGER,
        code_runs                      INTEGER,
        tokens_in                      INTEGER,
        tokens_out                     INTEGER,
        duration_sec                   DOUBLE PRECISION,
        generator_strategy             TEXT,
        reasoning_strategy             TEXT,
        expansion_type                 TEXT,
        experiment_informative         BOOLEAN,
        failure_reason                 TEXT,
        verifier_that_exposed_failure  TEXT,
        branch_outcome                 TEXT,
        record_json                    JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT hypothesis_trajectories_campaign_hyp_uq
            UNIQUE (campaign_id, hypothesis_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS hypothesis_trajectories_campaign_idx "
    "ON hypothesis_trajectories (campaign_id)",
    "CREATE INDEX IF NOT EXISTS hypothesis_trajectories_verdict_idx "
    "ON hypothesis_trajectories (verdict)",
    "CREATE INDEX IF NOT EXISTS hypothesis_trajectories_domain_idx "
    "ON hypothesis_trajectories (domain)",
]

_DOWNGRADE = [
    "DROP TABLE IF EXISTS hypothesis_trajectories CASCADE",
]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _UPGRADE:
        conn.execute(text(stmt))


def downgrade() -> None:
    conn = op.get_bind()
    for stmt in _DOWNGRADE:
        conn.execute(text(stmt))
