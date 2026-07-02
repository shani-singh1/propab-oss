"""Campaign health-metric tables and columns (ownership-contracts observability).

Adds the per-round / per-prior / per-campaign tables and the two research_campaigns
columns defined in ``propab_ownership_contracts.md`` so that every component's
health metric is queryable from Postgres:

- campaign_synthesis_events   : hypothesis duplicate rate, evidence-binding rejection
                                rate, belief citation integrity, belief stability
                                (one row per synthesis round)
- campaign_literature_priors  : literature citation verification rate (one row per prior build)
- campaign_audit_results      : verification artifact-gate precision (one row per post-campaign audit)
- research_campaigns.worker_experiment_success_rate / worker_utilization (per campaign)

Revision ID: 20260703120000
Revises: 20260703000000
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260703120000"
down_revision: Union[str, None] = "20260703000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_UPGRADE = [
    """
    CREATE TABLE IF NOT EXISTS campaign_synthesis_events (
        id                             BIGSERIAL PRIMARY KEY,
        campaign_id                    UUID NOT NULL,
        generation                     INTEGER NOT NULL DEFAULT 0,
        n_candidates_raw               INTEGER NOT NULL DEFAULT 0,
        n_added                        INTEGER NOT NULL DEFAULT 0,
        n_rejected_duplicate           INTEGER NOT NULL DEFAULT 0,
        binding_rejected_count         INTEGER NOT NULL DEFAULT 0,
        binding_accepted_count         INTEGER NOT NULL DEFAULT 0,
        falsifiability_rejected_count  INTEGER NOT NULL DEFAULT 0,
        belief_cap_rejected_count      INTEGER NOT NULL DEFAULT 0,
        hypothesis_duplicate_rate      DOUBLE PRECISION,
        evidence_binding_rejection_rate DOUBLE PRECISION,
        belief_citation_integrity      DOUBLE PRECISION,
        belief_stability               DOUBLE PRECISION,
        active_belief_statements       JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS synthesis_events_campaign_idx "
    "ON campaign_synthesis_events (campaign_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS campaign_literature_priors (
        id                          BIGSERIAL PRIMARY KEY,
        campaign_id                 UUID NOT NULL,
        citation_verification_rate  DOUBLE PRECISION,
        established_facts_count      INTEGER NOT NULL DEFAULT 0,
        verified_citation_count     INTEGER NOT NULL DEFAULT 0,
        created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS literature_priors_campaign_idx "
    "ON campaign_literature_priors (campaign_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS campaign_audit_results (
        id                       BIGSERIAL PRIMARY KEY,
        campaign_id              UUID NOT NULL,
        artifact_gate_precision  DOUBLE PRECISION,
        confirmed_findings_count INTEGER NOT NULL DEFAULT 0,
        survived_audit_count     INTEGER NOT NULL DEFAULT 0,
        created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS audit_results_campaign_idx "
    "ON campaign_audit_results (campaign_id, created_at)",
    "ALTER TABLE research_campaigns ADD COLUMN IF NOT EXISTS worker_experiment_success_rate DOUBLE PRECISION",
    "ALTER TABLE research_campaigns ADD COLUMN IF NOT EXISTS worker_utilization DOUBLE PRECISION",
]

_DOWNGRADE = [
    "ALTER TABLE research_campaigns DROP COLUMN IF EXISTS worker_utilization",
    "ALTER TABLE research_campaigns DROP COLUMN IF EXISTS worker_experiment_success_rate",
    "DROP TABLE IF EXISTS campaign_audit_results",
    "DROP TABLE IF EXISTS campaign_literature_priors",
    "DROP TABLE IF EXISTS campaign_synthesis_events",
]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _UPGRADE:
        conn.execute(text(stmt))


def downgrade() -> None:
    conn = op.get_bind()
    for stmt in _DOWNGRADE:
        conn.execute(text(stmt))
