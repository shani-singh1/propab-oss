"""Lifetime knowledge Postgres tables (T1-001).

Per-entity upserts replace JSON last-writer-wins for cross-campaign compounding.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260704130000"
down_revision: Union[str, None] = "20260703120000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UPGRADE = [
    """
    CREATE TABLE IF NOT EXISTS lifetime_knowledge_claims (
        id TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        campaign_id UUID NOT NULL,
        claim_text TEXT NOT NULL,
        claim_type TEXT,
        confidence DOUBLE PRECISION,
        claim_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(domain, campaign_id, claim_text)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifetime_knowledge_theories (
        id TEXT PRIMARY KEY,
        domain TEXT NOT NULL,
        theory_text TEXT NOT NULL,
        supporting_campaign_ids UUID[] DEFAULT '{}',
        theory_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifetime_numerical_seeds (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        domain TEXT NOT NULL,
        campaign_id UUID NOT NULL,
        finding_type TEXT,
        claim TEXT NOT NULL,
        parameters JSONB DEFAULT '{}'::jsonb,
        source_node_ids UUID[] DEFAULT '{}',
        next_hypotheses JSONB DEFAULT '[]'::jsonb,
        seed_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifetime_knowledge_meta (
        meta_key TEXT PRIMARY KEY,
        meta_value JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifetime_meta_observations (
        campaign_id UUID PRIMARY KEY,
        observation_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lifetime_policy_fitness (
        policy_id TEXT NOT NULL,
        campaign_id UUID NOT NULL,
        record_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (policy_id, campaign_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS lifetime_claims_domain_idx ON lifetime_knowledge_claims (domain)",
    "CREATE INDEX IF NOT EXISTS lifetime_seeds_domain_idx ON lifetime_numerical_seeds (domain, campaign_id)",
]

_DOWNGRADE = [
    "DROP TABLE IF EXISTS lifetime_policy_fitness",
    "DROP TABLE IF EXISTS lifetime_meta_observations",
    "DROP TABLE IF EXISTS lifetime_knowledge_meta",
    "DROP TABLE IF EXISTS lifetime_numerical_seeds",
    "DROP TABLE IF EXISTS lifetime_knowledge_theories",
    "DROP TABLE IF EXISTS lifetime_knowledge_claims",
]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _UPGRADE:
        conn.execute(text(stmt))


def downgrade() -> None:
    conn = op.get_bind()
    for stmt in _DOWNGRADE:
        conn.execute(text(stmt))
