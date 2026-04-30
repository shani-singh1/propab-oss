"""Long-running agent tables: research_rounds, session_checkpoints, agent_memory, kb_findings, session_budgets.

Revision ID: 20260501000000
Revises: 20260425120000
Create Date: 2026-05-01

Phase 3 (ARCHITECTURE.md §18): multi-round research loop schema additions.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260501000000"
down_revision: Union[str, None] = "20260425120000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UP_STATEMENTS = """
-- Research rounds: each pass of the multi-round hypothesis loop
CREATE TABLE IF NOT EXISTS research_rounds (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id          UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    round_number        INT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'running',
    confirmed_count     INT NOT NULL DEFAULT 0,
    refuted_count       INT NOT NULL DEFAULT 0,
    inconclusive_count  INT NOT NULL DEFAULT 0,
    marginal_return     FLOAT,
    budget_json         JSONB,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMPTZ,
    UNIQUE (session_id, round_number)
);
CREATE INDEX IF NOT EXISTS research_rounds_session_idx ON research_rounds(session_id);

-- Add round_id to hypotheses so each hypothesis is linked to the round it was generated in
ALTER TABLE hypotheses
    ADD COLUMN IF NOT EXISTS round_id UUID REFERENCES research_rounds(id),
    ADD COLUMN IF NOT EXISTS refinement_of UUID REFERENCES hypotheses(id),
    ADD COLUMN IF NOT EXISTS learned_from TEXT,
    ADD COLUMN IF NOT EXISTS promoted_to UUID REFERENCES hypotheses(id);

-- Add round_id to events
ALTER TABLE events
    ADD COLUMN IF NOT EXISTS round_id UUID REFERENCES research_rounds(id);

-- Session checkpoints: snapshot after each round for resumability
CREATE TABLE IF NOT EXISTS session_checkpoints (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    round_number    INT NOT NULL,
    ledger_json     JSONB NOT NULL,
    budget_json     JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, round_number)
);
CREATE INDEX IF NOT EXISTS session_checkpoints_session_idx ON session_checkpoints(session_id);

-- Session budget tracking
CREATE TABLE IF NOT EXISTS session_budgets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE UNIQUE,
    config_json     JSONB NOT NULL,
    consumed_json   JSONB NOT NULL DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Agent working memory snapshots: significance state + peer findings at each step
CREATE TABLE IF NOT EXISTS agent_memory (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hypothesis_id       UUID NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    step_index          INT NOT NULL,
    significance_json   JSONB,
    peer_findings_json  JSONB,
    results_summary     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS agent_memory_hypothesis_idx ON agent_memory(hypothesis_id);

-- Cross-session knowledge base: confirmed findings persist across sessions
CREATE TABLE IF NOT EXISTS kb_findings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES research_sessions(id),
    hypothesis_id   UUID REFERENCES hypotheses(id),
    text            TEXT NOT NULL,
    evidence        TEXT,
    confidence      FLOAT,
    embedding_id    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS kb_findings_session_idx ON kb_findings(session_id);

-- Extend tool_calls with significance columns
ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS significance_capable BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS produced_p_value FLOAT,
    ADD COLUMN IF NOT EXISTS produced_effect_size FLOAT;

-- Extend experiment_steps with significance snapshot column
ALTER TABLE experiment_steps
    ADD COLUMN IF NOT EXISTS significance_json JSONB;
"""

_DOWN_STATEMENTS = [
    "ALTER TABLE experiment_steps DROP COLUMN IF EXISTS significance_json",
    "ALTER TABLE tool_calls DROP COLUMN IF EXISTS produced_effect_size",
    "ALTER TABLE tool_calls DROP COLUMN IF EXISTS produced_p_value",
    "ALTER TABLE tool_calls DROP COLUMN IF EXISTS significance_capable",
    "DROP TABLE IF EXISTS kb_findings CASCADE",
    "DROP TABLE IF EXISTS agent_memory CASCADE",
    "DROP TABLE IF EXISTS session_budgets CASCADE",
    "DROP TABLE IF EXISTS session_checkpoints CASCADE",
    "ALTER TABLE events DROP COLUMN IF EXISTS round_id",
    "ALTER TABLE hypotheses DROP COLUMN IF EXISTS promoted_to",
    "ALTER TABLE hypotheses DROP COLUMN IF EXISTS learned_from",
    "ALTER TABLE hypotheses DROP COLUMN IF EXISTS refinement_of",
    "ALTER TABLE hypotheses DROP COLUMN IF EXISTS round_id",
    "DROP TABLE IF EXISTS research_rounds CASCADE",
]


def upgrade() -> None:
    conn = op.get_bind()
    statements = [s.strip() for s in _UP_STATEMENTS.split(";") if s.strip()]
    for stmt in statements:
        conn.execute(text(stmt))


def downgrade() -> None:
    conn = op.get_bind()
    for stmt in _DOWN_STATEMENTS:
        conn.execute(text(stmt))
