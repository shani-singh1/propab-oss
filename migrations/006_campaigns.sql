-- Migration 006: Research campaigns table
-- Supports the campaign model (Section 20 of ARCHITECTURE.md):
-- long-running research with HypothesisTree, baseline measurement,
-- and BreakthroughCriteria.

CREATE TABLE IF NOT EXISTS research_campaigns (
    id                          UUID PRIMARY KEY,
    question                    TEXT NOT NULL,
    status                      TEXT NOT NULL DEFAULT 'active',
    -- "active" | "paused" | "breakthrough" | "budget_exhausted"

    breakthrough_criteria_json  JSONB NOT NULL DEFAULT '{}',
    hypothesis_tree_json        JSONB,             -- full HypothesisTree state
    baseline_metric             FLOAT,             -- measured at campaign start
    best_metric                 FLOAT,             -- best achieved so far
    improvement_pct             FLOAT,             -- signed improvement vs baseline
    best_finding_json           JSONB,             -- ExperimentResult of best confirmed finding

    total_hypotheses            INT NOT NULL DEFAULT 0,
    total_confirmed             INT NOT NULL DEFAULT 0,
    compute_seconds_used        INT NOT NULL DEFAULT 0,
    compute_budget_seconds      INT NOT NULL DEFAULT 14400,  -- 4 hours default

    started_at                  TIMESTAMPTZ DEFAULT NOW(),
    last_checkpoint_at          TIMESTAMPTZ,
    completed_at                TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS campaigns_status_idx ON research_campaigns(status, started_at DESC);
