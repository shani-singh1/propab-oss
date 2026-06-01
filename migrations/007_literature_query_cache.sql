-- Migration 007: Query-scoped literature retrieval cache (domain-agnostic).
-- Stores which paper_ids were retrieved for a given research question so later
-- campaigns with a similar question can reuse them — never the global "last N papers".

CREATE TABLE IF NOT EXISTS literature_query_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash      TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    paper_ids       JSONB NOT NULL DEFAULT '[]',
    query_embedding JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS literature_query_cache_hash_idx
    ON literature_query_cache(query_hash);

CREATE INDEX IF NOT EXISTS literature_query_cache_created_idx
    ON literature_query_cache(created_at DESC);
