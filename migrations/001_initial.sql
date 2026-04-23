CREATE TABLE IF NOT EXISTS research_sessions (
    id              UUID PRIMARY KEY,
    question        TEXT NOT NULL,
    status          TEXT NOT NULL,
    stage           TEXT,
    prior_json      JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    authors         JSONB,
    abstract        TEXT,
    pdf_url         TEXT,
    sections_json   JSONB,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    status          TEXT
);

CREATE TABLE IF NOT EXISTS paper_citations (
    source_paper_id TEXT REFERENCES papers(id),
    cited_paper_id  TEXT,
    PRIMARY KEY (source_paper_id, cited_paper_id)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id               UUID PRIMARY KEY,
    session_id       UUID REFERENCES research_sessions(id),
    text             TEXT NOT NULL,
    test_methodology TEXT,
    scores_json      JSONB,
    rank             INT,
    status           TEXT,
    verdict          TEXT,
    confidence       FLOAT,
    evidence_summary TEXT,
    key_finding      TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_steps (
    id            UUID PRIMARY KEY,
    hypothesis_id UUID REFERENCES hypotheses(id),
    step_type     TEXT NOT NULL,
    step_index    INT,
    input_json    JSONB,
    output_json   JSONB,
    error_json    JSONB,
    duration_ms   INT,
    memory_mb     INT,
    timeout_sec   INT,
    summary       TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id            UUID PRIMARY KEY,
    step_id       UUID REFERENCES experiment_steps(id),
    hypothesis_id UUID REFERENCES hypotheses(id),
    tool_name     TEXT NOT NULL,
    domain        TEXT NOT NULL,
    params_json   JSONB,
    result_json   JSONB,
    success       BOOLEAN,
    duration_ms   INT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id            UUID PRIMARY KEY,
    session_id    UUID REFERENCES research_sessions(id),
    hypothesis_id UUID,
    call_purpose  TEXT,
    model         TEXT,
    prompt_text   TEXT NOT NULL,
    response_text TEXT NOT NULL,
    input_tokens  INT,
    output_tokens INT,
    duration_ms   INT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id              UUID PRIMARY KEY,
    session_id      UUID REFERENCES research_sessions(id),
    event_type      TEXT NOT NULL,
    source          TEXT,
    step            TEXT,
    hypothesis_id   UUID,
    parent_event_id UUID,
    payload_json    JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id, created_at);
