"""
Structured storage for claims, tables, and open problems, plus the coverage
bookkeeping ``GET /coverage`` reports on.

Real backend: Postgres via ``asyncpg``. Fallback: an in-memory store used
when ``postgres_backend != "postgres"`` — the default, so the service is
runnable with zero infrastructure. asyncpg is imported lazily inside methods
so merely constructing the store (choosing backend="memory") never requires
the dependency to be installed correctly configured.
"""
from __future__ import annotations

import datetime
from typing import Any

from services.literature.app.models import ExtractedClaim, OpenProblem, TabulatedSequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS literature_claims (
    claim_id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    status TEXT NOT NULL,
    text TEXT NOT NULL,
    verbatim TEXT NOT NULL,
    source_doi TEXT,
    source_title TEXT,
    source_authors TEXT,
    source_year INT,
    source_url TEXT,
    location TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_literature_claims_domain ON literature_claims (domain_id);

CREATE TABLE IF NOT EXISTS literature_tables (
    id SERIAL PRIMARY KEY,
    domain_id TEXT NOT NULL,
    description TEXT,
    index_variable TEXT,
    value_variable TEXT,
    values JSONB,
    max_index DOUBLE PRECISION,
    min_index DOUBLE PRECISION,
    source_doi TEXT,
    source_title TEXT,
    source_year INT,
    location TEXT,
    is_in_appendix BOOLEAN,
    is_in_supplementary BOOLEAN,
    extraction_confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_literature_tables_domain ON literature_tables (domain_id);

CREATE TABLE IF NOT EXISTS literature_open_problems (
    id SERIAL PRIMARY KEY,
    domain_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    source_doi TEXT,
    stated_by TEXT,
    year INT,
    context TEXT,
    computationally_approachable BOOLEAN,
    approachable_angle TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_literature_open_problems_domain ON literature_open_problems (domain_id);

CREATE TABLE IF NOT EXISTS literature_papers_indexed (
    domain_id TEXT NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    indexed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (domain_id, source, external_id)
);
"""


class InMemoryStructuredStore:
    def __init__(self) -> None:
        self._claims: dict[str, dict[str, list[ExtractedClaim]]] = {}
        self._tables: dict[str, list[TabulatedSequence]] = {}
        self._open_problems: dict[str, list[OpenProblem]] = {}
        self._papers: dict[str, set[str]] = {}
        self._last_updated: dict[str, str] = {}

    async def init(self) -> None:
        return None

    async def save_claims(self, domain_id: str, claims: list[ExtractedClaim]) -> None:
        bucket = self._claims.setdefault(domain_id, [])
        existing_ids = {c.claim_id for c in bucket}
        bucket.extend(c for c in claims if c.claim_id not in existing_ids)
        self._touch(domain_id)

    async def save_tables(self, domain_id: str, tables: list[TabulatedSequence]) -> None:
        self._tables.setdefault(domain_id, []).extend(tables)
        self._touch(domain_id)

    async def save_open_problems(self, domain_id: str, problems: list[OpenProblem]) -> None:
        existing = {p.statement for p in self._open_problems.get(domain_id, [])}
        bucket = self._open_problems.setdefault(domain_id, [])
        bucket.extend(p for p in problems if p.statement not in existing)
        self._touch(domain_id)

    async def mark_paper_indexed(self, domain_id: str, source: str, external_id: str) -> None:
        self._papers.setdefault(domain_id, set()).add(f"{source}:{external_id}")
        self._touch(domain_id)

    async def get_claims(self, domain_id: str) -> list[ExtractedClaim]:
        return list(self._claims.get(domain_id, []))

    async def get_tables(self, domain_id: str) -> list[TabulatedSequence]:
        return list(self._tables.get(domain_id, []))

    async def get_open_problems(self, domain_id: str) -> list[OpenProblem]:
        return list(self._open_problems.get(domain_id, []))

    async def coverage(self) -> list[dict[str, Any]]:
        domains = set(self._claims) | set(self._tables) | set(self._open_problems) | set(self._papers)
        return [
            {
                "domain_id": d,
                "papers_indexed": len(self._papers.get(d, set())),
                "claims_indexed": len(self._claims.get(d, [])),
                "last_updated": self._last_updated.get(d, ""),
            }
            for d in sorted(domains)
        ]

    def _touch(self, domain_id: str) -> None:
        self._last_updated[domain_id] = datetime.datetime.now(datetime.timezone.utc).isoformat()


class PostgresStructuredStore:
    def __init__(self, *, database_url: str) -> None:
        self._database_url = database_url
        self._pool = None

    async def _ensure_pool(self):
        if self._pool is not None:
            return self._pool
        import asyncpg

        self._pool = await asyncpg.create_pool(self._database_url)
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        return self._pool

    async def init(self) -> None:
        await self._ensure_pool()

    async def save_claims(self, domain_id: str, claims: list[ExtractedClaim]) -> None:
        if not claims:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO literature_claims
                    (claim_id, domain_id, claim_type, status, text, verbatim,
                     source_doi, source_title, source_authors, source_year, source_url, location)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (claim_id) DO NOTHING
                """,
                [
                    (c.claim_id, domain_id, c.claim_type, c.status, c.text, c.verbatim,
                     c.source_doi, c.source_title, c.source_authors, c.source_year, c.source_url, c.location)
                    for c in claims
                ],
            )

    async def save_tables(self, domain_id: str, tables: list[TabulatedSequence]) -> None:
        if not tables:
            return
        import json

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO literature_tables
                    (domain_id, description, index_variable, value_variable, values,
                     max_index, min_index, source_doi, source_title, source_year, location,
                     is_in_appendix, is_in_supplementary, extraction_confidence)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                """,
                [
                    (domain_id, t.description, t.index_variable, t.value_variable, json.dumps(t.values),
                     t.max_index, t.min_index, t.source_doi, t.source_title, t.source_year, t.location,
                     t.is_in_appendix, t.is_in_supplementary, t.extraction_confidence)
                    for t in tables
                ],
            )

    async def save_open_problems(self, domain_id: str, problems: list[OpenProblem]) -> None:
        if not problems:
            return
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO literature_open_problems
                    (domain_id, statement, source_doi, stated_by, year, context,
                     computationally_approachable, approachable_angle)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                """,
                [
                    (domain_id, p.statement, p.source_doi, p.stated_by, p.year, p.context,
                     p.computationally_approachable, p.approachable_angle)
                    for p in problems
                ],
            )

    async def mark_paper_indexed(self, domain_id: str, source: str, external_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO literature_papers_indexed (domain_id, source, external_id)
                VALUES ($1,$2,$3)
                ON CONFLICT (domain_id, source, external_id) DO NOTHING
                """,
                domain_id, source, external_id,
            )

    async def get_claims(self, domain_id: str) -> list[ExtractedClaim]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM literature_claims WHERE domain_id = $1", domain_id)
        return [
            ExtractedClaim(
                text=r["text"], claim_type=r["claim_type"], status=r["status"], verbatim=r["verbatim"],
                source_doi=r["source_doi"] or "", source_title=r["source_title"] or "",
                source_authors=r["source_authors"] or "", source_year=r["source_year"] or 0,
                source_url=r["source_url"] or "", location=r["location"] or "", claim_id=r["claim_id"],
            )
            for r in rows
        ]

    async def get_tables(self, domain_id: str) -> list[TabulatedSequence]:
        import json

        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM literature_tables WHERE domain_id = $1", domain_id)
        out = []
        for r in rows:
            values = r["values"]
            if isinstance(values, str):
                values = json.loads(values)
            out.append(
                TabulatedSequence(
                    description=r["description"] or "", index_variable=r["index_variable"] or "n",
                    value_variable=r["value_variable"] or "", values=values or {},
                    max_index=r["max_index"], min_index=r["min_index"],
                    source_doi=r["source_doi"] or "", source_title=r["source_title"] or "",
                    source_year=r["source_year"] or 0, location=r["location"] or "",
                    is_in_appendix=r["is_in_appendix"] or False, is_in_supplementary=r["is_in_supplementary"] or False,
                    extraction_confidence=r["extraction_confidence"] or 1.0,
                )
            )
        return out

    async def get_open_problems(self, domain_id: str) -> list[OpenProblem]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM literature_open_problems WHERE domain_id = $1", domain_id)
        return [
            OpenProblem(
                statement=r["statement"], source_doi=r["source_doi"] or "", stated_by=r["stated_by"] or "",
                year=r["year"] or 0, context=r["context"] or "",
                computationally_approachable=r["computationally_approachable"] or False,
                approachable_angle=r["approachable_angle"] or "",
            )
            for r in rows
        ]

    async def coverage(self) -> list[dict[str, Any]]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT domain_id,
                       count(DISTINCT source || ':' || external_id) FILTER (WHERE true) AS papers,
                       max(indexed_at) AS last_updated
                FROM literature_papers_indexed GROUP BY domain_id
                """
            )
            claim_counts = await conn.fetch(
                "SELECT domain_id, count(*) AS n FROM literature_claims GROUP BY domain_id"
            )
        claim_map = {r["domain_id"]: r["n"] for r in claim_counts}
        return [
            {
                "domain_id": r["domain_id"],
                "papers_indexed": r["papers"],
                "claims_indexed": claim_map.get(r["domain_id"], 0),
                "last_updated": r["last_updated"].isoformat() if r["last_updated"] else "",
            }
            for r in rows
        ]


def build_structured_store(*, backend: str, database_url: str):
    if backend == "postgres":
        return PostgresStructuredStore(database_url=database_url)
    return InMemoryStructuredStore()
