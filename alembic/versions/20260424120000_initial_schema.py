"""Initial Propab schema (mirrors migrations/001_initial.sql for Alembic).

Revision ID: 20260424120000
Revises:
Create Date: 2026-04-24

Phase 1 (ARCHITECTURE.md §16): Postgres schema + migrations (Alembic).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260424120000"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _sql_statements() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    sql_path = root / "migrations" / "001_initial.sql"
    raw = sql_path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
    full = "\n".join(lines)
    return [s.strip() for s in full.split(";") if s.strip()]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _sql_statements():
        conn.execute(text(stmt))


def downgrade() -> None:
    for tbl in (
        "events",
        "llm_calls",
        "tool_calls",
        "experiment_steps",
        "hypotheses",
        "paper_citations",
        "papers",
        "research_sessions",
    ):
        op.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
