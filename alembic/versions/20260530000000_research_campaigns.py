"""Research campaigns table (folds migrations/006_campaigns.sql into the Alembic chain).

Revision ID: 20260530000000
Revises: 20260501000000
Create Date: 2026-05-30

Makes ``alembic upgrade head`` the single, complete schema source of truth: previously the
``research_campaigns`` table existed only as a raw SQL file applied by the Postgres init mount,
so it was missing from Alembic-managed databases. This revision applies the same DDL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260530000000"
down_revision: Union[str, None] = "20260501000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _sql_statements() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    sql_path = root / "migrations" / "006_campaigns.sql"
    raw = sql_path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
    full = "\n".join(lines)
    return [s.strip() for s in full.split(";") if s.strip()]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _sql_statements():
        conn.execute(text(stmt))


def downgrade() -> None:
    op.execute(text("DROP TABLE IF EXISTS research_campaigns CASCADE"))
