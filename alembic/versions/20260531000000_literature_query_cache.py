"""Literature query-scoped retrieval cache (migrations/007_literature_query_cache.sql).

Revision ID: 20260531000000
Revises: 20260530000000
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260531000000"
down_revision: Union[str, None] = "20260530000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _sql_statements() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    raw = (root / "migrations" / "007_literature_query_cache.sql").read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("--")]
    full = "\n".join(lines)
    return [s.strip() for s in full.split(";") if s.strip()]


def upgrade() -> None:
    conn = op.get_bind()
    for stmt in _sql_statements():
        conn.execute(text(stmt))


def downgrade() -> None:
    op.execute(text("DROP TABLE IF EXISTS literature_query_cache CASCADE"))
