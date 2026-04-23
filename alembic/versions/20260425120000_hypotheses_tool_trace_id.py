"""Add tool_trace_id to hypotheses for experiment contract (ARCHITECTURE §7.1).

Revision ID: 20260425120000
Revises: 20260424120000
Create Date: 2026-04-25
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20260425120000"
down_revision: Union[str, None] = "20260424120000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("hypotheses", sa.Column("tool_trace_id", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("hypotheses", "tool_trace_id")
