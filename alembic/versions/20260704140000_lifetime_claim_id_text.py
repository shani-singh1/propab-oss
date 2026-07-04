"""Fix lifetime_knowledge_claims.id to TEXT (kg-* ids are not UUIDs)."""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260704140000"
down_revision: Union[str, None] = "20260704130000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(text("ALTER TABLE lifetime_knowledge_claims ALTER COLUMN id DROP DEFAULT"))
    conn.execute(text("ALTER TABLE lifetime_knowledge_claims ALTER COLUMN id TYPE TEXT USING id::text"))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(text(
        "ALTER TABLE lifetime_knowledge_claims ALTER COLUMN id TYPE UUID USING id::uuid"
    ))
