"""Add a top-level ``stop_reason`` column to research_campaigns.

Every campaign that stops records a meaningful enum stop reason (see
``propab.campaign.STOP_REASON_*``). It was previously stored only inside
``breakthrough_criteria_json._campaign_meta``; this promotes it to a queryable
column so campaign observability is readable directly from ``research_campaigns``
(Checklist 4 observability contract).

Revision ID: 20260703000000
Revises: 20260531000000
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "20260703000000"
down_revision: Union[str, None] = "20260531000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(text("ALTER TABLE research_campaigns ADD COLUMN IF NOT EXISTS stop_reason text"))
    # Backfill from the meta blob so historical rows expose their reason too.
    conn.execute(
        text(
            """
            UPDATE research_campaigns
            SET stop_reason = breakthrough_criteria_json #>> '{_campaign_meta,stop_reason}'
            WHERE stop_reason IS NULL
              AND breakthrough_criteria_json #>> '{_campaign_meta,stop_reason}' IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    op.execute(text("ALTER TABLE research_campaigns DROP COLUMN IF EXISTS stop_reason"))
