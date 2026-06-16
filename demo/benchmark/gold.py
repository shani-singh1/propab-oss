"""P1/P5 — Gold corpus and baseline campaign references."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.operator_credit.campaign_era import GoldCorpus, gold_corpus_path

# Shared baseline for V1 residual batch and demo comparisons.
BASELINE_CAMPAIGN_ID = "3351d2ab-bbfe-4868-821b-0cca3951533d"


def load_gold_corpus(path: Path | None = None) -> GoldCorpus:
    return GoldCorpus.load(path or gold_corpus_path())


def gold_campaign_ids(path: Path | None = None) -> list[str]:
    return load_gold_corpus(path).campaign_ids


def require_gold_campaign(campaign_id: str, path: Path | None = None) -> None:
    """Reject non-gold campaigns for demo training/evaluation."""
    allowed = set(gold_campaign_ids(path))
    if campaign_id not in allowed:
        raise ValueError(
            f"Campaign {campaign_id} is not in gold corpus ({len(allowed)} allowed). "
            "Archive campaigns must not affect priors or demo benchmarks."
        )


def archive_count(path: Path | None = None) -> int:
    era_path = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge" / "campaign_eras.json"
    if era_path.is_file():
        data = json.loads(era_path.read_text(encoding="utf-8"))
        return len((data.get("archive") or {}).get("campaign_ids") or [])
    return 0


async def load_baseline_metrics() -> dict[str, Any]:
    """Load baseline campaign metrics from Postgres."""
    from sqlalchemy import text

    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    async with session_factory() as db:
        row = (
            await db.execute(
                text(
                    """
                    SELECT id::text, question, baseline_metric, best_metric,
                           improvement_pct, total_confirmed, total_hypotheses
                    FROM research_campaigns
                    WHERE id = CAST(:id AS uuid)
                    """
                ),
                {"id": BASELINE_CAMPAIGN_ID},
            )
        ).mappings().first()
    await engine.dispose()
    if not row:
        return {"campaign_id": BASELINE_CAMPAIGN_ID, "found": False}
    return {"found": True, **dict(row)}
