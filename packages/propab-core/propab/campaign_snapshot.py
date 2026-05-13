"""Serialize campaign + prior for offline replay (paper synthesis, debugging)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from propab.campaign import ResearchCampaign
from propab.config import settings

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 1


def snapshot_dir(campaign_id: str) -> Path:
    base = Path(settings.propab_data_dir).resolve() / "campaign_snapshots" / campaign_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_snapshot_document(*, phase: str, campaign: ResearchCampaign, prior: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": SNAPSHOT_VERSION,
        "phase": phase,
        "campaign": campaign.to_dict(),
        "prior": dict(prior or {}),
    }


def write_campaign_snapshot(phase: str, campaign: ResearchCampaign, prior: dict[str, Any]) -> Path | None:
    """Write JSON snapshot under ``PROPAB_DATA_DIR/campaign_snapshots/{id}/{phase}.json``."""
    try:
        path = snapshot_dir(campaign.id) / f"{phase}.json"
        doc = build_snapshot_document(phase=phase, campaign=campaign, prior=prior)
        path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        logger.info("[campaign %s] wrote snapshot %s", campaign.id, path)
        return path
    except OSError as exc:
        logger.warning("[campaign %s] snapshot write failed: %s", campaign.id, exc)
        return None


def read_snapshot(path: Path) -> tuple[dict[str, Any], ResearchCampaign, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if int(raw.get("version", 0) or 0) != SNAPSHOT_VERSION:
        logger.warning("Snapshot %s has unexpected version %r", path, raw.get("version"))
    camp = ResearchCampaign.from_dict(raw["campaign"])
    prior = raw.get("prior") if isinstance(raw.get("prior"), dict) else {}
    return raw, camp, prior
