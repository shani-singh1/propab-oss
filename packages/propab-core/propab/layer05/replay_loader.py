"""Load historical campaign artifacts for offline replay."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from propab.hypothesis_tree import HypothesisTree
from propab.meta_science import MetaScienceLedger
from propab.policy_fitness_ledger import PolicyFitnessLedger
from propab.policy_store import PolicyStore


@dataclass
class CampaignReplayBundle:
    campaign_id: str
    snapshots: list[dict[str, Any]]
    tree: HypothesisTree | None = None
    fitness_records: list[dict[str, Any]] = field(default_factory=list)
    observation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "n_snapshots": len(self.snapshots),
            "has_tree": self.tree is not None,
            "fitness_records": len(self.fitness_records),
            "observation": self.observation,
        }


def load_snapshots_from_json(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    for camp in data.get("campaigns") or []:
        cid = camp.get("campaign_id")
        traj = camp.get("trajectory") or []
        if cid and traj:
            out[cid] = traj
    return out


async def load_campaign_bundle(
    campaign_id: str,
    *,
    include_tree: bool = True,
) -> CampaignReplayBundle:
    from sqlalchemy import text

    from propab.config import settings
    from propab.db import create_engine, create_session_factory

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    snapshots: list[dict[str, Any]] = []
    tree: HypothesisTree | None = None

    async with session_factory() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT payload_json
                    FROM events
                    WHERE session_id = CAST(:id AS uuid)
                      AND step = 'campaign.frontier_snapshot'
                    ORDER BY created_at ASC
                    """
                ),
                {"id": campaign_id},
            )
        ).fetchall()
        for row in rows:
            payload = row[0]
            if isinstance(payload, str):
                payload = json.loads(payload)
            snapshots.append(payload if isinstance(payload, dict) else {})

        if include_tree:
            tree_row = (
                await db.execute(
                    text(
                        """
                        SELECT hypothesis_tree_json
                        FROM research_campaigns
                        WHERE id = CAST(:id AS uuid)
                        """
                    ),
                    {"id": campaign_id},
                )
            ).scalar_one_or_none()
            if tree_row:
                raw = tree_row if isinstance(tree_row, dict) else json.loads(tree_row)
                tree = HypothesisTree.from_dict(raw)

    await engine.dispose()

    meta = MetaScienceLedger.load()
    obs = meta.observation_for_campaign(campaign_id)
    fitness = PolicyFitnessLedger.load()
    records = [
        r.to_dict()
        for r in fitness.records
        if r.campaign_id == campaign_id
    ]

    return CampaignReplayBundle(
        campaign_id=campaign_id,
        snapshots=snapshots,
        tree=tree,
        fitness_records=records,
        observation=obs.to_dict() if obs else None,
    )


def load_bucket_context(
    *,
    domain_bucket: str = "graphs",
    budget_bucket: str = "3h",
) -> dict[str, Any]:
    store = PolicyStore.load()
    fitness = PolicyFitnessLedger.load()
    meta = MetaScienceLedger.load()
    acc = store.accepted_policy(domain_bucket=domain_bucket, budget_bucket=budget_bucket)
    cands = store.latest_candidate(domain_bucket=domain_bucket, budget_bucket=budget_bucket)
    bucket_obs = meta.observations_in_bucket(
        budget_bucket=budget_bucket,
        domain_bucket=domain_bucket,
    )
    return {
        "accepted_policy": acc.to_dict() if acc else None,
        "latest_candidate": cands.to_dict() if cands else None,
        "fitness_record_count": len(fitness.records),
        "bucket_observations": len(bucket_obs),
    }
