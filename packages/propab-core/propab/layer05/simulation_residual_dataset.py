"""SimulationResidualDataset — training rows for simulator evolution (fixes.md P0)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.trajectories import CampaignTrajectories
from propab.layer05.simulation_fitness_ledger import SimulationFitnessLedger


def residual_dataset_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "simulation_residual_dataset.json"


@dataclass
class SimulationResidualRow:
    replay_campaign_id: str
    simulator_version: str
    policy_id: str
    predicted_trajectories: dict[str, Any]
    observed_trajectories: dict[str, Any]
    residuals: dict[str, float]
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    budget_bucket: str = "3h"
    domain_bucket: str = "graphs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResidualDataset:
    rows: list[SimulationResidualRow] = field(default_factory=list)

    def add(self, row: SimulationResidualRow) -> None:
        self.rows = [
            r for r in self.rows
            if not (
                r.replay_campaign_id == row.replay_campaign_id
                and r.simulator_version == row.simulator_version
                and r.policy_id == row.policy_id
            )
        ]
        self.rows.append(row)

    def campaign_ids(self) -> list[str]:
        return list({r.replay_campaign_id for r in self.rows})

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [r.to_dict() for r in self.rows], "count": len(self.rows)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationResidualDataset:
        rows = [SimulationResidualRow(**r) for r in (data.get("rows") or [])]
        return cls(rows=rows)

    def save(self, path: Path | None = None) -> Path:
        p = path or residual_dataset_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> SimulationResidualDataset:
        p = path or residual_dataset_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()

    @classmethod
    def build_from_ledger(
        cls,
        ledger: SimulationFitnessLedger | None = None,
    ) -> SimulationResidualDataset:
        fitness = ledger or SimulationFitnessLedger.load()
        ds = cls()
        for rec in fitness.records:
            ds.add(SimulationResidualRow(
                replay_campaign_id=rec.replay_campaign_id,
                simulator_version=rec.simulator_version,
                policy_id=rec.policy_id,
                predicted_trajectories=rec.predicted_trajectory,
                observed_trajectories=rec.observed_trajectory,
                residuals=dict(rec.residuals),
                budget_bucket=rec.budget_bucket,
                domain_bucket=rec.domain_bucket,
            ))
        return ds

    @classmethod
    def build_from_trajectory_file(
        cls,
        path: Path | str,
        *,
        simulator_version: str = "observed",
        policy_id: str = "historical",
    ) -> SimulationResidualDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ds = cls()
        for camp in data.get("campaigns") or []:
            cid = camp.get("campaign_id")
            snaps = camp.get("trajectory") or []
            if not cid or not snaps:
                continue
            full_snaps = [_trajectory_point_to_snapshot(p) for p in snaps]
            observed = CampaignTrajectories.from_snapshots(full_snaps).to_dict()
            ds.add(SimulationResidualRow(
                replay_campaign_id=cid,
                simulator_version=simulator_version,
                policy_id=policy_id,
                predicted_trajectories={},
                observed_trajectories=observed,
                residuals={},
                snapshots=full_snaps,
            ))
        return ds

    @classmethod
    def build_merged(
        cls,
        *,
        trajectory_path: Path | str | None = None,
        ledger: SimulationFitnessLedger | None = None,
    ) -> SimulationResidualDataset:
        ds = cls()
        if trajectory_path and Path(trajectory_path).is_file():
            for row in cls.build_from_trajectory_file(trajectory_path).rows:
                ds.add(row)
        for row in cls.build_from_ledger(ledger).rows:
            if row.snapshots:
                ds.add(row)
            else:
                existing = next(
                    (r for r in ds.rows if r.replay_campaign_id == row.replay_campaign_id),
                    None,
                )
                if existing:
                    existing.predicted_trajectories = row.predicted_trajectories
                    existing.residuals = row.residuals
                    existing.simulator_version = row.simulator_version
                    existing.policy_id = row.policy_id
        return ds


def _trajectory_point_to_snapshot(p: dict[str, Any]) -> dict[str, Any]:
    tested = int(p.get("tested") or 0)
    return {
        "tested": tested,
        "executed": tested,
        "generated": tested + 1,
        "theme_entropy": float(p.get("theme_entropy") or 0),
        "closure_ratio": float(p.get("closure_ratio") or 0),
        "theme_histogram": p.get("theme_histogram") or {},
        "pending": 1,
        "frontier_size": 1,
    }
