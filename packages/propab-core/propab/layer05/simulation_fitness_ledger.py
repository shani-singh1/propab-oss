"""Simulation fitness ledger — simulator residuals as first-class objects (fixes.md P0)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings

SIMULATOR_VERSION = "v1"


def simulation_fitness_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "simulation_fitness_ledger.json"


@dataclass
class SimulationFitnessRecord:
    simulator_version: str
    replay_campaign_id: str
    policy_id: str
    predicted_trajectory: dict[str, Any]
    observed_trajectory: dict[str, Any]
    residuals: dict[str, float]
    accept_or_reject: str
    budget_bucket: str = "3h"
    domain_bucket: str = "graphs"
    detail: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationFitnessLedger:
    accepted_simulator_version: str = SIMULATOR_VERSION
    records: list[SimulationFitnessRecord] = field(default_factory=list)

    def record(self, rec: SimulationFitnessRecord) -> None:
        self.records = [
            r for r in self.records
            if not (
                r.replay_campaign_id == rec.replay_campaign_id
                and r.policy_id == rec.policy_id
                and r.simulator_version == rec.simulator_version
            )
        ]
        self.records.append(rec)

    def for_campaign(self, campaign_id: str) -> list[SimulationFitnessRecord]:
        return [r for r in self.records if r.replay_campaign_id == campaign_id]

    def for_bucket(
        self,
        *,
        budget_bucket: str,
        domain_bucket: str,
        limit: int = 20,
    ) -> list[SimulationFitnessRecord]:
        rows = [
            r for r in self.records
            if r.budget_bucket == budget_bucket and r.domain_bucket == domain_bucket
        ]
        return rows[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_simulator_version": self.accepted_simulator_version,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationFitnessLedger:
        recs = [SimulationFitnessRecord(**r) for r in (data.get("records") or [])]
        return cls(
            accepted_simulator_version=str(
                data.get("accepted_simulator_version") or SIMULATOR_VERSION
            ),
            records=recs,
        )

    def save(self, path: Path | None = None) -> Path:
        p = path or simulation_fitness_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> SimulationFitnessLedger:
        p = path or simulation_fitness_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
