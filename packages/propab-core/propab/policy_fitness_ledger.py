"""Policy fitness ledger — predictions, observations, residuals, accept/reject."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings


def fitness_ledger_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "policy_fitness_ledger.json"


@dataclass
class FitnessRecord:
    policy_id: str
    campaign_id: str
    budget_bucket: str
    domain_bucket: str
    predictions: dict[str, float]
    observations: dict[str, float]
    residuals: dict[str, float]
    accept_or_reject: str
    detail: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyFitnessLedger:
    records: list[FitnessRecord] = field(default_factory=list)

    def record(self, rec: FitnessRecord) -> None:
        self.records = [
            r for r in self.records
            if not (r.policy_id == rec.policy_id and r.campaign_id == rec.campaign_id)
        ]
        self.records.append(rec)

    def for_policy(self, policy_id: str) -> list[FitnessRecord]:
        return [r for r in self.records if r.policy_id == policy_id]

    def to_dict(self) -> dict[str, Any]:
        return {"records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyFitnessLedger:
        recs = [FitnessRecord(**r) for r in (data.get("records") or [])]
        return cls(records=recs)

    def save(self, path: Path | None = None) -> Path:
        from propab.lifetime_postgres import lifetime_postgres_enabled, save_fitness_ledger

        if lifetime_postgres_enabled():
            save_fitness_ledger(self)
            return fitness_ledger_path()
        p = path or fitness_ledger_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> PolicyFitnessLedger:
        from propab.lifetime_postgres import lifetime_postgres_enabled, load_fitness_ledger

        if lifetime_postgres_enabled():
            try:
                return load_fitness_ledger()
            except Exception:
                pass
        p = path or fitness_ledger_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
