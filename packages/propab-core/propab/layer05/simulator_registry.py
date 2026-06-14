"""Simulator versioning — never overwrite, accept on improvement (fixes.md P5)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.simulator_acceptance import evaluate_simulator_acceptance

SIM_V1 = "sim_v1"
SIM_V2 = "sim_v2"
SIM_V3 = "sim_v3"
SIM_V4 = "sim_v4"
DEFAULT_ACTIVE = SIM_V1


def simulator_registry_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "simulator_registry.json"


@dataclass
class SimulatorVersionRecord:
    version: str
    metrics: dict[str, float]
    accepted: bool
    reason: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    component_bench: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulatorRegistry:
    active_version: str = DEFAULT_ACTIVE
    versions: dict[str, SimulatorVersionRecord] = field(default_factory=dict)

    def register(
        self,
        *,
        version: str,
        metrics: dict[str, float],
        component_bench: dict[str, Any] | None = None,
        directional_min: float = 0.80,
    ) -> SimulatorVersionRecord:
        """Never overwrite — only add if version is new; re-evaluate acceptance."""
        previous = self._previous_metrics(version)
        acceptance = evaluate_simulator_acceptance(
            current=metrics,
            previous=previous,
            version=version,
            directional_min=directional_min,
        )
        if version in self.versions:
            rec = self.versions[version]
            rec.metrics = metrics
            rec.accepted = acceptance.accepted
            rec.reason = acceptance.reason
            if component_bench:
                rec.component_bench = component_bench
            return rec

        rec = SimulatorVersionRecord(
            version=version,
            metrics=metrics,
            accepted=acceptance.accepted,
            reason=acceptance.reason,
            component_bench=component_bench or {},
        )
        self.versions[version] = rec
        if acceptance.accepted:
            self.active_version = version
        return rec

    def _previous_metrics(self, version: str) -> dict[str, float] | None:
        active = self.versions.get(self.active_version)
        if active and active.version != version:
            return active.metrics
        others = sorted(self.versions.values(), key=lambda r: r.created_at)
        if len(others) < 2:
            return None
        return others[-2].metrics

    def get(self, version: str) -> SimulatorVersionRecord | None:
        return self.versions.get(version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_version": self.active_version,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulatorRegistry:
        versions = {
            k: SimulatorVersionRecord(**v)
            for k, v in (data.get("versions") or {}).items()
        }
        return cls(
            active_version=str(data.get("active_version") or DEFAULT_ACTIVE),
            versions=versions,
        )

    def save(self, path: Path | None = None) -> Path:
        p = path or simulator_registry_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> SimulatorRegistry:
        p = path or simulator_registry_path()
        if not p.is_file():
            reg = cls()
            reg.versions[SIM_V1] = SimulatorVersionRecord(
                version=SIM_V1,
                metrics={"directional_agreement": 0.14, "mae_entropy": 1.44, "mae_closure": 0.49},
                accepted=True,
                reason="baseline_rules_only",
            )
            return reg
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()
