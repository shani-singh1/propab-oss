"""SimulationErrorLedger — residual error taxonomy (fixes.md P5)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.trajectories import CampaignTrajectories, EntropyTrajectory


def error_ledger_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "simulation_error_ledger.json"


@dataclass
class SimulationErrorRecord:
    campaign_id: str
    simulator_version: str
    magnitude_errors: int = 0
    direction_errors: int = 0
    plateau_errors: int = 0
    threshold_crossing_errors: int = 0
    n_points: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationErrorLedger:
    records: list[SimulationErrorRecord] = field(default_factory=list)

    def add(self, record: SimulationErrorRecord) -> None:
        self.records = [
            r for r in self.records
            if not (
                r.campaign_id == record.campaign_id
                and r.simulator_version == record.simulator_version
            )
        ]
        self.records.append(record)

    def summary(self) -> dict[str, int]:
        return {
            "magnitude_errors": sum(r.magnitude_errors for r in self.records),
            "direction_errors": sum(r.direction_errors for r in self.records),
            "plateau_errors": sum(r.plateau_errors for r in self.records),
            "threshold_crossing_errors": sum(r.threshold_crossing_errors for r in self.records),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationErrorLedger:
        records = [SimulationErrorRecord(**r) for r in (data.get("records") or [])]
        return cls(records=records)

    def save(self, path: Path | None = None) -> Path:
        p = path or error_ledger_path()
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: Path | None = None) -> SimulationErrorLedger:
        p = path or error_ledger_path()
        if not p.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError):
            return cls()


def classify_simulation_errors(
    *,
    campaign_id: str,
    simulator_version: str,
    simulated_entropy_points: list[dict[str, Any]],
    simulated_closure: list[float],
    observed_snapshots: list[dict[str, Any]],
    magnitude_threshold: float = 0.35,
) -> SimulationErrorRecord:
    obs_h = [float(s.get("theme_entropy") or 0) for s in observed_snapshots]
    sim_h = [float(p.get("theme_entropy") or 0) for p in simulated_entropy_points]
    n = min(len(obs_h), len(sim_h))
    details: list[dict[str, Any]] = []
    mag_err = dir_err = 0

    for i in range(n):
        delta = abs(sim_h[i] - obs_h[i])
        if delta > magnitude_threshold:
            mag_err += 1
            details.append({"type": "magnitude", "index": i, "delta": round(delta, 4)})

    for i in range(1, n):
        ds = sim_h[i] - sim_h[i - 1]
        do = obs_h[i] - obs_h[i - 1]
        if ds * do < 0 and not (ds == 0 and do == 0):
            dir_err += 1
            details.append({"type": "direction", "index": i, "sim_delta": round(ds, 4), "obs_delta": round(do, 4)})

    sim_traj = EntropyTrajectory.from_points(simulated_entropy_points)
    obs_traj = EntropyTrajectory.from_points([
        {"tested": s.get("tested"), "theme_entropy": s.get("theme_entropy")}
        for s in observed_snapshots
    ])
    plateau_err = 0
    if sim_traj.plateau_point is not None and obs_traj.plateau_point is not None:
        if abs(sim_traj.plateau_point - obs_traj.plateau_point) > 5:
            plateau_err = 1
            details.append({
                "type": "plateau",
                "sim": sim_traj.plateau_point,
                "obs": obs_traj.plateau_point,
            })

    thresh_err = 0
    for attr in ("cross_H_1_5_at", "cross_H_2_0_at"):
        s = getattr(sim_traj, attr)
        o = getattr(obs_traj, attr)
        if s is None and o is None:
            continue
        if abs(float(s or 999) - float(o or 999)) > 3:
            thresh_err += 1
            details.append({"type": "threshold_crossing", "field": attr, "sim": s, "obs": o})

    return SimulationErrorRecord(
        campaign_id=campaign_id,
        simulator_version=simulator_version,
        magnitude_errors=mag_err,
        direction_errors=dir_err,
        plateau_errors=plateau_err,
        threshold_crossing_errors=thresh_err,
        n_points=n,
        details=details[:50],
    )
