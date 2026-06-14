"""SimulatorBench — simulated vs historical trajectory metrics (fixes.md P2)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from propab.layer05.trajectories import CampaignTrajectories, EntropyTrajectory


@dataclass
class SimulatorBenchResult:
    mae_entropy: float
    rmse_entropy: float
    mae_closure: float
    rmse_closure: float
    directional_agreement: float
    threshold_crossing_error: float
    passed: bool
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _align_series(
    sim_values: list[float],
    obs_values: list[float],
) -> tuple[list[float], list[float]]:
    n = max(len(sim_values), len(obs_values), 1)
    if len(sim_values) < n:
        sim_values = sim_values + [sim_values[-1] if sim_values else 0.0] * (n - len(sim_values))
    if len(obs_values) < n:
        obs_values = obs_values + [obs_values[-1] if obs_values else 0.0] * (n - len(obs_values))
    return sim_values[:n], obs_values[:n]


def _mae(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return round(sum(abs(x - y) for x, y in zip(a, b)) / len(a), 4)


def _rmse(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return round(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)), 4)


def _directional_agreement(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(1, min(len(a), len(b))):
        da = a[i] - a[i - 1]
        db = b[i] - b[i - 1]
        if da == 0 and db == 0:
            agree += 1
        elif da * db > 0:
            agree += 1
        total += 1
    return round(agree / max(1, total), 4)


def _threshold_crossing_error(
    sim: EntropyTrajectory,
    obs: EntropyTrajectory,
) -> float:
    errors: list[float] = []
    for attr in ("cross_H_1_5_at", "cross_H_2_0_at"):
        s = getattr(sim, attr)
        o = getattr(obs, attr)
        if s is None and o is None:
            continue
        errors.append(abs(float(s or 999) - float(o or 999)))
    return round(sum(errors) / len(errors), 4) if errors else 0.0


def run_simulator_bench(
    *,
    simulated_entropy_points: list[dict[str, Any]],
    simulated_closure_values: list[float],
    observed_snapshots: list[dict[str, Any]],
    directional_threshold: float = 0.80,
    mae_entropy_max: float = 0.45,
    mae_closure_max: float = 0.12,
) -> SimulatorBenchResult:
    observed = CampaignTrajectories.from_snapshots(observed_snapshots)
    sim_entropy = EntropyTrajectory.from_points(simulated_entropy_points)
    obs_entropy_pts = [
        {"tested": s.get("tested"), "theme_entropy": s.get("theme_entropy")}
        for s in observed_snapshots
    ]
    obs_entropy = EntropyTrajectory.from_points(obs_entropy_pts)

    sim_h = [float(p.get("theme_entropy") or 0) for p in simulated_entropy_points]
    obs_h = [float(s.get("theme_entropy") or 0) for s in observed_snapshots]
    sim_h, obs_h = _align_series(sim_h, obs_h)

    obs_closure = [float(s.get("closure_ratio") or 0) for s in observed_snapshots]
    sim_c, obs_c = _align_series(simulated_closure_values, obs_closure)

    mae_e = _mae(sim_h, obs_h)
    rmse_e = _rmse(sim_h, obs_h)
    mae_c = _mae(sim_c, obs_c)
    rmse_c = _rmse(sim_c, obs_c)
    direction = _directional_agreement(sim_h, obs_h)
    thresh_err = _threshold_crossing_error(sim_entropy, obs_entropy)

    passed = (
        direction >= directional_threshold
        and mae_e <= mae_entropy_max
        and mae_c <= mae_closure_max
    )

    return SimulatorBenchResult(
        mae_entropy=mae_e,
        rmse_entropy=rmse_e,
        mae_closure=mae_c,
        rmse_closure=rmse_c,
        directional_agreement=direction,
        threshold_crossing_error=thresh_err,
        passed=passed,
        metrics={
            "simulated_entropy": sim_entropy.to_dict(),
            "observed_entropy": obs_entropy.to_dict(),
            "observed_closure": observed.closure.to_dict(),
        },
    )
