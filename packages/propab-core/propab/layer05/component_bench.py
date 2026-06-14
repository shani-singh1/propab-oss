"""ComponentBench V2 — per-trajectory calibration metrics (fixes.md P2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from propab.layer05.bench_metrics import component_bench_result, mae
from propab.layer05.trajectories import (
    BranchingTrajectory,
    CampaignTrajectories,
    ClosureTrajectory,
    EntropyTrajectory,
    ThemeSaturationTrajectory,
)


@dataclass
class ComponentBenchResult:
    name: str
    mae: float
    rmse: float
    directional_agreement: float
    scalar_residuals: dict[str, float]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scalar_residuals(predicted: dict, observed: dict, keys: tuple[str, ...]) -> dict[str, float]:
    return {
        k: round(float(predicted.get(k) or 0) - float(observed.get(k) or 0), 4)
        for k in keys
    }


def entropy_bench_v2(
    *,
    simulated_points: list[dict[str, Any]],
    observed_snapshots: list[dict[str, Any]],
) -> ComponentBenchResult:
    sim_traj = EntropyTrajectory.from_points(simulated_points)
    obs_pts = [
        {"tested": s.get("tested"), "theme_entropy": s.get("theme_entropy")}
        for s in observed_snapshots
    ]
    obs_traj = EntropyTrajectory.from_points(obs_pts)
    sim_h = [float(p.get("theme_entropy") or 0) for p in simulated_points]
    obs_h = [float(s.get("theme_entropy") or 0) for s in observed_snapshots]
    raw = component_bench_result(
        component="entropy",
        simulated=sim_h,
        observed=obs_h,
        scalar_residuals=_scalar_residuals(
            sim_traj.to_dict(), obs_traj.to_dict(),
            ("start", "mid", "end", "growth_rate"),
        ),
    )
    return ComponentBenchResult(
        name="EntropyBenchV2",
        mae=raw["mae"],
        rmse=raw["rmse"],
        directional_agreement=raw["directional_agreement"],
        scalar_residuals=raw["scalar_residuals"],
        metrics={"simulated": sim_traj.to_dict(), "observed": obs_traj.to_dict()},
    )


def closure_bench_v2(
    *,
    simulated_closure: list[float],
    observed_snapshots: list[dict[str, Any]],
) -> ComponentBenchResult:
    obs_vals = [float(s.get("closure_ratio") or 0) for s in observed_snapshots]
    sim_traj = ClosureTrajectory.from_values(simulated_closure)
    obs_traj = ClosureTrajectory.from_points([
        {"tested": s.get("tested"), "closure_ratio": s.get("closure_ratio")}
        for s in observed_snapshots
    ])
    raw = component_bench_result(
        component="closure",
        simulated=simulated_closure,
        observed=obs_vals,
        scalar_residuals=_scalar_residuals(
            sim_traj.to_dict(), obs_traj.to_dict(),
            ("start", "mid", "end"),
        ),
    )
    return ComponentBenchResult(
        name="ClosureBenchV2",
        mae=raw["mae"],
        rmse=raw["rmse"],
        directional_agreement=raw["directional_agreement"],
        scalar_residuals=raw["scalar_residuals"],
        metrics={"simulated": sim_traj.to_dict(), "observed": obs_traj.to_dict()},
    )


def branching_bench_v2(
    *,
    simulated_branching: list[float],
    observed_snapshots: list[dict[str, Any]],
) -> ComponentBenchResult:
    obs_traj = BranchingTrajectory.from_snapshots(observed_snapshots)
    obs_vals: list[float] = [obs_traj.start]
    for i in range(1, len(observed_snapshots)):
        prev = int(observed_snapshots[i - 1].get("generated") or observed_snapshots[i - 1].get("tested") or 0)
        cur = int(observed_snapshots[i].get("generated") or observed_snapshots[i].get("tested") or 0)
        obs_vals.append(float(max(0, cur - prev)))
    while len(simulated_branching) < len(obs_vals):
        simulated_branching.append(simulated_branching[-1] if simulated_branching else 0)
    raw = component_bench_result(
        component="branching",
        simulated=simulated_branching[: len(obs_vals)],
        observed=obs_vals,
        scalar_residuals={"mean_delta": round(
            (sum(simulated_branching[: len(obs_vals)]) / max(1, len(obs_vals)))
            - obs_traj.growth_rate, 4
        )},
    )
    return ComponentBenchResult(
        name="BranchingBenchV2",
        mae=raw["mae"],
        rmse=raw["rmse"],
        directional_agreement=raw["directional_agreement"],
        scalar_residuals=raw["scalar_residuals"],
        metrics={"simulated_mean": sum(simulated_branching) / max(1, len(simulated_branching)), "observed": obs_traj.to_dict()},
    )


def theme_bench_v2(
    *,
    simulated_saturation: list[float],
    observed_snapshots: list[dict[str, Any]],
) -> ComponentBenchResult:
    obs_traj = ThemeSaturationTrajectory.from_snapshots(observed_snapshots)
    obs_vals: list[float] = []
    for s in observed_snapshots:
        hist = s.get("theme_histogram") or {}
        total = sum(hist.values()) or 1
        obs_vals.append(max(hist.values()) / total if hist else obs_traj.start)
    if not obs_vals:
        obs_vals = [obs_traj.start, obs_traj.mid, obs_traj.end]
    while len(simulated_saturation) < len(obs_vals):
        simulated_saturation.append(simulated_saturation[-1] if simulated_saturation else 0.5)
    raw = component_bench_result(
        component="theme_saturation",
        simulated=simulated_saturation[: len(obs_vals)],
        observed=obs_vals,
        scalar_residuals=_scalar_residuals(
            {"start": simulated_saturation[0], "end": simulated_saturation[-1]},
            obs_traj.to_dict(),
            ("start", "end"),
        ),
    )
    return ComponentBenchResult(
        name="ThemeBenchV2",
        mae=raw["mae"],
        rmse=raw["rmse"],
        directional_agreement=raw["directional_agreement"],
        scalar_residuals=raw["scalar_residuals"],
        metrics={"observed": obs_traj.to_dict()},
    )


@dataclass
class ComponentBenchSuite:
    entropy: ComponentBenchResult
    closure: ComponentBenchResult
    branching: ComponentBenchResult
    theme: ComponentBenchResult
    weakest_component: str
    aggregate_directional: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy": self.entropy.to_dict(),
            "closure": self.closure.to_dict(),
            "branching": self.branching.to_dict(),
            "theme": self.theme.to_dict(),
            "weakest_component": self.weakest_component,
            "aggregate_directional": self.aggregate_directional,
        }


def run_component_bench_suite(
    *,
    simulated_entropy_points: list[dict[str, Any]],
    simulated_closure: list[float],
    simulated_branching: list[float],
    simulated_saturation: list[float],
    observed_snapshots: list[dict[str, Any]],
) -> ComponentBenchSuite:
    ent = entropy_bench_v2(
        simulated_points=simulated_entropy_points,
        observed_snapshots=observed_snapshots,
    )
    clo = closure_bench_v2(
        simulated_closure=simulated_closure,
        observed_snapshots=observed_snapshots,
    )
    br = branching_bench_v2(
        simulated_branching=simulated_branching,
        observed_snapshots=observed_snapshots,
    )
    th = theme_bench_v2(
        simulated_saturation=simulated_saturation,
        observed_snapshots=observed_snapshots,
    )
    components = {
        "entropy": ent,
        "closure": clo,
        "branching": br,
        "theme_saturation": th,
    }
    weakest = min(components, key=lambda k: components[k].directional_agreement)
    agg_dir = round(
        sum(c.directional_agreement for c in components.values()) / 4, 4
    )
    return ComponentBenchSuite(
        entropy=ent,
        closure=clo,
        branching=br,
        theme=th,
        weakest_component=weakest,
        aggregate_directional=agg_dir,
    )
