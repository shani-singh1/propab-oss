"""Simulator calibration cycle — dataset, hybrid sim, component bench, registry."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from propab.layer05.component_bench import run_component_bench_suite
from propab.layer05.ensemble_simulator import SIM_V3
from propab.layer05.stage_simulators import SIM_V4
from propab.layer05.hybrid_simulator import SIM_V2, _branching_series, _saturation_series
from propab.layer05.replay_loader import load_snapshots_from_json
from propab.layer05.simulation_residual_dataset import SimulationResidualDataset
from propab.layer05.simulator_bench import run_simulator_bench
from propab.layer05.simulator_dispatch import simulate_for_version
from propab.layer05.simulator_registry import SIM_V1, SimulatorRegistry
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.policy_store import PolicyStore


@dataclass
class CalibrationCampaignResult:
    campaign_id: str
    simulator_version: str
    directional_agreement: float
    mae_entropy: float
    mae_closure: float
    component_suite: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationReport:
    simulator_version: str
    n_campaigns: int
    elapsed_ms: float
    aggregate: dict[str, float]
    campaigns: list[dict[str, Any]]
    component_weakest: str
    registry: dict[str, Any]
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_campaign(
    *,
    campaign_id: str,
    snapshots: list[dict[str, Any]],
    index: StateEmbeddingIndex,
    simulator_version: str = SIM_V2,
    policy_id: str | None = None,
    exclude_self: bool = True,
) -> CalibrationCampaignResult:
    store = PolicyStore.load()
    policy = store.get_policy(policy_id) if policy_id else None
    policy = policy or store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")
    if policy is None:
        raise ValueError("No policy available for simulator calibration")
    from propab.layer05.replay_state import SearchState

    state = SearchState.from_snapshot(snapshots[0])
    steps = max(10, len(snapshots) - 1)
    cal_index = (
        StateEmbeddingIndex(
            entries=[e for e in index.entries if e.campaign_id != campaign_id],
            max_tested=index.max_tested,
        )
        if exclude_self and simulator_version in (SIM_V2, SIM_V3, SIM_V4)
        else index
    )

    sim = simulate_for_version(
        version=simulator_version,
        state=state,
        policy=policy,
        steps=steps,
        index=cal_index,
        exclude_campaign_id=campaign_id if exclude_self else None,
    )

    bench = run_simulator_bench(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure_values=sim.closure_trajectory,
        observed_snapshots=snapshots,
    )
    pseudo_snaps = [
        {
            "generated": int(sim.entropy_points[i].get("tested") or i),
            "tested": int(sim.entropy_points[i].get("tested") or i),
            "theme_histogram": sim.theme_saturation if isinstance(sim.theme_saturation, dict) else {},
        }
        for i in range(len(sim.closure_trajectory))
    ]
    branch_sim = _branching_series(pseudo_snaps, steps)
    sat_sim = _saturation_series(pseudo_snaps, steps)

    suite = run_component_bench_suite(
        simulated_entropy_points=sim.entropy_points,
        simulated_closure=sim.closure_trajectory,
        simulated_branching=branch_sim,
        simulated_saturation=sat_sim,
        observed_snapshots=snapshots,
    )

    return CalibrationCampaignResult(
        campaign_id=campaign_id,
        simulator_version=simulator_version,
        directional_agreement=bench.directional_agreement,
        mae_entropy=bench.mae_entropy,
        mae_closure=bench.mae_closure,
        component_suite=suite.to_dict(),
        passed=bench.passed,
    )


def run_calibration_cycle(
    *,
    trajectory_path: Path | str | None = None,
    simulator_version: str = SIM_V2,
    persist: bool = True,
) -> CalibrationReport:
    t0 = time.perf_counter()
    traj_path = trajectory_path or Path("artifacts/entropy_trajectories.json")
    by_id = load_snapshots_from_json(traj_path) if Path(traj_path).is_file() else {}

    dataset = SimulationResidualDataset.build_merged(trajectory_path=traj_path)
    if persist:
        dataset.save()

    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    if persist:
        index.save()

    store = PolicyStore.load()
    policy = store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")

    results: list[CalibrationCampaignResult] = []
    for cid, snaps in by_id.items():
        full_snaps = [_enrich_snapshot(s) for s in snaps]
        results.append(calibrate_campaign(
            campaign_id=cid,
            snapshots=full_snaps,
            index=index,
            simulator_version=simulator_version,
            policy_id=policy.id if policy else None,
        ))

    if not results:
        return CalibrationReport(
            simulator_version=simulator_version,
            n_campaigns=0,
            elapsed_ms=0,
            aggregate={},
            campaigns=[],
            component_weakest="none",
            registry={},
            accepted=False,
        )

    agg_dir = sum(r.directional_agreement for r in results) / len(results)
    agg_mae_e = sum(r.mae_entropy for r in results) / len(results)
    agg_mae_c = sum(r.mae_closure for r in results) / len(results)
    weakest_counts: dict[str, int] = {}
    for r in results:
        w = r.component_suite.get("weakest_component", "entropy")
        weakest_counts[w] = weakest_counts.get(w, 0) + 1
    component_weakest = max(weakest_counts, key=weakest_counts.get)

    registry = SimulatorRegistry.load()
    rec = registry.register(
        version=simulator_version,
        metrics={
            "directional_agreement": round(agg_dir, 4),
            "mae_entropy": round(agg_mae_e, 4),
            "mae_closure": round(agg_mae_c, 4),
        },
        component_bench={
            "weakest_component": component_weakest,
            "per_campaign": [r.component_suite for r in results],
        },
    )
    if persist:
        registry.save()

    elapsed = (time.perf_counter() - t0) * 1000
    return CalibrationReport(
        simulator_version=simulator_version,
        n_campaigns=len(results),
        elapsed_ms=round(elapsed, 2),
        aggregate={
            "directional_agreement": round(agg_dir, 4),
            "mae_entropy": round(agg_mae_e, 4),
            "mae_closure": round(agg_mae_c, 4),
        },
        campaigns=[r.to_dict() for r in results],
        component_weakest=component_weakest,
        registry=registry.to_dict(),
        accepted=rec.accepted,
    )


def _enrich_snapshot(p: dict[str, Any]) -> dict[str, Any]:
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
