"""Direction error reduction calibration cycle (fixes.md P0–P5)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from propab.layer05.cross_validation import leave_one_out_evaluate
from propab.layer05.direction_error_dataset import DirectionErrorDataset
from propab.layer05.hyperparameter_search import grid_search_loo
from propab.layer05.replay_loader import load_snapshots_from_json
from propab.layer05.simulator_calibration import _enrich_snapshot
from propab.layer05.simulator_calibration_v3 import hyperparams_path
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.simulator_registry import SIM_V2, SimulatorRegistry
from propab.layer05.stage_simulators import SIM_V4
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.layer05.weak_campaign_analysis import (
    analyze_campaign_stages,
    analyze_weak_campaigns,
)
from propab.policy_store import PolicyStore


@dataclass
class DirectionCalibrationReport:
    baseline_v2_loo: dict[str, Any]
    v4_grid: dict[str, Any]
    v4_loo: dict[str, Any]
    direction_dataset_count: int
    weak_campaigns: dict[str, Any]
    direction_errors_before: int
    direction_errors_after: int
    meets_70pct_gate: bool
    direction_errors_dominant_before: bool
    direction_errors_dominant_after: bool
    registry: dict[str, Any]
    v4_accepted: bool
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_campaigns(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    raw = load_snapshots_from_json(path)
    return {cid: [_enrich_snapshot(s) for s in snaps] for cid, snaps in raw.items()}


def _policy():
    return PolicyStore.load().accepted_policy(domain_bucket="graphs", budget_bucket="3h")


def _load_v2_hyperparams() -> SimulatorHyperparams:
    p = hyperparams_path()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if SIM_V2 in data:
                return SimulatorHyperparams.from_dict(data[SIM_V2])
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return SimulatorHyperparams(retrieval_weight=1.0, k_neighbors=3, distance_metric="cosine")


def run_direction_calibration(
    *,
    trajectory_path: Path | str | None = None,
    persist: bool = True,
) -> DirectionCalibrationReport:
    t0 = time.perf_counter()
    traj_path = Path(trajectory_path or "artifacts/entropy_trajectories.json")
    campaigns = _load_campaigns(traj_path) if traj_path.is_file() else {}
    policy = _policy()
    if not campaigns or policy is None:
        return DirectionCalibrationReport(
            baseline_v2_loo={}, v4_grid={}, v4_loo={},
            direction_dataset_count=0, weak_campaigns={},
            direction_errors_before=0, direction_errors_after=0,
            meets_70pct_gate=False,
            direction_errors_dominant_before=True,
            direction_errors_dominant_after=True,
            registry={}, v4_accepted=False, elapsed_ms=0,
        )

    v2_hp = _load_v2_hyperparams()
    baseline_loo = leave_one_out_evaluate(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V2,
        hyperparams=v2_hp,
    )
    dir_errors_before = int(baseline_loo.aggregate.get("direction_errors") or 0)

    v4_grid = grid_search_loo(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V4,
        direction_weight=5.0,
        magnitude_weight=0.2,
    )
    v4_hp = SimulatorHyperparams.from_dict(v4_grid.best_hyperparams)
    v4_loo = leave_one_out_evaluate(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V4,
        hyperparams=v4_hp,
    )
    dir_errors_after = int(v4_loo.aggregate.get("direction_errors") or 0)

    direction_ds = DirectionErrorDataset()
    campaign_reports = []
    for cid, snaps in campaigns.items():
        train = {k: v for k, v in campaigns.items() if k != cid}
        index = StateEmbeddingIndex().build_from_snapshots_map(train)
        from propab.layer05.replay_state import SearchState
        from propab.layer05.simulator_dispatch import simulate_for_version

        state = SearchState.from_snapshot(snaps[0])
        steps = max(10, len(snaps) - 1)
        sim = simulate_for_version(
            version=SIM_V4,
            state=state,
            policy=policy,
            steps=steps,
            index=index,
            hyperparams=v4_hp,
        )
        fold_ds = DirectionErrorDataset.build_from_simulation(
            campaign_id=cid,
            simulator_version=SIM_V4,
            simulated_entropy_points=sim.entropy_points,
            observed_snapshots=snaps,
            index=index,
            hyperparams=v4_hp,
        )
        for row in fold_ds.rows:
            direction_ds.add(row)
        campaign_reports.append(analyze_campaign_stages(
            campaign_id=cid,
            simulated_entropy_points=sim.entropy_points,
            observed_snapshots=snaps,
            direction_dataset=direction_ds,
        ))

    weak = analyze_weak_campaigns(
        loo_folds=v4_loo.folds,
        campaign_reports=campaign_reports,
    )

    loo_dir = float(v4_loo.aggregate.get("directional_agreement") or 0)
    meets_70 = loo_dir >= 0.70
    mag_dom_before = dir_errors_before > 40
    mag_dom_after = dir_errors_after >= dir_errors_before

    registry = SimulatorRegistry.load()
    v4_rec = registry.register(
        version=SIM_V4,
        metrics=v4_loo.aggregate,
        component_bench={
            "loo": v4_loo.to_dict(),
            "grid": v4_grid.to_dict(),
            "weak_campaigns": weak.to_dict(),
        },
        directional_min=0.70,
    )
    v4_accepted = meets_70 and dir_errors_after < dir_errors_before
    v4_rec.accepted = v4_accepted
    v4_rec.reason = "direction_reduction" if v4_accepted else "below_70pct_or_no_improvement"
    if v4_accepted:
        registry.active_version = SIM_V4

    hp_payload: dict[str, Any] = {}
    hp_file = hyperparams_path()
    if hp_file.is_file():
        try:
            hp_payload = json.loads(hp_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            hp_payload = {}
    hp_payload[SIM_V4] = v4_hp.to_dict()

    if persist:
        direction_ds.save()
        hp_file.write_text(json.dumps(hp_payload, indent=2), encoding="utf-8")
        registry.save()

    elapsed = (time.perf_counter() - t0) * 1000
    return DirectionCalibrationReport(
        baseline_v2_loo=baseline_loo.to_dict(),
        v4_grid=v4_grid.to_dict(),
        v4_loo=v4_loo.to_dict(),
        direction_dataset_count=len(direction_ds.rows),
        weak_campaigns=weak.to_dict(),
        direction_errors_before=dir_errors_before,
        direction_errors_after=dir_errors_after,
        meets_70pct_gate=meets_70,
        direction_errors_dominant_before=mag_dom_before,
        direction_errors_dominant_after=dir_errors_after > 50,
        registry=registry.to_dict(),
        v4_accepted=v4_accepted,
        elapsed_ms=round(elapsed, 2),
    )
