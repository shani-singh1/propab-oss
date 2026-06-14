"""Simulator calibration v3 — grid search, LOO-CV, ensemble, error ledger."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from propab.config import settings
from propab.layer05.cross_validation import leave_one_out_evaluate
from propab.layer05.ensemble_simulator import SIM_V3
from propab.layer05.hyperparameter_search import grid_search_loo
from propab.layer05.hybrid_simulator import SIM_V2
from propab.layer05.replay_loader import load_snapshots_from_json
from propab.layer05.simulation_error_ledger import (
    SimulationErrorLedger,
    classify_simulation_errors,
)
from propab.layer05.simulator_acceptance import evaluate_v2_acceptance, evaluate_v3_acceptance
from propab.layer05.simulator_calibration import _enrich_snapshot
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.simulator_registry import SIM_V1, SimulatorRegistry
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.policy_store import PolicyStore


def hyperparams_path() -> Path:
    base = Path(settings.propab_data_dir).resolve() / "lifetime_knowledge"
    base.mkdir(parents=True, exist_ok=True)
    return base / "simulator_hyperparams.json"


@dataclass
class V3CalibrationReport:
    v2_grid: dict[str, Any]
    v3_grid: dict[str, Any]
    v2_loo: dict[str, Any]
    v3_loo: dict[str, Any]
    error_ledger: dict[str, Any]
    registry: dict[str, Any]
    v2_accepted: bool
    v3_accepted: bool
    meets_80pct_loo_gate: bool
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_campaigns(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    raw = load_snapshots_from_json(path)
    return {cid: [_enrich_snapshot(s) for s in snaps] for cid, snaps in raw.items()}


def _policy():
    store = PolicyStore.load()
    return store.accepted_policy(domain_bucket="graphs", budget_bucket="3h")


def run_v3_calibration(
    *,
    trajectory_path: Path | str | None = None,
    persist: bool = True,
) -> V3CalibrationReport:
    t0 = time.perf_counter()
    traj_path = Path(trajectory_path or "artifacts/entropy_trajectories.json")
    campaigns = _load_campaigns(traj_path) if traj_path.is_file() else {}
    policy = _policy()
    if not campaigns or policy is None:
        return V3CalibrationReport(
            v2_grid={}, v3_grid={}, v2_loo={}, v3_loo={},
            error_ledger={}, registry={}, v2_accepted=False,
            v3_accepted=False, meets_80pct_loo_gate=False, elapsed_ms=0,
        )

    v2_grid = grid_search_loo(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V2,
    )
    v2_hp = SimulatorHyperparams.from_dict(v2_grid.best_hyperparams)
    v2_loo = leave_one_out_evaluate(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V2,
        hyperparams=v2_hp,
    ).to_dict()

    v3_grid = grid_search_loo(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V3,
    )
    v3_hp = SimulatorHyperparams.from_dict(v3_grid.best_hyperparams)
    v3_loo = leave_one_out_evaluate(
        campaigns=campaigns,
        policy=policy,
        simulator_version=SIM_V3,
        hyperparams=v3_hp,
    ).to_dict()

    index = StateEmbeddingIndex().build_from_snapshots_map(campaigns)
    error_ledger = SimulationErrorLedger()
    for cid, snaps in campaigns.items():
        for version in (SIM_V2, SIM_V3):
            from propab.layer05.simulator_dispatch import simulate_for_version

            cal_index = StateEmbeddingIndex(
                entries=[e for e in index.entries if e.campaign_id != cid],
                max_tested=index.max_tested,
            )
            from propab.layer05.replay_state import SearchState

            state = SearchState.from_snapshot(snaps[0])
            steps = max(10, len(snaps) - 1)
            sim = simulate_for_version(
                version=version,
                state=state,
                policy=policy,
                steps=steps,
                index=cal_index,
                hyperparams=v2_hp if version == SIM_V2 else v3_hp,
            )
            err = classify_simulation_errors(
                campaign_id=cid,
                simulator_version=version,
                simulated_entropy_points=sim.entropy_points,
                simulated_closure=sim.closure_trajectory,
                observed_snapshots=snaps,
            )
            error_ledger.add(err)

    registry = SimulatorRegistry.load()
    v1_metrics = registry.versions.get(SIM_V1)
    v1_agg = v1_metrics.metrics if v1_metrics else {
        "directional_agreement": 0.14,
        "mae_entropy": 1.44,
        "mae_closure": 0.49,
    }

    v2_accept = evaluate_v2_acceptance(
        current=v2_loo["aggregate"],
        baseline_v1=v1_agg,
    )
    v2_rec = registry.register(
        version=SIM_V2,
        metrics=v2_loo["aggregate"],
        component_bench={"loo": v2_loo, "grid": v2_grid.to_dict()},
        directional_min=0.0,
    )
    v2_rec.accepted = v2_accept.accepted
    v2_rec.reason = v2_accept.reason
    if v2_accept.accepted:
        registry.active_version = SIM_V2

    v3_accept = evaluate_v3_acceptance(
        aggregate=v3_loo["aggregate"],
        loo_aggregate=v3_loo["aggregate"],
    )
    v3_rec = registry.register(
        version=SIM_V3,
        metrics=v3_loo["aggregate"],
        component_bench={"loo": v3_loo, "grid": v3_grid.to_dict()},
        directional_min=0.80,
    )
    v3_rec.accepted = v3_accept.accepted
    v3_rec.reason = v3_accept.reason
    if v3_accept.accepted:
        registry.active_version = SIM_V3

    hp_payload = {
        SIM_V2: v2_hp.to_dict(),
        SIM_V3: v3_hp.to_dict(),
    }
    if persist:
        hyperparams_path().write_text(json.dumps(hp_payload, indent=2), encoding="utf-8")
        index.save()
        error_ledger.save()
        registry.save()

    elapsed = (time.perf_counter() - t0) * 1000
    loo_dir = float(v3_loo.get("aggregate", {}).get("directional_agreement") or 0)
    return V3CalibrationReport(
        v2_grid=v2_grid.to_dict(),
        v3_grid=v3_grid.to_dict(),
        v2_loo=v2_loo,
        v3_loo=v3_loo,
        error_ledger=error_ledger.to_dict(),
        registry=registry.to_dict(),
        v2_accepted=v2_accept.accepted,
        v3_accepted=v3_accept.accepted,
        meets_80pct_loo_gate=loo_dir >= 0.80,
        elapsed_ms=round(elapsed, 2),
    )
