"""Leave-one-campaign-out validation (fixes.md P4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from propab.layer05.direction_loss import count_direction_errors
from propab.layer05.ensemble_simulator import SIM_V3, simulate_search_ensemble
from propab.layer05.hybrid_simulator import SIM_V2, simulate_search_hybrid
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import simulate_search
from propab.layer05.simulator_bench import run_simulator_bench
from propab.layer05.stage_simulators import SIM_V4, simulate_search_stage_aware
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.policy_record import PolicyRecord


@dataclass
class LOOFoldResult:
    held_out_campaign_id: str
    directional_agreement: float
    mae_entropy: float
    mae_closure: float
    rmse_entropy: float
    direction_errors: int = 0
    direction_error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LOOReport:
    simulator_version: str
    hyperparams: dict[str, Any]
    folds: list[dict[str, Any]]
    aggregate: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _simulate_fold(
    *,
    version: str,
    state: SearchState,
    policy: PolicyRecord,
    index: StateEmbeddingIndex,
    steps: int,
    hyperparams: SimulatorHyperparams,
    snapshots: list[dict[str, Any]],
):
    if version == SIM_V3:
        return simulate_search_ensemble(
            state=state,
            policy=policy,
            index=index,
            steps=steps,
            hyperparams=hyperparams,
            query_snapshots=snapshots[:3],
        )
    if version == SIM_V2:
        return simulate_search_hybrid(
            state=state,
            policy=policy,
            index=index,
            steps=steps,
            hyperparams=hyperparams.for_v2(),
            query_snapshots=snapshots[:3],
        )
    if version == SIM_V4:
        return simulate_search_stage_aware(
            state=state,
            policy=policy,
            index=index,
            steps=steps,
            hyperparams=hyperparams,
            query_snapshots=snapshots,
        )
    return simulate_search(state=state, policy=policy, steps=steps)


def leave_one_out_evaluate(
    *,
    campaigns: dict[str, list[dict[str, Any]]],
    policy: PolicyRecord,
    simulator_version: str,
    hyperparams: SimulatorHyperparams | None = None,
) -> LOOReport:
    hp = hyperparams or SimulatorHyperparams()
    folds: list[LOOFoldResult] = []

    for held_out, snaps in campaigns.items():
        train_map = {cid: s for cid, s in campaigns.items() if cid != held_out}
        index = StateEmbeddingIndex().build_from_snapshots_map(train_map)
        state = SearchState.from_snapshot(snaps[0])
        steps = max(10, len(snaps) - 1)
        sim = _simulate_fold(
            version=simulator_version,
            state=state,
            policy=policy,
            index=index,
            steps=steps,
            hyperparams=hp,
            snapshots=snaps,
        )
        bench = run_simulator_bench(
            simulated_entropy_points=sim.entropy_points,
            simulated_closure_values=sim.closure_trajectory,
            observed_snapshots=snaps,
        )
        sim_h = [float(p.get("theme_entropy") or 0) for p in sim.entropy_points]
        obs_h = [float(s.get("theme_entropy") or 0) for s in snaps]
        dir_errs = count_direction_errors(sim_h, obs_h)
        n_steps = max(1, min(len(sim_h), len(obs_h)) - 1)
        folds.append(LOOFoldResult(
            held_out_campaign_id=held_out,
            directional_agreement=bench.directional_agreement,
            mae_entropy=bench.mae_entropy,
            mae_closure=bench.mae_closure,
            rmse_entropy=bench.rmse_entropy,
            direction_errors=dir_errs,
            direction_error_rate=round(dir_errs / n_steps, 4),
        ))

    n = len(folds) or 1
    return LOOReport(
        simulator_version=simulator_version,
        hyperparams=hp.to_dict(),
        folds=[f.to_dict() for f in folds],
        aggregate={
            "directional_agreement": round(
                sum(f.directional_agreement for f in folds) / n, 4
            ),
            "mae_entropy": round(sum(f.mae_entropy for f in folds) / n, 4),
            "mae_closure": round(sum(f.mae_closure for f in folds) / n, 4),
            "rmse_entropy": round(sum(f.rmse_entropy for f in folds) / n, 4),
            "direction_errors": sum(f.direction_errors for f in folds),
            "direction_error_rate": round(
                sum(f.direction_error_rate for f in folds) / n, 4
            ),
        },
    )
