"""Local stage simulators — mixture of experts (fixes.md P4)."""
from __future__ import annotations

from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory
from propab.layer05.hybrid_simulator import (
    _retrieve_component_series,
    simulate_search_hybrid,
)
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import SimulationResult, simulate_search
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.stage_aware_predictors import StageAwareEntropyPredictor
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.policy_record import PolicyRecord
from propab.search_policy import SearchPolicy

SIM_V4 = "sim_v4"


class ColdStartSimulator:
    retrieval_weight = 0.65
    k_neighbors = 1


class GrowthSimulator:
    retrieval_weight = 0.85
    k_neighbors = 3


class PlateauSimulator:
    retrieval_weight = 0.75
    k_neighbors = 5


def simulate_search_stage_aware(
    *,
    state: SearchState,
    policy: PolicyRecord | SearchPolicy,
    index: StateEmbeddingIndex,
    steps: int = 30,
    hyperparams: SimulatorHyperparams | None = None,
    query_snapshots: list[dict[str, Any]] | None = None,
) -> SimulationResult:
    """
    Mixture of experts: stage-aware entropy + hybrid closure retrieval.
    """
    hp = hyperparams or SimulatorHyperparams()
    hybrid = simulate_search_hybrid(
        state=state,
        policy=policy,
        index=index,
        steps=steps,
        hyperparams=hp.for_v2(),
        query_snapshots=query_snapshots,
    )
    neighbors = index.nearest(
        state,
        k=max(hp.k_neighbors, 5),
        hyperparams=hp,
        query_snapshots=query_snapshots,
    )
    if not neighbors:
        return hybrid

    hybrid_h = [float(p.get("theme_entropy") or 0) for p in hybrid.entropy_points]
    rules_h = [float(p.get("theme_entropy") or 0) for p in simulate_search(
        state=state, policy=policy, steps=steps,
    ).entropy_points]
    predictor = StageAwareEntropyPredictor()
    stage_h = predictor.predict(
        neighbors,
        steps=steps,
        rules_series=rules_h,
        query_snapshots=query_snapshots,
    )
    n = steps + 1
    while len(hybrid_h) < n:
        hybrid_h.append(hybrid_h[-1] if hybrid_h else 0)
    while len(stage_h) < n:
        stage_h.append(stage_h[-1] if stage_h else 0)
    from propab.layer05.trajectory_stages import stage_at_index

    blended_h: list[float] = []
    for i in range(n):
        stage = stage_at_index(i, steps)
        if stage == "growth":
            blended_h.append(round(0.55 * hybrid_h[i] + 0.45 * stage_h[i], 4))
        elif stage == "plateau":
            blended_h.append(round(0.65 * hybrid_h[i] + 0.35 * stage_h[i], 4))
        else:
            blended_h.append(hybrid_h[i])

    retrieved = _retrieve_component_series(neighbors[: hp.k_neighbors], steps=steps)
    rules_c = hybrid.closure_trajectory
    r_clo = retrieved["closure"]
    n = steps + 1
    while len(rules_c) < n:
        rules_c.append(rules_c[-1] if rules_c else 0)
    while len(r_clo) < n:
        r_clo.append(r_clo[-1] if r_clo else 0)
    blended_c = [
        round(0.3 * rules_c[i] + 0.7 * r_clo[i], 4)
        for i in range(n)
    ]

    tested0 = state.tested_count
    entropy_points = [
        {"tested": tested0 + i, "theme_entropy": blended_h[i]}
        for i in range(n)
    ]
    summary = summarize_entropy_trajectory(entropy_points)
    sat = retrieved["saturation"][-1] if retrieved.get("saturation") else 0.5

    return SimulationResult(
        policy_id=getattr(policy, "id", "search_policy"),
        steps=steps,
        entropy_trajectory=summary.to_dict(),
        entropy_points=entropy_points,
        closure_trajectory=blended_c,
        branching_factor=round(
            sum(retrieved.get("branching", [1.0])) / max(1, len(retrieved.get("branching", [1]))),
            4,
        ),
        theme_saturation={"dominant": round(sat, 4)},
        expected_compute_cost=round((tested0 + steps) * 45.0, 1),
    )
