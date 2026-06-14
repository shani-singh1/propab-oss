"""Ensemble simulator sim_v3 — rules + kNN + component predictors (fixes.md P3)."""
from __future__ import annotations

from typing import Any

from propab.entropy_trajectory import summarize_entropy_trajectory
from propab.layer05.hybrid_simulator import _retrieve_component_series
from propab.layer05.replay_state import SearchState
from propab.layer05.search_simulator import SimulationResult, simulate_search
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.layer05.trajectory_predictors import predict_all_components
from propab.policy_record import PolicyRecord
from propab.search_policy import SearchPolicy

SIM_V3 = "sim_v3"


def _blend_three(
    rules: list[float],
    knn: list[float],
    pred: list[float],
    *,
    w_rules: float,
    w_knn: float,
    w_pred: float,
    n: int,
) -> list[float]:
    def pad(series: list[float]) -> list[float]:
        while len(series) < n:
            series.append(series[-1] if series else 0.0)
        return series

    rules, knn, pred = pad(list(rules)), pad(list(knn)), pad(list(pred))
    return [
        round(w_rules * rules[i] + w_knn * knn[i] + w_pred * pred[i], 4)
        for i in range(n)
    ]


def simulate_search_ensemble(
    *,
    state: SearchState,
    policy: PolicyRecord | SearchPolicy,
    index: StateEmbeddingIndex,
    steps: int = 30,
    hyperparams: SimulatorHyperparams | None = None,
    query_snapshots: list[dict[str, Any]] | None = None,
    weakest_component: str | None = None,
) -> SimulationResult:
    hp = (hyperparams or SimulatorHyperparams()).for_v3()
    w_rules, w_knn, w_pred = hp.rules_weight, hp.knn_weight, hp.predictor_weight
    if weakest_component:
        boost = 0.1
        if weakest_component in ("entropy",):
            w_pred = min(0.6, w_pred + boost)
        elif weakest_component in ("closure",):
            w_pred = min(0.6, w_pred + boost)
        w_rules = max(0.05, 1.0 - w_knn - w_pred)

    rules = simulate_search(state=state, policy=policy, steps=steps)
    neighbors = index.nearest(
        state,
        k=hp.k_neighbors,
        hyperparams=hp,
        query_snapshots=query_snapshots,
    )
    if not neighbors:
        rules.policy_id = getattr(policy, "id", "search_policy")
        return rules

    retrieved = _retrieve_component_series(neighbors, steps=steps)
    predicted = predict_all_components(neighbors, steps=steps)
    rules_h = [float(p.get("theme_entropy") or 0) for p in rules.entropy_points]
    rules_c = rules.closure_trajectory
    n = steps + 1

    blended_h = _blend_three(
        rules_h, retrieved["entropy"], predicted["entropy"],
        w_rules=w_rules, w_knn=w_knn, w_pred=w_pred, n=n,
    )
    blended_c = _blend_three(
        rules_c, retrieved["closure"], predicted["closure"],
        w_rules=w_rules, w_knn=w_knn, w_pred=w_pred, n=n,
    )

    tested0 = state.tested_count
    entropy_points = [
        {"tested": tested0 + i, "theme_entropy": blended_h[i]}
        for i in range(n)
    ]
    summary = summarize_entropy_trajectory(entropy_points)
    sat_end = predicted["saturation"][-1] if predicted["saturation"] else 0.5

    return SimulationResult(
        policy_id=getattr(policy, "id", "search_policy"),
        steps=steps,
        entropy_trajectory=summary.to_dict(),
        entropy_points=entropy_points,
        closure_trajectory=blended_c,
        branching_factor=round(
            sum(predicted["branching"]) / max(1, len(predicted["branching"])), 4
        ),
        theme_saturation={"dominant": round(sat_end, 4)},
        expected_compute_cost=round((tested0 + steps) * 45.0, 1),
    )
