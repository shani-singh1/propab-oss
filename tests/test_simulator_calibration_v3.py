"""Tests for simulator calibration v3 — StateVectorV2, LOO, grid search, error ledger."""
from __future__ import annotations

import json

from propab.layer05.cross_validation import leave_one_out_evaluate
from propab.layer05.ensemble_simulator import SIM_V3, simulate_search_ensemble
from propab.layer05.hybrid_simulator import SIM_V2
from propab.layer05.hyperparameter_search import grid_search_loo
from propab.layer05.replay_state import SearchState
from propab.layer05.simulation_error_ledger import classify_simulation_errors
from propab.layer05.simulator_acceptance import evaluate_v2_acceptance, evaluate_v3_acceptance
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.layer05.state_embedding_index import StateEmbeddingIndex
from propab.layer05.state_vector_v2 import (
    build_state_vector,
    feature_distance,
    normalize_features,
    state_vector_v2,
)
from propab.layer05.trajectory_predictors import predict_all_components
from propab.policy_record import PolicyRecord, PolicyStatus


def _synthetic_snapshots(n: int = 8, offset: int = 0) -> list[dict]:
    snaps = []
    hist = {"diffusion_dynamics": 1}
    for i in range(1, n + 1):
        if i > 2:
            hist["spectral"] = hist.get("spectral", 0) + 1
        tested = i + offset
        snaps.append({
            "tested": tested,
            "executed": tested,
            "generated": tested + 2,
            "pending": 2,
            "frontier_size": 3,
            "theme_histogram": dict(hist),
            "theme_entropy": 0.5 + i * 0.12,
            "closure_ratio": round(0.1 + i * 0.07, 4),
        })
    return snaps


def _policy() -> PolicyRecord:
    return PolicyRecord(
        id="pol-v3",
        generation=1,
        parent_policy_id="pol-root",
        budget_bucket="3h",
        domain_bucket="graphs",
        boosts={"diffusion_dynamics": 0.3},
        status=PolicyStatus.CANDIDATE,
    )


def test_state_vector_v2_expanded():
    state = SearchState.from_snapshot(_synthetic_snapshots(5)[0])
    v2 = state_vector_v2(state, snapshots=_synthetic_snapshots(5))
    assert len(v2) == 12
    v1 = build_state_vector(state, version="v1")
    assert len(v1) == 5


def test_feature_distance_metrics():
    a = [0.1, 0.2, 0.3]
    b = [0.2, 0.3, 0.4]
    assert feature_distance(a, b, "euclidean") > 0
    assert feature_distance(a, b, "manhattan") > 0
    assert 0 <= feature_distance(a, b, "cosine") <= 2


def test_normalize_l2():
    feats = [3.0, 4.0]
    normed = normalize_features(feats, "l2")
    assert abs(sum(x * x for x in normed) - 1.0) < 0.01


def test_component_predictors():
    by_id = {"a": _synthetic_snapshots(6), "b": _synthetic_snapshots(6, 3)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    neighbors = index.nearest(SearchState.from_snapshot(by_id["a"][0]), k=2)
    preds = predict_all_components(neighbors, steps=5)
    assert len(preds["entropy"]) == 6
    assert len(preds["closure"]) == 6


def test_ensemble_simulator():
    by_id = {"a": _synthetic_snapshots(8), "b": _synthetic_snapshots(8, 4)}
    index = StateEmbeddingIndex().build_from_snapshots_map(by_id)
    sim = simulate_search_ensemble(
        state=SearchState.from_snapshot(by_id["a"][0]),
        policy=_policy(),
        index=index,
        steps=7,
        hyperparams=SimulatorHyperparams(predictor_weight=0.25, knn_weight=0.55),
    )
    assert len(sim.entropy_points) == 8


def test_leave_one_out_evaluate():
    campaigns = {
        "a": _synthetic_snapshots(6),
        "b": _synthetic_snapshots(6, 2),
        "c": _synthetic_snapshots(6, 4),
    }
    loo = leave_one_out_evaluate(
        campaigns=campaigns,
        policy=_policy(),
        simulator_version=SIM_V2,
        hyperparams=SimulatorHyperparams(retrieval_weight=0.7, k_neighbors=2),
    )
    assert len(loo.folds) == 3
    assert "directional_agreement" in loo.aggregate


def test_grid_search_small():
    campaigns = {"a": _synthetic_snapshots(5), "b": _synthetic_snapshots(5, 2)}
    result = grid_search_loo(
        campaigns=campaigns,
        policy=_policy(),
        simulator_version=SIM_V2,
        max_configs=8,
    )
    assert result.n_evaluated <= 8
    assert "retrieval_weight" in result.best_hyperparams


def test_error_classification():
    snaps = _synthetic_snapshots(6)
    sim_h = [{"tested": i, "theme_entropy": 0.1} for i in range(6)]
    err = classify_simulation_errors(
        campaign_id="x",
        simulator_version=SIM_V2,
        simulated_entropy_points=sim_h,
        simulated_closure=[0.1] * 6,
        observed_snapshots=snaps,
    )
    assert err.magnitude_errors >= 0
    assert err.n_points > 0


def test_v2_v3_acceptance_rules():
    v2_ok = evaluate_v2_acceptance(
        current={"directional_agreement": 0.5, "mae_entropy": 0.5, "mae_closure": 0.1},
        baseline_v1={"directional_agreement": 0.14, "mae_entropy": 1.4, "mae_closure": 0.3},
    )
    assert v2_ok.accepted is True

    v3_fail = evaluate_v3_acceptance(
        aggregate={"directional_agreement": 0.75, "mae_entropy": 0.4, "mae_closure": 0.1},
        loo_aggregate={"directional_agreement": 0.72},
    )
    assert v3_fail.accepted is False

    v3_ok = evaluate_v3_acceptance(
        aggregate={"directional_agreement": 0.85, "mae_entropy": 0.3, "mae_closure": 0.08},
        loo_aggregate={"directional_agreement": 0.82},
    )
    assert v3_ok.accepted is True
