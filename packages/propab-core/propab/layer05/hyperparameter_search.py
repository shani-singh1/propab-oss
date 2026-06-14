"""Hyperparameter grid search on residual dataset — no campaigns (fixes.md P0)."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from propab.layer05.cross_validation import leave_one_out_evaluate
from propab.layer05.direction_loss import direction_weighted_score
from propab.layer05.ensemble_simulator import SIM_V3
from propab.layer05.hybrid_simulator import SIM_V2
from propab.layer05.simulator_hyperparams import SimulatorHyperparams
from propab.policy_record import PolicyRecord

RETRIEVAL_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
K_NEIGHBORS = [1, 3, 5, 7, 10, 15]
NORMALIZATIONS = ["none", "l2", "minmax"]
DISTANCE_METRICS = ["euclidean", "manhattan", "cosine"]
V3_PREDICTOR_WEIGHTS = [0.15, 0.25, 0.35]


@dataclass
class GridSearchResult:
    best_hyperparams: dict[str, Any]
    best_score: dict[str, float]
    n_evaluated: int
    top_configs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_v2_grid() -> Iterator[SimulatorHyperparams]:
    for rw, k, norm, dist in itertools.product(
        RETRIEVAL_WEIGHTS, K_NEIGHBORS, NORMALIZATIONS, DISTANCE_METRICS,
    ):
        yield SimulatorHyperparams(
            retrieval_weight=rw,
            k_neighbors=k,
            normalization=norm,
            distance_metric=dist,
            feature_version="v2",
        )


def _iter_v3_grid() -> Iterator[SimulatorHyperparams]:
    for k, norm, dist, pw in itertools.product(
        K_NEIGHBORS, NORMALIZATIONS, DISTANCE_METRICS, V3_PREDICTOR_WEIGHTS,
    ):
        knn_w = max(0.3, 0.85 - pw)
        yield SimulatorHyperparams(
            k_neighbors=k,
            normalization=norm,
            distance_metric=dist,
            feature_version="v2",
            knn_weight=round(knn_w, 2),
            predictor_weight=pw,
            rules_weight=round(max(0.05, 1.0 - knn_w - pw), 2),
        )


def _score_loo(
    aggregate: dict[str, float],
    *,
    direction_weight: float = 5.0,
    magnitude_weight: float = 0.2,
) -> float:
    return direction_weighted_score(
        aggregate,
        direction_weight=direction_weight,
        magnitude_weight=magnitude_weight,
    )


def grid_search_loo(
    *,
    campaigns: dict[str, list[dict[str, Any]]],
    policy: PolicyRecord,
    simulator_version: str = SIM_V2,
    max_configs: int | None = None,
    direction_weight: float = 5.0,
    magnitude_weight: float = 0.2,
) -> GridSearchResult:
    grid = (
        _iter_v3_grid() if simulator_version == SIM_V3
        else _iter_v2_grid()
    )
    results: list[tuple[float, SimulatorHyperparams, dict[str, float]]] = []

    for i, hp in enumerate(grid):
        if max_configs is not None and i >= max_configs:
            break
        sim_ver = simulator_version
        loo = leave_one_out_evaluate(
            campaigns=campaigns,
            policy=policy,
            simulator_version=sim_ver,
            hyperparams=hp,
        )
        score = _score_loo(
            loo.aggregate,
            direction_weight=direction_weight,
            magnitude_weight=magnitude_weight,
        )
        results.append((score, hp, loo.aggregate))

    if not results:
        empty = SimulatorHyperparams()
        return GridSearchResult(
            best_hyperparams=empty.to_dict(),
            best_score={},
            n_evaluated=0,
            top_configs=[],
        )

    results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_hp, best_agg = results[0]
    top = [
        {
            "score": round(s, 4),
            "hyperparams": hp.to_dict(),
            "aggregate": agg,
        }
        for s, hp, agg in results[:5]
    ]
    return GridSearchResult(
        best_hyperparams=best_hp.to_dict(),
        best_score=best_agg,
        n_evaluated=len(results),
        top_configs=top,
    )
