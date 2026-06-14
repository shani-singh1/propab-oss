"""Local trajectory predictors — calibrated per component (fixes.md P2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from propab.layer05.hybrid_simulator import (
    _branching_series,
    _resample_series,
    _saturation_series,
)
from propab.layer05.state_embedding_index import StateIndexEntry


@dataclass
class ComponentPrediction:
    component: str
    series: list[float]


class EntropyPredictor:
    def predict(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
    ) -> list[float]:
        return _blend_neighbors(neighbors, steps, "theme_entropy", _resample_series)


class ClosurePredictor:
    def predict(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
    ) -> list[float]:
        return _blend_neighbors(neighbors, steps, "closure_ratio", _resample_series)


class BranchingPredictor:
    def predict(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
    ) -> list[float]:
        return _blend_neighbors(neighbors, steps, "", _branching_series)


class ThemePredictor:
    def predict(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
    ) -> list[float]:
        return _blend_neighbors(neighbors, steps, "", _saturation_series)


def _blend_neighbors(
    neighbors: list[tuple[StateIndexEntry, float]],
    steps: int,
    key: str,
    fn,
) -> list[float]:
    if not neighbors:
        return [0.0] * (steps + 1)
    weights = [1.0 / (d + 0.01) for _, d in neighbors]
    wsum = sum(weights) or 1.0
    n = steps + 1
    out = [0.0] * n
    for (entry, _), w in zip(neighbors, weights):
        snaps = entry.snapshots
        if key:
            series = fn(snaps, key, steps)
        else:
            series = fn(snaps, steps)
        for i in range(n):
            out[i] += w * (series[i] if i < len(series) else series[-1])
    return [round(v / wsum, 4) for v in out]


def predict_all_components(
    neighbors: list[tuple[StateIndexEntry, float]],
    *,
    steps: int,
) -> dict[str, list[float]]:
    return {
        "entropy": EntropyPredictor().predict(neighbors, steps=steps),
        "closure": ClosurePredictor().predict(neighbors, steps=steps),
        "branching": BranchingPredictor().predict(neighbors, steps=steps),
        "saturation": ThemePredictor().predict(neighbors, steps=steps),
    }
