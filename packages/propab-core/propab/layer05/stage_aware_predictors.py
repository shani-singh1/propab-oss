"""Stage-aware entropy predictor — cold_start / growth / plateau (fixes.md P2)."""
from __future__ import annotations

from typing import Any

from propab.layer05.hybrid_simulator import _resample_series
from propab.layer05.state_embedding_index import StateIndexEntry
from propab.layer05.trajectory_stages import stage_at_index, stage_ranges


def _neighbor_weight(entry: StateIndexEntry, dist: float, *, feature_idx: int, target: float) -> float:
    feats = entry.features_v2 or entry.features
    feat_val = feats[feature_idx] if len(feats) > feature_idx else 0.5
    regime_penalty = abs(feat_val - target)
    return (1.0 / (dist + 0.01)) * (1.0 / (1.0 + regime_penalty))


class ColdStartPredictor:
    """Low-tested regime — favor nearest single neighbor + rules anchor."""

    def predict_series(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
        rules_series: list[float],
    ) -> list[float]:
        n = steps + 1
        start, end = stage_ranges(steps)["cold_start"]
        out = list(rules_series[:n])
        if not neighbors:
            return out
        top = neighbors[:1]
        retrieved = _resample_series(top[0][0].snapshots, "theme_entropy", steps)
        for i in range(start, min(end, n)):
            out[i] = round(0.35 * rules_series[i] + 0.65 * retrieved[i], 4)
        return out


class GrowthPredictor:
    """Rising entropy — neighbors matched on early growth rate."""

    def predict_series(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
        rules_series: list[float],
        target_growth: float = 0.1,
    ) -> list[float]:
        n = steps + 1
        start, end = stage_ranges(steps)["growth"]
        out = list(rules_series[:n])
        if not neighbors:
            return out
        weights = [_neighbor_weight(e, d, feature_idx=10, target=target_growth) for e, d in neighbors]
        wsum = sum(weights) or 1.0
        for i in range(start, min(end, n)):
            val = 0.0
            for (entry, _), w in zip(neighbors, weights):
                series = _resample_series(entry.snapshots, "theme_entropy", steps)
                val += (w / wsum) * series[i]
            out[i] = round(0.15 * rules_series[i] + 0.85 * val, 4)
        return out


class PlateauPredictor:
    """Late flat regime — neighbors matched on plateau position."""

    def predict_series(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
        rules_series: list[float],
        target_plateau: float = 0.7,
    ) -> list[float]:
        n = steps + 1
        start, end = stage_ranges(steps)["plateau"]
        out = list(rules_series[:n])
        if not neighbors:
            return out
        weights = [_neighbor_weight(e, d, feature_idx=11, target=target_plateau) for e, d in neighbors]
        wsum = sum(weights) or 1.0
        for i in range(start, min(end, n)):
            val = 0.0
            for (entry, _), w in zip(neighbors, weights):
                series = _resample_series(entry.snapshots, "theme_entropy", steps)
                val += (w / wsum) * series[i]
            out[i] = round(0.25 * rules_series[i] + 0.75 * val, 4)
        return out


class StageAwareEntropyPredictor:
    """Dispatch cold_start / growth / plateau predictors per step region."""

    def __init__(self) -> None:
        self.cold = ColdStartPredictor()
        self.growth = GrowthPredictor()
        self.plateau = PlateauPredictor()

    def predict(
        self,
        neighbors: list[tuple[StateIndexEntry, float]],
        *,
        steps: int,
        rules_series: list[float],
        query_snapshots: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        n = steps + 1
        while len(rules_series) < n:
            rules_series.append(rules_series[-1] if rules_series else 0.0)

        from propab.layer05.state_vector_v2 import early_growth_rate, plateau_position

        snaps = query_snapshots or []
        target_growth = early_growth_rate(snaps) if snaps else 0.1
        target_plateau = plateau_position(snaps) if snaps else 0.7

        cold = self.cold.predict_series(neighbors, steps=steps, rules_series=rules_series)
        growth = self.growth.predict_series(
            neighbors, steps=steps, rules_series=cold,
            target_growth=target_growth,
        )
        return self.plateau.predict_series(
            neighbors, steps=steps, rules_series=growth,
            target_plateau=target_plateau,
        )
