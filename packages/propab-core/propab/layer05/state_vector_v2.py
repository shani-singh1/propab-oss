"""StateVectorV2 — expanded state features for k-NN retrieval (fixes.md P1)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from propab.layer05.replay_state import SearchState


FEATURE_NAMES_V2 = (
    "entropy",
    "tested",
    "closure",
    "diversity",
    "saturation",
    "frontier_size",
    "pending",
    "confirmed_frac",
    "inconclusive_frac",
    "theme_count",
    "growth_rate_early",
    "plateau_position",
)


@dataclass
class FeatureStats:
    mins: list[float]
    maxs: list[float]

    @classmethod
    def from_matrix(cls, rows: list[list[float]]) -> FeatureStats:
        if not rows:
            return cls(mins=[], maxs=[])
        n = len(rows[0])
        mins = [min(r[i] for r in rows) for i in range(n)]
        maxs = [max(r[i] for r in rows) for i in range(n)]
        return cls(mins=mins, maxs=maxs)


def _theme_saturation(hist: dict[str, int]) -> float:
    total = sum(hist.values()) or 1
    return max(hist.values()) / total if hist else 0.5


def early_growth_rate(snapshots: list[dict[str, Any]]) -> float:
    if len(snapshots) < 2:
        return 0.0
    h0 = float(snapshots[0].get("theme_entropy") or 0)
    h1 = float(snapshots[min(2, len(snapshots) - 1)].get("theme_entropy") or 0)
    t0 = int(snapshots[0].get("tested") or 0)
    t1 = int(snapshots[min(2, len(snapshots) - 1)].get("tested") or 0)
    dt = max(1, t1 - t0)
    return round((h1 - h0) / dt, 4)


def plateau_position(snapshots: list[dict[str, Any]], epsilon: float = 0.06) -> float:
    values = [float(s.get("theme_entropy") or 0) for s in snapshots]
    tested = [int(s.get("tested") or i) for i, s in enumerate(snapshots)]
    for i in range(len(values) - 1):
        if abs(values[i + 1] - values[i]) <= epsilon:
            return round(tested[i] / max(1, tested[-1]), 4)
    return 1.0


def state_features_v1(
    state: SearchState | dict[str, Any],
    *,
    max_tested: float = 250.0,
) -> list[float]:
    if isinstance(state, SearchState):
        entropy = state.entropy
        tested = float(state.tested_count)
        closure = state.closure_ratio
        hist = state.theme_histogram
    else:
        entropy = float(state.get("entropy") or state.get("theme_entropy") or 0)
        tested = float(state.get("tested_count") or state.get("tested") or 0)
        closure = float(state.get("closure_ratio") or 0)
        hist = state.get("theme_histogram") or {}

    total = sum(hist.values()) or 1
    top_frac = max(hist.values()) / total if hist else 1.0
    diversity = 1.0 - top_frac
    branching_proxy = min(1.0, tested / max_tested)

    return [
        round(entropy / 3.0, 4),
        round(tested / max_tested, 4),
        round(closure, 4),
        round(branching_proxy, 4),
        round(diversity, 4),
    ]


def state_vector_v2(
    state: SearchState | dict[str, Any],
    *,
    snapshots: list[dict[str, Any]] | None = None,
    max_tested: float = 250.0,
) -> list[float]:
    if isinstance(state, SearchState):
        entropy = state.entropy
        tested = float(state.tested_count)
        closure = state.closure_ratio
        hist = state.theme_histogram
        pending = float(state.pending_nodes)
        frontier = float(getattr(state, "frontier_size", 0) or len(state.frontier))
    else:
        entropy = float(state.get("entropy") or state.get("theme_entropy") or 0)
        tested = float(state.get("tested_count") or state.get("tested") or 0)
        closure = float(state.get("closure_ratio") or 0)
        hist = state.get("theme_histogram") or {}
        pending = float(state.get("pending") or state.get("pending_nodes") or 0)
        frontier = float(state.get("frontier_size") or 0)

    total = sum(hist.values()) or 1
    top_frac = max(hist.values()) / total if hist else 1.0
    diversity = 1.0 - top_frac
    saturation = _theme_saturation(hist)
    confirmed = float(
        (state.get("confirmed") if isinstance(state, dict) else 0) or 0
    )
    refuted = float((state.get("refuted") if isinstance(state, dict) else 0) or 0)
    inconclusive = float((state.get("inconclusive") if isinstance(state, dict) else 0) or 0)
    outcome_total = confirmed + refuted + inconclusive or tested or 1
    snaps = snapshots or []
    growth = early_growth_rate(snaps) if snaps else 0.0
    plateau = plateau_position(snaps) if snaps else 0.5

    return [
        round(entropy / 3.0, 4),
        round(tested / max_tested, 4),
        round(closure, 4),
        round(diversity, 4),
        round(saturation, 4),
        round(frontier / 20.0, 4),
        round(pending / 20.0, 4),
        round(confirmed / outcome_total, 4),
        round(inconclusive / outcome_total, 4),
        round(len(hist) / 10.0, 4),
        round(growth, 4),
        round(plateau, 4),
    ]


def build_state_vector(
    state: SearchState | dict[str, Any],
    *,
    version: str = "v1",
    snapshots: list[dict[str, Any]] | None = None,
    max_tested: float = 250.0,
) -> list[float]:
    if version == "v2":
        return state_vector_v2(state, snapshots=snapshots, max_tested=max_tested)
    return state_features_v1(state, max_tested=max_tested)


# Backward-compatible alias
state_features = state_features_v1


def normalize_features(
    features: list[float],
    method: str,
    stats: FeatureStats | None = None,
) -> list[float]:
    if method == "none" or not features:
        return features
    if method == "l2":
        norm = math.sqrt(sum(x * x for x in features)) or 1.0
        return [round(x / norm, 6) for x in features]
    if method == "minmax" and stats and stats.mins and stats.maxs:
        out: list[float] = []
        for x, lo, hi in zip(features, stats.mins, stats.maxs):
            span = hi - lo
            out.append(round((x - lo) / span, 6) if span > 1e-9 else 0.0)
        return out
    return features


def feature_distance(
    a: list[float],
    b: list[float],
    metric: str = "euclidean",
) -> float:
    if metric == "manhattan":
        return sum(abs(x - y) for x, y in zip(a, b))
    if metric == "cosine":
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return 1.0 - dot / (na * nb)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
