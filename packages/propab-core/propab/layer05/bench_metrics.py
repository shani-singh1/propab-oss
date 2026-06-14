"""Shared benchmark metrics for simulator calibration."""
from __future__ import annotations

import math
from typing import Any


def align_series(a: list[float], b: list[float]) -> tuple[list[float], list[float]]:
    n = max(len(a), len(b), 1)
    if len(a) < n:
        a = a + [a[-1] if a else 0.0] * (n - len(a))
    if len(b) < n:
        b = b + [b[-1] if b else 0.0] * (n - len(b))
    return a[:n], b[:n]


def mae(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return round(sum(abs(x - y) for x, y in zip(a, b)) / len(a), 4)


def rmse(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return round(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)), 4)


def directional_agreement(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 1.0
    agree = 0
    total = 0
    for i in range(1, min(len(a), len(b))):
        da = a[i] - a[i - 1]
        db = b[i] - b[i - 1]
        if da == 0 and db == 0:
            agree += 1
        elif da * db > 0:
            agree += 1
        total += 1
    return round(agree / max(1, total), 4)


def component_bench_result(
    *,
    component: str,
    simulated: list[float],
    observed: list[float],
    scalar_residuals: dict[str, float] | None = None,
) -> dict[str, Any]:
    sim, obs = align_series(simulated, observed)
    return {
        "component": component,
        "mae": mae(sim, obs),
        "rmse": rmse(sim, obs),
        "directional_agreement": directional_agreement(sim, obs),
        "scalar_residuals": scalar_residuals or {},
        "n_points": len(sim),
    }
