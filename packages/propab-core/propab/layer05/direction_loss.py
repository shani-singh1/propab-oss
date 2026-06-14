"""Asymmetric direction-weighted loss for grid search (fixes.md P5)."""
from __future__ import annotations

from propab.layer05.bench_metrics import directional_agreement, mae


def count_direction_errors(sim: list[float], obs: list[float]) -> int:
    errors = 0
    for i in range(1, min(len(sim), len(obs))):
        ds = sim[i] - sim[i - 1]
        do = obs[i] - obs[i - 1]
        if ds * do < 0 and not (ds == 0 and do == 0):
            errors += 1
    return errors


def direction_accuracy_weighted_loss(
    sim: list[float],
    obs: list[float],
    *,
    direction_weight: float = 5.0,
    magnitude_weight: float = 1.0,
) -> float:
    """
    Lower is better. Direction mistakes dominate magnitude mistakes.
    """
    direction = directional_agreement(sim, obs)
    mag = mae(sim, obs)
    dir_errors = count_direction_errors(sim, obs)
    n_steps = max(1, min(len(sim), len(obs)) - 1)
    return round(
        direction_weight * (1.0 - direction)
        + direction_weight * (dir_errors / n_steps)
        + magnitude_weight * mag,
        4,
    )


def direction_weighted_score(
    aggregate: dict[str, float],
    *,
    direction_weight: float = 5.0,
    magnitude_weight: float = 0.2,
) -> float:
    """Higher is better — for grid search ranking."""
    direction = float(aggregate.get("directional_agreement") or 0)
    mae_e = float(aggregate.get("mae_entropy") or 99)
    dir_err_rate = float(aggregate.get("direction_error_rate") or 0)
    return round(
        direction_weight * direction
        - magnitude_weight * mae_e
        - direction_weight * dir_err_rate,
        4,
    )
