"""Trajectory stage detection — cold_start, growth, plateau."""
from __future__ import annotations

from typing import Any, Literal

StageName = Literal["cold_start", "growth", "plateau"]


def stage_at_index(step_index: int, total_steps: int) -> StageName:
    if total_steps <= 0:
        return "cold_start"
    frac = step_index / max(1, total_steps)
    if frac < 0.33:
        return "cold_start"
    if frac < 0.67:
        return "growth"
    return "plateau"


def stage_ranges(total_steps: int) -> dict[StageName, tuple[int, int]]:
    n = total_steps + 1
    third = max(1, n // 3)
    return {
        "cold_start": (0, third),
        "growth": (third, 2 * third),
        "plateau": (2 * third, n),
    }


def directional_agreement_for_stage(
    sim: list[float],
    obs: list[float],
    *,
    stage: StageName,
    total_steps: int,
) -> float:
    start, end = stage_ranges(total_steps)[stage]
    if end - start < 2:
        return 1.0
    from propab.layer05.bench_metrics import directional_agreement

    return directional_agreement(sim[start:end], obs[start:end])


def entropy_values_from_snapshots(snapshots: list[dict[str, Any]]) -> list[float]:
    return [float(s.get("theme_entropy") or 0) for s in snapshots]
