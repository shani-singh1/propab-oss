"""Within-campaign theme entropy trajectory — summarize for analyst and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EntropyTrajectorySummary:
    H_start: float
    H_mid: float
    H_end: float
    growth_pattern: str
    plateau_at_tested: int | None
    cross_H_1_5_at_tested: int | None
    cross_H_2_0_at_tested: int | None
    growth_rate: float
    saturation_H: float
    n_snapshots: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy_value(point: dict[str, Any]) -> float:
    return float(point.get("theme_entropy") or 0)


def _classify_growth_pattern(points: list[dict[str, Any]]) -> str:
    if len(points) < 3:
        return "insufficient_data"
    hs = [_entropy_value(p) for p in points]
    start, end = hs[0], hs[-1]
    peak = max(hs)
    peak_idx = hs.index(peak)
    if end < start * 0.85 and peak_idx < len(hs) * 0.4:
        return "early_collapse"
    if end < start * 0.85:
        return "mid_late_collapse"
    if peak > start * 1.15 and end <= peak * 0.95 and peak_idx < len(hs) * 0.55:
        return "early_peak_then_plateau"
    if end > start * 1.1:
        return "monotone_rise"
    return "flat_or_mixed"


def _cross_at(points: list[dict[str, Any]], threshold: float) -> int | None:
    for p in points:
        if _entropy_value(p) >= threshold:
            tested = p.get("tested")
            return int(tested) if tested is not None else None
    return None


def _plateau_at_tested(points: list[dict[str, Any]], *, epsilon: float = 0.06) -> int | None:
    """First tested count where entropy stays within epsilon for 2+ consecutive snaps."""
    if len(points) < 2:
        return None
    hs = [_entropy_value(p) for p in points]
    tested = [int(p.get("tested") or 0) for p in points]
    for i in range(len(hs) - 1):
        if abs(hs[i + 1] - hs[i]) <= epsilon:
            return tested[i]
    if len(hs) >= 3 and abs(hs[-1] - hs[-2]) <= epsilon:
        return tested[-2]
    return None


def _growth_rate(points: list[dict[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    h0 = _entropy_value(points[0])
    h1 = _entropy_value(points[-1])
    t0 = int(points[0].get("tested") or 1)
    t1 = int(points[-1].get("tested") or t0)
    dt = max(1, t1 - t0)
    return round((h1 - h0) / dt, 4)


def summarize_entropy_trajectory(points: list[dict[str, Any]]) -> EntropyTrajectorySummary:
    """Build P2 trajectory summary from frontier_snapshot payloads or point dicts."""
    if not points:
        return EntropyTrajectorySummary(
            H_start=0.0,
            H_mid=0.0,
            H_end=0.0,
            growth_pattern="insufficient_data",
            plateau_at_tested=None,
            cross_H_1_5_at_tested=None,
            cross_H_2_0_at_tested=None,
            growth_rate=0.0,
            saturation_H=0.0,
            n_snapshots=0,
        )
    hs = [_entropy_value(p) for p in points]
    return EntropyTrajectorySummary(
        H_start=round(hs[0], 4),
        H_mid=round(hs[len(hs) // 2], 4),
        H_end=round(hs[-1], 4),
        growth_pattern=_classify_growth_pattern(points),
        plateau_at_tested=_plateau_at_tested(points),
        cross_H_1_5_at_tested=_cross_at(points, 1.5),
        cross_H_2_0_at_tested=_cross_at(points, 2.0),
        growth_rate=_growth_rate(points),
        saturation_H=round(hs[-1], 4),
        n_snapshots=len(points),
    )


def trajectory_point_from_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    return {
        "tested": snap.get("tested"),
        "theme_entropy": snap.get("theme_entropy"),
    }


def observed_entropy_dynamics(summary: EntropyTrajectorySummary | dict[str, Any]) -> dict[str, float]:
    d = summary.to_dict() if isinstance(summary, EntropyTrajectorySummary) else dict(summary)
    cross15 = d.get("cross_H_1_5_at_tested")
    cross20 = d.get("cross_H_2_0_at_tested")
    return {
        "start_H": float(d.get("H_start") or 0),
        "growth_rate": float(d.get("growth_rate") or 0),
        "saturation_H": float(d.get("H_end") or d.get("saturation_H") or 0),
        "cross_H_1_5_at_tested": float(cross15 if cross15 is not None else 999),
        "cross_H_2_0_at_tested": float(cross20 if cross20 is not None else 999),
    }
