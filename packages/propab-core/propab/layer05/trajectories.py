"""Trajectory objects — first-class time-series metrics (fixes.md P1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TrajectoryBase:
    start: float
    mid: float
    end: float
    growth_rate: float
    plateau_point: int | None
    n_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_values(
        cls,
        values: list[float],
        *,
        tested: list[int] | None = None,
        plateau_epsilon: float = 0.06,
    ) -> TrajectoryBase:
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, None, 0)
        start, end = values[0], values[-1]
        mid = values[len(values) // 2]
        t0 = (tested or [0])[0] if tested else 0
        t1 = (tested or [len(values) - 1])[-1] if tested else max(1, len(values) - 1)
        dt = max(1, t1 - t0)
        growth = round((end - start) / dt, 4)
        plateau: int | None = None
        if tested and len(values) >= 2:
            for i in range(len(values) - 1):
                if abs(values[i + 1] - values[i]) <= plateau_epsilon:
                    plateau = tested[i]
                    break
        return cls(
            start=round(start, 4),
            mid=round(mid, 4),
            end=round(end, 4),
            growth_rate=growth,
            plateau_point=plateau,
            n_points=len(values),
        )


@dataclass
class EntropyTrajectory(TrajectoryBase):
    cross_H_1_5_at: int | None = None
    cross_H_2_0_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["cross_H_1_5_at"] = self.cross_H_1_5_at
        d["cross_H_2_0_at"] = self.cross_H_2_0_at
        return d

    @classmethod
    def from_points(cls, points: list[dict[str, Any]]) -> EntropyTrajectory:
        values = [float(p.get("theme_entropy") or 0) for p in points]
        tested = [int(p.get("tested") or i) for i, p in enumerate(points)]
        base = TrajectoryBase.from_values(values, tested=tested)
        cross15 = next((t for v, t in zip(values, tested) if v >= 1.5), None)
        cross20 = next((t for v, t in zip(values, tested) if v >= 2.0), None)
        return cls(
            start=base.start,
            mid=base.mid,
            end=base.end,
            growth_rate=base.growth_rate,
            plateau_point=base.plateau_point,
            n_points=base.n_points,
            cross_H_1_5_at=cross15,
            cross_H_2_0_at=cross20,
        )


@dataclass
class ClosureTrajectory(TrajectoryBase):
    @classmethod
    def from_points(cls, points: list[dict[str, Any]]) -> ClosureTrajectory:
        values = [float(p.get("closure_ratio") or 0) for p in points]
        tested = [int(p.get("tested") or i) for i, p in enumerate(points)]
        base = TrajectoryBase.from_values(values, tested=tested)
        return cls(
            start=base.start,
            mid=base.mid,
            end=base.end,
            growth_rate=base.growth_rate,
            plateau_point=base.plateau_point,
            n_points=base.n_points,
        )


@dataclass
class BranchingTrajectory(TrajectoryBase):
    @classmethod
    def from_snapshots(cls, snapshots: list[dict[str, Any]]) -> BranchingTrajectory:
        deltas: list[float] = []
        tested: list[int] = []
        for i in range(1, len(snapshots)):
            prev = int(snapshots[i - 1].get("generated") or snapshots[i - 1].get("tested") or 0)
            cur = int(snapshots[i].get("generated") or snapshots[i].get("tested") or 0)
            deltas.append(float(max(0, cur - prev)))
            tested.append(int(snapshots[i].get("tested") or i))
        if not deltas:
            return cls(0.0, 0.0, 0.0, 0.0, None, 0)
        base = TrajectoryBase.from_values(deltas, tested=tested)
        return cls(
            start=base.start,
            mid=base.mid,
            end=base.end,
            growth_rate=base.growth_rate,
            plateau_point=base.plateau_point,
            n_points=base.n_points,
        )


@dataclass
class ThemeSaturationTrajectory(TrajectoryBase):
    dominant_theme: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["dominant_theme"] = self.dominant_theme
        return d

    @classmethod
    def from_snapshots(cls, snapshots: list[dict[str, Any]]) -> ThemeSaturationTrajectory:
        fracs: list[float] = []
        tested: list[int] = []
        dominant: str | None = None
        for i, snap in enumerate(snapshots):
            hist = snap.get("theme_histogram") or {}
            total = sum(hist.values()) or 1
            if hist:
                top = max(hist, key=hist.get)
                dominant = top
                fracs.append(hist[top] / total)
            else:
                fracs.append(0.0)
            tested.append(int(snap.get("tested") or i))
        base = TrajectoryBase.from_values(fracs, tested=tested)
        return cls(
            start=base.start,
            mid=base.mid,
            end=base.end,
            growth_rate=base.growth_rate,
            plateau_point=base.plateau_point,
            n_points=base.n_points,
            dominant_theme=dominant,
        )


@dataclass
class CampaignTrajectories:
    entropy: EntropyTrajectory
    closure: ClosureTrajectory
    branching: BranchingTrajectory
    theme_saturation: ThemeSaturationTrajectory

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy": self.entropy.to_dict(),
            "closure": self.closure.to_dict(),
            "branching": self.branching.to_dict(),
            "theme_saturation": self.theme_saturation.to_dict(),
        }

    @classmethod
    def from_snapshots(cls, snapshots: list[dict[str, Any]]) -> CampaignTrajectories:
        entropy_pts = [
            {"tested": s.get("tested"), "theme_entropy": s.get("theme_entropy")}
            for s in snapshots
        ]
        closure_pts = [
            {"tested": s.get("tested"), "closure_ratio": s.get("closure_ratio")}
            for s in snapshots
        ]
        return cls(
            entropy=EntropyTrajectory.from_points(entropy_pts),
            closure=ClosureTrajectory.from_points(closure_pts),
            branching=BranchingTrajectory.from_snapshots(snapshots),
            theme_saturation=ThemeSaturationTrajectory.from_snapshots(snapshots),
        )
