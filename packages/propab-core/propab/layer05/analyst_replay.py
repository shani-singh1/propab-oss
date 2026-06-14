"""Analyst replay — offline prediction evaluation (fixes.md Component 3)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from propab.entropy_trajectory import observed_entropy_dynamics
from propab.policy_evaluation import compute_entropy_residuals
from propab.policy_fitness_ledger import FitnessRecord, PolicyFitnessLedger
from propab.policy_record import PredictedEffects


@dataclass
class AnalystReplayRow:
    campaign_id: str
    policy_id: str
    predicted: dict[str, float]
    observed_entropy: dict[str, float]
    entropy_residuals: dict[str, float]
    scalar_residuals: dict[str, float]
    accept_or_reject: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalystReplayResult:
    n_evaluations: int
    rows: list[AnalystReplayRow]
    mean_abs_entropy_residual: float
    directional_bias: dict[str, bool]
    growth_rate_residuals: list[float]
    learning_trend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_evaluations": self.n_evaluations,
            "mean_abs_entropy_residual": self.mean_abs_entropy_residual,
            "directional_bias": self.directional_bias,
            "growth_rate_residuals": self.growth_rate_residuals,
            "learning_trend": self.learning_trend,
            "rows": [r.to_dict() for r in self.rows],
        }


def _directional(values: list[float]) -> bool:
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in values]
    nz = [s for s in signs if s != 0]
    return len(set(nz)) <= 1 if nz else False


def replay_analyst_on_fitness_record(
    record: FitnessRecord,
    *,
    trajectory: dict[str, Any] | None = None,
) -> AnalystReplayRow | None:
    traj = trajectory or (record.detail or {}).get("observed_trajectory")
    if not traj:
        return None
    predicted = PredictedEffects.from_dict(record.predictions)
    if not predicted.uses_entropy_dynamics():
        return None
    obs_entropy = observed_entropy_dynamics(traj)
    ent_res = compute_entropy_residuals(predicted, obs_entropy)
    scalar_res = {
        k: v
        for k, v in (record.residuals or {}).items()
        if k in ("closure_ratio", "refute_ratio", "compute_efficiency", "theme_entropy")
    }
    return AnalystReplayRow(
        campaign_id=record.campaign_id,
        policy_id=record.policy_id,
        predicted=predicted.to_dict(),
        observed_entropy=obs_entropy,
        entropy_residuals=ent_res,
        scalar_residuals=scalar_res,
        accept_or_reject=record.accept_or_reject,
    )


def replay_analyst_on_history(
    fitness: PolicyFitnessLedger | None = None,
    *,
    limit: int = 50,
) -> AnalystReplayResult:
    ledger = fitness or PolicyFitnessLedger.load()
    rows: list[AnalystReplayRow] = []
    for rec in ledger.records[-limit:]:
        row = replay_analyst_on_fitness_record(rec)
        if row:
            rows.append(row)

    flat = [abs(v) for r in rows for v in r.entropy_residuals.values()]
    mean_abs = sum(flat) / len(flat) if flat else 0.0
    growth_res = [r.entropy_residuals.get("growth_rate", 0) for r in rows]
    preds = [r.predicted.get("growth_rate", 0) for r in rows]

    directional = {
        "growth_rate": _directional(growth_res),
        "saturation_H": _directional([r.entropy_residuals.get("saturation_H", 0) for r in rows]),
        "start_H": _directional([r.entropy_residuals.get("start_H", 0) for r in rows]),
    }
    trend = "flat"
    if len(preds) >= 2:
        if preds[-1] < preds[0]:
            trend = "decreasing_growth_predictions"
        elif preds[-1] > preds[0]:
            trend = "increasing_growth_predictions"

    return AnalystReplayResult(
        n_evaluations=len(rows),
        rows=rows,
        mean_abs_entropy_residual=round(mean_abs, 4),
        directional_bias=directional,
        growth_rate_residuals=[round(x, 4) for x in growth_res],
        learning_trend=trend,
    )
