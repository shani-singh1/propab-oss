"""Weak campaign analysis — stage-local directional metrics (fixes.md P1)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from propab.layer05.bench_metrics import directional_agreement
from propab.layer05.direction_error_dataset import DirectionErrorDataset
from propab.layer05.trajectory_stages import (
    StageName,
    directional_agreement_for_stage,
    entropy_values_from_snapshots,
)


@dataclass
class StageMetrics:
    stage: str
    directional_agreement: float
    direction_errors: int
    n_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignStageReport:
    campaign_id: str
    overall_directional: float
    cold_start: StageMetrics
    growth: StageMetrics
    plateau: StageMetrics
    worst_stage: str
    direction_error_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "overall_directional": self.overall_directional,
            "cold_start": self.cold_start.to_dict(),
            "growth": self.growth.to_dict(),
            "plateau": self.plateau.to_dict(),
            "worst_stage": self.worst_stage,
            "direction_error_count": self.direction_error_count,
        }


@dataclass
class WeakCampaignReport:
    campaigns: list[dict[str, Any]]
    worst_campaign_id: str
    worst_fold_campaign_id: str
    worst_overall_directional: float
    stage_error_totals: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stage_metrics(
    sim: list[float],
    obs: list[float],
    *,
    stage: StageName,
    total_steps: int,
    direction_errors: int,
) -> StageMetrics:
    from propab.layer05.trajectory_stages import stage_ranges

    s, e = stage_ranges(total_steps)[stage]
    return StageMetrics(
        stage=stage,
        directional_agreement=directional_agreement_for_stage(
            sim, obs, stage=stage, total_steps=total_steps,
        ),
        direction_errors=direction_errors,
        n_points=max(0, e - s),
    )


def analyze_campaign_stages(
    *,
    campaign_id: str,
    simulated_entropy_points: list[dict[str, Any]],
    observed_snapshots: list[dict[str, Any]],
    direction_dataset: DirectionErrorDataset | None = None,
) -> CampaignStageReport:
    sim = [float(p.get("theme_entropy") or 0) for p in simulated_entropy_points]
    obs = entropy_values_from_snapshots(observed_snapshots)
    steps = max(1, min(len(sim), len(obs)) - 1)
    overall = directional_agreement(sim, obs)

    rows = direction_dataset.by_campaign(campaign_id) if direction_dataset else []
    stage_err = {
        "cold_start": sum(1 for r in rows if r.stage == "cold_start"),
        "growth": sum(1 for r in rows if r.stage == "growth"),
        "plateau": sum(1 for r in rows if r.stage == "plateau"),
    }
    cold = _stage_metrics(sim, obs, stage="cold_start", total_steps=steps, direction_errors=stage_err["cold_start"])
    growth = _stage_metrics(sim, obs, stage="growth", total_steps=steps, direction_errors=stage_err["growth"])
    plateau = _stage_metrics(sim, obs, stage="plateau", total_steps=steps, direction_errors=stage_err["plateau"])
    stages = {"cold_start": cold, "growth": growth, "plateau": plateau}
    worst = min(stages, key=lambda k: stages[k].directional_agreement)

    return CampaignStageReport(
        campaign_id=campaign_id,
        overall_directional=overall,
        cold_start=cold,
        growth=growth,
        plateau=plateau,
        worst_stage=worst,
        direction_error_count=len(rows) or sum(stage_err.values()),
    )


def analyze_weak_campaigns(
    *,
    loo_folds: list[dict[str, Any]],
    campaign_reports: list[CampaignStageReport],
) -> WeakCampaignReport:
    sorted_folds = sorted(
        loo_folds,
        key=lambda f: float(f.get("directional_agreement") or 0),
    )
    sorted_camps = sorted(
        campaign_reports,
        key=lambda c: c.overall_directional,
    )
    stage_totals = {"cold_start": 0, "growth": 0, "plateau": 0}
    for c in campaign_reports:
        stage_totals["cold_start"] += c.cold_start.direction_errors
        stage_totals["growth"] += c.growth.direction_errors
        stage_totals["plateau"] += c.plateau.direction_errors

    worst_fold = sorted_folds[0] if sorted_folds else {}
    worst_camp = sorted_camps[0] if sorted_camps else None
    return WeakCampaignReport(
        campaigns=[c.to_dict() for c in campaign_reports],
        worst_campaign_id=worst_camp.campaign_id if worst_camp else "",
        worst_fold_campaign_id=str(worst_fold.get("held_out_campaign_id") or ""),
        worst_overall_directional=worst_camp.overall_directional if worst_camp else 0.0,
        stage_error_totals=stage_totals,
    )
