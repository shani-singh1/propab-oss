"""Policy evaluation — V1 metrics, predicted vs observed, tolerance bands."""
from __future__ import annotations

from typing import Any

from propab.meta_science import CampaignObservation
from propab.policy_record import PredictedEffects

# V1 metrics only — no proxy metrics per fixes.md
METRICS_V1 = (
    "closure_ratio",
    "theme_entropy",
    "compute_efficiency",
    "refute_ratio",
)


def compute_efficiency(obs: CampaignObservation) -> float:
    return obs.confirmed / max(1, obs.compute_seconds)


def compute_refute_ratio(obs: CampaignObservation) -> float:
    return obs.refuted / max(1, obs.tested)


def observation_metrics(obs: CampaignObservation) -> dict[str, float]:
    return {
        "closure_ratio": float(obs.closure_ratio),
        "theme_entropy": float(obs.theme_entropy),
        "compute_efficiency": compute_efficiency(obs),
        "refute_ratio": compute_refute_ratio(obs),
    }


def metric_deltas(
    current: CampaignObservation,
    baseline: CampaignObservation,
) -> dict[str, float]:
    cur = observation_metrics(current)
    base = observation_metrics(baseline)
    return {k: cur[k] - base[k] for k in METRICS_V1}


def tolerance_for_campaign(*, budget_bucket: str, tested: int) -> dict[str, float]:
    """Noise tolerance scales with budget bucket and sample count."""
    base = {"1h": 0.12, "3h": 0.08, "8h": 0.05}.get(budget_bucket, 0.10)
    sample_scale = max(0.5, min(1.5, 20 / max(5, tested)))
    t = base * sample_scale
    return {
        "closure_ratio": t,
        "theme_entropy": t * 1.5,
        "compute_efficiency": t * 0.5,
        "refute_ratio": t,
    }


def compute_residuals(
    predicted: PredictedEffects,
    observed_deltas: dict[str, float],
) -> dict[str, float]:
    pred = predicted.to_dict()
    return {
        "closure_ratio": observed_deltas.get("closure_ratio", 0) - pred.get("closure_ratio_delta", 0),
        "theme_entropy": observed_deltas.get("theme_entropy", 0) - pred.get("theme_entropy_delta", 0),
        "compute_efficiency": observed_deltas.get("compute_efficiency", 0) - pred.get("compute_efficiency_delta", 0),
        "refute_ratio": observed_deltas.get("refute_ratio", 0) - pred.get("refute_ratio_delta", 0),
    }


def predictions_within_tolerance(
    residuals: dict[str, float],
    tolerance: dict[str, float],
) -> bool:
    return all(abs(residuals.get(k, 0)) <= tolerance.get(k, 0.15) for k in METRICS_V1)


def evaluate_candidate_policy(
    *,
    predicted: PredictedEffects,
    baseline_obs: CampaignObservation,
    current_obs: CampaignObservation,
    budget_bucket: str,
    calibration_closure_target: float | None = 0.306,
) -> tuple[bool, dict[str, Any]]:
    """
    Accept if prediction residuals within tolerance AND closure did not collapse.
    Rejected policies remain in history; accepted never overwritten without evaluation.
    """
    observed = metric_deltas(current_obs, baseline_obs)
    residuals = compute_residuals(predicted, observed)
    tol = tolerance_for_campaign(budget_bucket=budget_bucket, tested=current_obs.tested)

    pred_ok = predictions_within_tolerance(residuals, tol)
    closure_ok = True
    if calibration_closure_target is not None:
        closure_ok = current_obs.closure_ratio >= calibration_closure_target * 0.85

    collapse_ok = current_obs.closure_ratio >= baseline_obs.closure_ratio * 0.5
    accepted = pred_ok and closure_ok and collapse_ok

    return accepted, {
        "predicted": predicted.to_dict(),
        "observed": observed,
        "residuals": residuals,
        "tolerance": tol,
        "pred_ok": pred_ok,
        "closure_ok": closure_ok,
        "collapse_ok": collapse_ok,
        "accept_or_reject": "ACCEPT" if accepted else "REJECT",
    }
