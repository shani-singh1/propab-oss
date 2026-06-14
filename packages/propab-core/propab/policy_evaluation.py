"""Policy evaluation — V1 metrics + entropy trajectory dynamics (fixes.md P1)."""
from __future__ import annotations

from typing import Any

from propab.entropy_trajectory import EntropyTrajectorySummary, observed_entropy_dynamics
from propab.meta_science import CampaignObservation
from propab.policy_record import PredictedEffects

# V1 scalar deltas vs baseline
METRICS_V1 = (
    "closure_ratio",
    "compute_efficiency",
    "refute_ratio",
)

# Legacy scalar — used only when no trajectory summary is available
METRIC_LEGACY_ENTROPY = "theme_entropy"

ENTROPY_DYNAMIC_METRICS = (
    "start_H",
    "growth_rate",
    "saturation_H",
    "cross_H_1_5_at_tested",
    "cross_H_2_0_at_tested",
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
    keys = METRICS_V1 + (METRIC_LEGACY_ENTROPY,)
    return {k: cur[k] - base[k] for k in keys}


def tolerance_for_campaign(*, budget_bucket: str, tested: int) -> dict[str, float]:
    """Noise tolerance scales with budget bucket and sample count."""
    base = {"1h": 0.12, "3h": 0.08, "8h": 0.05}.get(budget_bucket, 0.10)
    sample_scale = max(0.5, min(1.5, 20 / max(5, tested)))
    t = base * sample_scale
    cross_scale = max(8.0, 20.0 * sample_scale)
    return {
        "closure_ratio": t,
        "theme_entropy": t * 1.5,
        "compute_efficiency": t * 0.5,
        "refute_ratio": t,
        "start_H": 0.25,
        "growth_rate": 0.03,
        "saturation_H": 0.35,
        "cross_H_1_5_at_tested": cross_scale,
        "cross_H_2_0_at_tested": cross_scale * 1.25,
    }


def compute_entropy_residuals(
    predicted: PredictedEffects,
    observed_dynamics: dict[str, float],
) -> dict[str, float]:
    pred = predicted.to_dict()
    return {
        k: observed_dynamics.get(k, 0) - pred.get(k, 0)
        for k in ENTROPY_DYNAMIC_METRICS
    }


def compute_residuals(
    predicted: PredictedEffects,
    observed_deltas: dict[str, float],
    *,
    observed_entropy: dict[str, float] | None = None,
    use_entropy_dynamics: bool = False,
) -> dict[str, float]:
    pred = predicted.to_dict()
    residuals: dict[str, float] = {
        "closure_ratio": observed_deltas.get("closure_ratio", 0) - pred.get("closure_ratio_delta", 0),
        "compute_efficiency": observed_deltas.get("compute_efficiency", 0) - pred.get("compute_efficiency_delta", 0),
        "refute_ratio": observed_deltas.get("refute_ratio", 0) - pred.get("refute_ratio_delta", 0),
    }
    if use_entropy_dynamics and observed_entropy:
        residuals.update(compute_entropy_residuals(predicted, observed_entropy))
    else:
        residuals[METRIC_LEGACY_ENTROPY] = (
            observed_deltas.get(METRIC_LEGACY_ENTROPY, 0) - pred.get("theme_entropy_delta", 0)
        )
    return residuals


def predictions_within_tolerance(
    residuals: dict[str, float],
    tolerance: dict[str, float],
    *,
    use_entropy_dynamics: bool = False,
) -> bool:
    for k in METRICS_V1:
        if abs(residuals.get(k, 0)) > tolerance.get(k, 0.15):
            return False
    if use_entropy_dynamics:
        for k in ENTROPY_DYNAMIC_METRICS:
            if abs(residuals.get(k, 0)) > tolerance.get(k, 0.15):
                return False
    else:
        if abs(residuals.get(METRIC_LEGACY_ENTROPY, 0)) > tolerance.get(METRIC_LEGACY_ENTROPY, 0.15):
            return False
    return True


def evaluate_candidate_policy(
    *,
    predicted: PredictedEffects,
    baseline_obs: CampaignObservation,
    current_obs: CampaignObservation,
    budget_bucket: str,
    calibration_closure_target: float | None = 0.306,
    trajectory_summary: EntropyTrajectorySummary | dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Accept if prediction residuals within tolerance AND closure did not collapse.
    Entropy: V2 trajectory dynamics when summary present; else legacy theme_entropy_delta.
    """
    observed = metric_deltas(current_obs, baseline_obs)
    use_dynamics = trajectory_summary is not None and predicted.uses_entropy_dynamics()
    observed_entropy = (
        observed_entropy_dynamics(trajectory_summary)
        if trajectory_summary is not None
        else None
    )
    residuals = compute_residuals(
        predicted,
        observed,
        observed_entropy=observed_entropy,
        use_entropy_dynamics=use_dynamics and observed_entropy is not None,
    )
    tol = tolerance_for_campaign(budget_bucket=budget_bucket, tested=current_obs.tested)

    pred_ok = predictions_within_tolerance(
        residuals,
        tol,
        use_entropy_dynamics=use_dynamics and observed_entropy is not None,
    )
    closure_ok = True
    if calibration_closure_target is not None:
        closure_ok = current_obs.closure_ratio >= calibration_closure_target * 0.85

    collapse_ok = current_obs.closure_ratio >= baseline_obs.closure_ratio * 0.5
    accepted = pred_ok and closure_ok and collapse_ok

    detail: dict[str, Any] = {
        "predicted": predicted.to_dict(),
        "observed": observed,
        "residuals": residuals,
        "tolerance": tol,
        "pred_ok": pred_ok,
        "closure_ok": closure_ok,
        "collapse_ok": collapse_ok,
        "accept_or_reject": "ACCEPT" if accepted else "REJECT",
        "entropy_eval_mode": "dynamics" if (use_dynamics and observed_entropy) else "legacy",
    }
    if trajectory_summary is not None:
        if isinstance(trajectory_summary, EntropyTrajectorySummary):
            detail["observed_trajectory"] = trajectory_summary.to_dict()
        else:
            detail["observed_trajectory"] = dict(trajectory_summary)
    return accepted, detail
