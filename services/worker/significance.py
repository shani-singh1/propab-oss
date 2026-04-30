from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SignificanceResult:
    gate_passed: bool = False
    gate_definitively_failed: bool = False
    p_value: float | None = None
    effect_size: float | None = None
    confidence_interval: list[float] | None = None
    n_observations: int = 0
    method: str | None = None


# Keys the scanner looks for in tool outputs
_P_VALUE_KEYS = ("p_value", "p", "pvalue", "p_val")
_EFFECT_KEYS = ("effect_size", "cohens_d", "d", "eta_squared", "omega_squared")
_CI_KEYS = ("confidence_interval", "ci", "interval", "ci_lower")  # ci_lower triggers pair search


def _scan_float(payload: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for k in keys:
        v = payload.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _scan_ci(payload: Any) -> list[float] | None:
    if not isinstance(payload, dict):
        return None
    for k in ("confidence_interval", "ci", "interval"):
        v = payload.get(k)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
            return [float(v[0]), float(v[1])]
    # Separate ci_lower / ci_upper pattern
    lo = payload.get("ci_lower")
    hi = payload.get("ci_upper")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return [float(lo), float(hi)]
    return None


def _intervals_overlap(ci: list[float], null: list[float]) -> bool:
    """True if ci overlaps the null interval (e.g. [0,0] for no-difference)."""
    lo, hi = ci[0], ci[1]
    nlo, nhi = null[0], null[1]
    return lo <= nhi and nlo <= hi


def check_significance(results: list[dict[str, Any]]) -> SignificanceResult:
    """
    Scan a list of successful tool outputs for any statistical evidence.

    Gate passes if ANY of:
      - p < 0.05
      - |effect_size| > 0.2
      - confidence interval that does not overlap zero

    Gate definitively fails if:
      - At least one statistical test ran AND p >= 0.30 AND effect small

    Gate is pending (not passed, not definitively failed) if no stat test has run yet.
    """
    p_value: float | None = None
    effect_size: float | None = None
    ci: list[float] | None = None
    n_obs = 0

    stat_test_ran = False

    for out in results:
        if not isinstance(out, dict):
            continue
        n_obs += 1

        p = _scan_float(out, _P_VALUE_KEYS)
        if p is not None:
            stat_test_ran = True
            if p_value is None or p < p_value:
                p_value = p

        es = _scan_float(out, _EFFECT_KEYS)
        if es is not None:
            if effect_size is None or abs(es) > abs(effect_size):
                effect_size = es

        ci_candidate = _scan_ci(out)
        if ci_candidate is not None:
            ci = ci_candidate

    # Determine gate status
    gate_passed = False
    method = None

    if p_value is not None and p_value < 0.05:
        gate_passed = True
        method = "p_value"
    if effect_size is not None and abs(effect_size) > 0.2:
        gate_passed = True
        method = method or "effect_size"
    if ci is not None and not _intervals_overlap(ci, [-1e-9, 1e-9]):
        gate_passed = True
        method = method or "confidence_interval"

    gate_definitively_failed = (
        stat_test_ran
        and not gate_passed
        and p_value is not None
        and p_value >= 0.30
        and (effect_size is None or abs(effect_size) < 0.05)
    )

    return SignificanceResult(
        gate_passed=gate_passed,
        gate_definitively_failed=gate_definitively_failed,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=ci,
        n_observations=n_obs,
        method=method,
    )


def fisher_combine_p_values(p_values: list[float]) -> float:
    """Fisher's method: combine independent p-values from multiple rounds."""
    valid = [p for p in p_values if 0 < p <= 1.0]
    if not valid:
        return 1.0
    chi2 = -2.0 * sum(math.log(p) for p in valid)
    df = 2 * len(valid)
    # Approximate chi2 survival function via regularized incomplete gamma
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore
        return float(chi2_dist.sf(chi2, df))
    except Exception:
        # Fallback: if chi2 is large relative to df, p is small
        return 1.0 / (1.0 + chi2 / df)


_SIGNIFICANCE_TOOL_NAMES = frozenset({
    "statistical_significance",
    "bootstrap_confidence",
    "literature_baseline_compare",
})


def any_significance_tool_ran(tool_names: list[str]) -> bool:
    return any(n in _SIGNIFICANCE_TOOL_NAMES for n in tool_names)
