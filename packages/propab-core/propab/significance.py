from __future__ import annotations

import math
from dataclasses import dataclass
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


_P_VALUE_KEYS = ("p_value", "p", "pvalue", "p_val")
_EFFECT_KEYS = ("effect_size", "cohens_d", "d", "eta_squared", "omega_squared")


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
    lo = payload.get("ci_lower")
    hi = payload.get("ci_upper")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        return [float(lo), float(hi)]
    return None


def _intervals_overlap(ci: list[float], null: list[float]) -> bool:
    lo, hi = ci[0], ci[1]
    nlo, nhi = null[0], null[1]
    return lo <= nhi and nlo <= hi


def check_significance(results: list[dict[str, Any]]) -> SignificanceResult:
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


def scan_verification(results: list[dict[str, Any]]) -> tuple[int, int]:
    n_true = 0
    n_false = 0
    for out in results:
        if not isinstance(out, dict):
            continue
        v = out.get("verified")
        if isinstance(v, bool):
            if v:
                n_true += 1
            else:
                n_false += 1
        ce = out.get("counterexample")
        if ce not in (None, "", [], {}, False):
            n_false += 1
    return n_true, n_false


def classify_verdict(
    evidence: dict[str, Any],
    sig_result: SignificanceResult,
    *,
    min_metric_steps_for_confirm: int = 2,
    relevance_min: float = 0.12,
) -> tuple[str, str]:
    min_steps = max(1, int(min_metric_steps_for_confirm))

    vt = int(evidence.get("verified_true_steps") or 0)
    vf = int(evidence.get("verified_false_steps") or 0)
    if vf > 0:
        return "refuted", "deterministic counterexample found (verified=false)"
    if vt > 0:
        if vt >= min_steps:
            return (
                "confirmed",
                f"deterministic verification reproduced ({vt} independent checks, verified=true)",
            )
        return (
            "inconclusive",
            f"needs replication: verified once but unreplicated ({vt} check; need >= {min_steps} to confirm)",
        )

    n_metric = int(evidence.get("n_metric_steps") or 0)
    if n_metric == 0:
        return "inconclusive", "no metric-bearing steps executed"
    if sig_result.gate_definitively_failed:
        return (
            "refuted",
            "significance test ran and found no effect (p >= 0.30, negligible effect size)",
        )
    if not sig_result.gate_passed:
        return (
            "inconclusive",
            "no significance evidence: p_value/effect_size/CI not produced or not decisive",
        )

    relevance = float(evidence.get("relevance_score") or 0.0)
    supports = False
    if evidence.get("p_value") is not None:
        supports = bool(
            evidence.get("delta") is not None
            and evidence["p_value"] < 0.05
            and relevance >= relevance_min
        )
    elif evidence.get("effect_size") is not None:
        supports = bool(abs(evidence["effect_size"]) > 0.2 and relevance >= relevance_min)
    elif evidence.get("delta_pct") is not None:
        supports = bool(abs(evidence["delta_pct"]) >= 2.0 and relevance >= relevance_min)

    if not supports:
        return "inconclusive", "significance gate passed but metric direction ambiguous"

    if n_metric < min_steps:
        return (
            "inconclusive",
            (
                f"needs replication: significance supports the hypothesis but result is "
                f"unreplicated ({n_metric} metric step; need >= {min_steps} to confirm)"
            ),
        )
    return (
        "confirmed",
        "significance gate passed; metric direction supports hypothesis and is replicated",
    )


def fisher_combine_p_values(p_values: list[float]) -> float:
    valid = [p for p in p_values if 0 < p <= 1.0]
    if not valid:
        return 1.0
    chi2 = -2.0 * sum(math.log(p) for p in valid)
    df = 2 * len(valid)
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore
        return float(chi2_dist.sf(chi2, df))
    except Exception:
        return 1.0 / (1.0 + chi2 / df)


_SIGNIFICANCE_TOOL_NAMES = frozenset({
    "statistical_significance",
    "bootstrap_confidence",
    "literature_baseline_compare",
})


def any_significance_tool_ran(tool_names: list[str]) -> bool:
    return any(n in _SIGNIFICANCE_TOOL_NAMES for n in tool_names)
