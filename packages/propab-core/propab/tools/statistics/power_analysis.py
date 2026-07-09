"""S1 — statistical power / required sample size (two-sample t, two-proportion).

Power analysis is the guard against over-claiming on an underpowered design: a
non-significant result from an underpowered study is NOT evidence of no effect, and
a design that cannot detect the smallest effect worth caring about cannot support a
positive claim. Given an effect size (Cohen's d for the t-test; Cohen's h — or raw
p1, p2 — for the proportion test), the level ``alpha``, and EITHER ``n`` per group
(-> achieved power) OR a target ``power`` (-> required n per group), this reports
the number AND the assumptions behind it.

Honesty by construction:
  * Exact noncentral-t power for the two-sample t-test (``scipy.stats.nct``);
    normal / arcsine approximation for two proportions (Cohen's h). The method and
    every assumption are returned, never hidden.
  * Refuses nonsensical inputs — ``n < 2``, ``alpha`` not in (0, 1), target
    ``power`` not in (0, 1), proportions not in [0, 1], a zero effect when solving
    for n, or missing effect information -> ``validation_error``.
  * Assumes EQUAL group sizes and (for the t-test) EQUAL variances, and says so.
  * Flags an underpowered design (achieved power < 0.8) so a null result is not
    mis-read as "no effect".
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from propab.tools.types import ToolError, ToolResult

_T_TEST_ALIASES = {
    "two_sample_t", "two-sample-t", "two_sample", "two-sample", "t", "t_test",
    "ttest", "student_t", "students_t", "independent_t", "two_sample_ttest",
}
_PROP_ALIASES = {
    "two_proportion", "two_proportions", "two-proportion", "two-proportions",
    "proportion", "proportions", "prop", "two_prop", "2prop", "z_prop",
}

TOOL_SPEC = {
    "name": "power_analysis",
    "domain": "statistics",
    "audience": "worker",
    # Part of the significance-honesty toolchain: surfaced to significance
    # workflows AND never auto-filled from the spec example (a placeholder effect
    # size would silently manufacture a power / sample-size number).
    "significance_capable": True,
    "description": (
        "Statistical power / required sample size for a two-sample t-test or a "
        "two-proportion z-test. Provide test in {two_sample_t (default), "
        "two_proportion}; an effect_size (Cohen's d for t, Cohen's h for proportions) "
        "OR raw means/sds (t) or p1/p2 (proportions); alpha; and EITHER n per group "
        "(-> achieved power) OR target power (-> required n per group). Reports the "
        "assumptions and flags an underpowered design so a null result is not "
        "mis-read as 'no effect'. Refuses n<2, alpha/power out of (0,1), zero effect."
    ),
    "params": {
        "test": {"type": "str", "required": False, "default": "two_sample_t",
                  "description": "two_sample_t (default) | two_proportion."},
        "effect_size": {"type": "float", "required": False,
                         "description": "Cohen's d (t-test) or Cohen's h (proportion). Optional if raw stats given."},
        "mean1": {"type": "float", "required": False, "description": "Group-1 mean (t-test, with mean2/sd1/sd2)."},
        "mean2": {"type": "float", "required": False, "description": "Group-2 mean (t-test)."},
        "sd1": {"type": "float", "required": False, "description": "Group-1 std dev (t-test)."},
        "sd2": {"type": "float", "required": False, "description": "Group-2 std dev (t-test)."},
        "p1": {"type": "float", "required": False, "description": "Group-1 proportion in [0,1] (proportion test)."},
        "p2": {"type": "float", "required": False, "description": "Group-2 proportion in [0,1] (proportion test)."},
        "alpha": {"type": "float", "required": False, "default": 0.05, "description": "Significance level in (0,1)."},
        "n": {"type": "int", "required": False,
               "description": "Sample size PER GROUP (>=2). Give this to get achieved power."},
        "power": {"type": "float", "required": False,
                   "description": "Target power in (0,1). Give this to get required n per group."},
        "alternative": {"type": "str", "required": False, "default": "two_sided",
                         "description": "two_sided (default) | greater | less (one-sided)."},
    },
    "output": {
        "test": "str — 'two_sample_t' or 'two_proportion'",
        "mode": "str — 'achieved_power' (n given) or 'required_n' (power given)",
        "effect_size": "float — Cohen's d (t) or Cohen's h (proportion) used",
        "effect_size_metric": "str — 'cohens_d' or 'cohens_h'",
        "alpha": "float",
        "alternative": "str",
        "achieved_power": "float — power of the design (at n given, or at required_n)",
        "n_per_group": "int — sample size per group used/required",
        "total_n": "int — 2 * n_per_group",
        "target_power": "float — only in required_n mode",
        "adequately_powered": "bool — achieved_power >= 0.8",
        "assumptions": "list[str] — equal group sizes, variance/approx assumptions",
        "caveat": "str — the honesty note on underpowered designs",
    },
    "example": {
        "params": {"test": "two_sample_t", "effect_size": 0.5, "alpha": 0.05, "power": 0.8},
        "output": {"n_per_group": 64, "achieved_power": 0.8},
    },
}

_CAVEAT = (
    "A non-significant result from an underpowered design is NOT evidence of no "
    "effect. Power/sample-size assume equal group sizes; interpret accordingly."
)


def _normalize_alternative(value: str) -> str | None:
    v = str(value).strip().lower().replace("-", "_")
    if v in ("two_sided", "two_sides", "2_sided", "twosided", "two"):
        return "two_sided"
    if v in ("greater", "larger", "right"):
        return "greater"
    if v in ("less", "smaller", "left"):
        return "less"
    if v in ("one_sided", "onesided", "one"):
        return "greater"  # one-sided in the hypothesized direction
    return None


def _t_power(abs_d: float, n: int, alpha: float, alternative: str) -> float:
    """Achieved power of a two-sample t-test, equal n per group, via noncentral t."""
    df = 2 * n - 2
    ncp = abs_d * math.sqrt(n / 2.0)
    if alternative == "two_sided":
        tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
        power = stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)
    else:  # greater / less -> one-sided of magnitude abs_d in the hypothesized direction
        tcrit = stats.t.ppf(1.0 - alpha, df)
        power = stats.nct.sf(tcrit, df, ncp)
    return float(min(max(power, 0.0), 1.0))


def _prop_power(abs_h: float, n: int, alpha: float, alternative: str) -> float:
    """Achieved power of a two-proportion z-test, equal n per group (arcsine/normal)."""
    lam = abs_h * math.sqrt(n / 2.0)
    if alternative == "two_sided":
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        power = stats.norm.sf(z - lam) + stats.norm.cdf(-z - lam)
    else:
        z = stats.norm.ppf(1.0 - alpha)
        power = stats.norm.sf(z - lam)
    return float(min(max(power, 0.0), 1.0))


def _required_n(power_fn, target: float, n_max: int = 10_000_000) -> int | None:
    """Smallest integer n per group (>=2) with power_fn(n) >= target, else None."""
    if power_fn(2) >= target:
        return 2
    hi = 4
    while power_fn(hi) < target:
        hi *= 2
        if hi > n_max:
            return None
    lo = hi // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if power_fn(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def power_analysis(
    test: str = "two_sample_t",
    effect_size: float | None = None,
    mean1: float | None = None,
    mean2: float | None = None,
    sd1: float | None = None,
    sd2: float | None = None,
    p1: float | None = None,
    p2: float | None = None,
    alpha: float = 0.05,
    n: int | None = None,
    power: float | None = None,
    alternative: str = "two_sided",
) -> ToolResult:
    try:
        test_key = str(test).strip().lower().replace("-", "_")
        if test_key in _T_TEST_ALIASES:
            test_norm = "two_sample_t"
        elif test_key in _PROP_ALIASES:
            test_norm = "two_proportion"
        else:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=f"Unknown test {test!r}. Use 'two_sample_t' or 'two_proportion'.",
                ),
            )

        alt = _normalize_alternative(alternative)
        if alt is None:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=f"Unknown alternative {alternative!r}. Use 'two_sided', 'greater', or 'less'.",
                ),
            )

        try:
            a = float(alpha)
        except (TypeError, ValueError):
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be numeric; got {alpha!r}."))
        if not np.isfinite(a) or not (0.0 < a < 1.0):
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"alpha must be in (0, 1); got {a}."))

        # Exactly one of n (achieved power) / power (required n) must be supplied.
        if n is None and power is None:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Provide either n (-> achieved power) or power (-> required n)."),
            )
        if n is not None and power is not None:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Provide only ONE of n or power, not both."),
            )

        # ---- Resolve the effect size (and its metric) for the chosen test. ----
        if test_norm == "two_sample_t":
            metric = "cohens_d"
            if effect_size is not None:
                try:
                    eff = float(effect_size)
                except (TypeError, ValueError):
                    return ToolResult(success=False, error=ToolError(type="validation_error", message=f"effect_size must be numeric; got {effect_size!r}."))
            elif None not in (mean1, mean2, sd1, sd2):
                try:
                    m1, m2, s1, s2 = float(mean1), float(mean2), float(sd1), float(sd2)
                except (TypeError, ValueError):
                    return ToolResult(success=False, error=ToolError(type="validation_error", message="mean1/mean2/sd1/sd2 must be numeric."))
                if s1 < 0 or s2 < 0:
                    return ToolResult(success=False, error=ToolError(type="validation_error", message="Standard deviations must be non-negative."))
                pooled = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
                if pooled <= 0.0:
                    return ToolResult(success=False, error=ToolError(type="validation_error", message="Pooled SD is zero; cannot form an effect size."))
                eff = (m1 - m2) / pooled
            else:
                return ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message="Provide effect_size (Cohen's d) OR mean1, mean2, sd1, sd2 for the t-test."),
                )
        else:  # two_proportion
            metric = "cohens_h"
            if p1 is not None and p2 is not None:
                try:
                    pp1, pp2 = float(p1), float(p2)
                except (TypeError, ValueError):
                    return ToolResult(success=False, error=ToolError(type="validation_error", message="p1/p2 must be numeric."))
                if not (0.0 <= pp1 <= 1.0) or not (0.0 <= pp2 <= 1.0):
                    return ToolResult(success=False, error=ToolError(type="validation_error", message="p1 and p2 must be proportions in [0, 1]."))
                # Cohen's h via the arcsine (variance-stabilizing) transform.
                eff = 2.0 * math.asin(math.sqrt(pp1)) - 2.0 * math.asin(math.sqrt(pp2))
            elif effect_size is not None:
                try:
                    eff = float(effect_size)  # interpreted as Cohen's h
                except (TypeError, ValueError):
                    return ToolResult(success=False, error=ToolError(type="validation_error", message=f"effect_size must be numeric; got {effect_size!r}."))
            else:
                return ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message="Provide p1 and p2 (proportions) OR effect_size (Cohen's h) for the proportion test."),
                )

        if not np.isfinite(eff):
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Effect size is not finite."))
        abs_eff = abs(eff)

        power_fn = (
            (lambda k: _t_power(abs_eff, k, a, alt))
            if test_norm == "two_sample_t"
            else (lambda k: _prop_power(abs_eff, k, a, alt))
        )

        assumptions = [
            "Equal sample size per group.",
            "Two-sample t-test with equal variances (pooled SD)." if test_norm == "two_sample_t"
            else "Two-proportion z-test via the arcsine (Cohen's h) normal approximation.",
            f"Alternative: {alt}.",
        ]

        # ---- Mode A: n given -> achieved power. ----
        if n is not None:
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                return ToolResult(success=False, error=ToolError(type="validation_error", message=f"n must be an integer >= 2; got {n!r}."))
            if n_int < 2:
                return ToolResult(success=False, error=ToolError(type="validation_error", message=f"n per group must be >= 2; got {n_int}."))
            achieved = power_fn(n_int)
            return ToolResult(
                success=True,
                output={
                    "test": test_norm,
                    "mode": "achieved_power",
                    "effect_size": round(float(eff), 6),
                    "effect_size_metric": metric,
                    "alpha": a,
                    "alternative": alt,
                    "achieved_power": round(achieved, 6),
                    "n_per_group": n_int,
                    "total_n": 2 * n_int,
                    "adequately_powered": bool(achieved >= 0.8),
                    "assumptions": assumptions,
                    "caveat": _CAVEAT,
                },
            )

        # ---- Mode B: target power given -> required n per group. ----
        try:
            target = float(power)
        except (TypeError, ValueError):
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"power must be numeric in (0, 1); got {power!r}."))
        if not np.isfinite(target) or not (0.0 < target < 1.0):
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Target power must be in (0, 1); got {target}."))
        if abs_eff <= 1e-12:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=(
                        "Effect size is zero — no finite sample size yields power above alpha. "
                        "A design cannot detect a zero effect."
                    ),
                ),
            )

        req = _required_n(power_fn, target)
        if req is None:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=(
                        "Required n per group exceeds 1e7 — the effect is too small "
                        "(and/or target power too high) for a feasible design."
                    ),
                ),
            )
        achieved = power_fn(req)
        return ToolResult(
            success=True,
            output={
                "test": test_norm,
                "mode": "required_n",
                "effect_size": round(float(eff), 6),
                "effect_size_metric": metric,
                "alpha": a,
                "alternative": alt,
                "target_power": round(target, 6),
                "achieved_power": round(achieved, 6),
                "n_per_group": int(req),
                "total_n": 2 * int(req),
                "adequately_powered": bool(achieved >= 0.8),
                "assumptions": assumptions,
                "caveat": _CAVEAT,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
