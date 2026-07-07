from __future__ import annotations

import numpy as np
from scipy import stats

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "statistical_significance",
    "domain": "ml_research",
    "significance_capable": True,
    "description": "Compare two result vectors (t-test, Wilcoxon, or bootstrap difference of means). Produces p_value and effect_size.",
    "params": {
        "results_a": {"type": "list[float]", "required": True},
        "results_b": {"type": "list[float]", "required": True},
        "test": {"type": "str", "required": False, "default": "auto"},
        "alpha": {"type": "float", "required": False, "default": 0.05},
        "alternative": {"type": "str", "required": False, "default": "two_sided"},
    },
    "output": {
        "p_value": "float",
        "statistic": "float",
        "significant": "bool",
        "effect_size": "float",
        "effect_magnitude": "str",
        "test_used": "str",
        "confidence_interval": "list[float]",
        "recommendation": "str",
    },
    "example": {"params": {"results_a": [0.9, 0.88, 0.91], "results_b": [0.82, 0.8, 0.79]}, "output": {}},
}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2 + 1e-12))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _has_zero_within_group_variance(a: np.ndarray, b: np.ndarray) -> bool:
    """True when a group is constant (including bitwise-identical cached replicates)."""
    if a.size < 2 or b.size < 2:
        return False
    sa = float(np.std(a, ddof=1))
    sb = float(np.std(b, ddof=1))
    identical_a = bool(np.all(a == a[0]))
    identical_b = bool(np.all(b == b[0]))
    return (
        sa < 1e-14
        and sb < 1e-14
        and identical_a
        and identical_b
    )


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


# Known leaked placeholder from tool-spec / registry fills (three identical draws × 3 metrics).
_SPEC_LEAK_VECTORS: tuple[tuple[float, ...], ...] = (
    (
        79.648735,
        79.648735,
        79.648735,
        54.044083,
        54.044083,
        54.044083,
        63.826187,
        63.826187,
        63.826187,
    ),
)


def _tuple_rounded(vals: list, nd: int = 6) -> tuple[float, ...]:
    return tuple(round(float(x), nd) for x in vals)


def statistical_significance(
    results_a: list,
    results_b: list,
    test: str = "auto",
    alpha: float = 0.05,
    alternative: str = "two_sided",
) -> ToolResult:
    try:
        ta = _tuple_rounded(list(results_a))
        tb = _tuple_rounded(list(results_b))
        if ta in _SPEC_LEAK_VECTORS or tb in _SPEC_LEAK_VECTORS:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=(
                        "Detected known placeholder metric vector (spec/registry leak). "
                        "Pass real measurements from train_model / run_experiment_grid output."
                    ),
                ),
            )
        a = np.array([float(x) for x in results_a], dtype=np.float64)
        b = np.array([float(x) for x in results_b], dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Need at least 2 samples per group."))
        if _has_zero_within_group_variance(a, b):
            return ToolResult(
                success=False,
                error=ToolError(
                    type="zero_variance",
                    message=(
                        "Both samples are bitwise-constant across replicates (zero within-group variance). "
                        "This usually means memoized/stale measurements, not independent runs — "
                        "refuse fake p-values; rerun experiments with distinct seeds/configs."
                    ),
                ),
            )
        alt = alternative if alternative in ("two_sided", "greater", "less") else "two_sided"
        scipy_alt = "two-sided" if alt == "two_sided" else alt
        test_l = str(test).lower()
        if test_l == "auto":
            if len(a) == len(b) and len(a) >= 5:
                test_l = "wilcoxon"
            else:
                test_l = "t_test"
        d = _cohens_d(a, b)
        stat = 0.0
        p = 1.0
        used = test_l
        if test_l == "t_test":
            r = stats.ttest_ind(a, b, equal_var=False, alternative=scipy_alt)
            stat, p = float(r.statistic), float(r.pvalue)
        elif test_l == "wilcoxon":
            if len(a) != len(b):
                r = stats.mannwhitneyu(a, b, alternative=scipy_alt)
                used = "mannwhitneyu"
            else:
                try:
                    r = stats.wilcoxon(a, b, alternative=scipy_alt)
                except ValueError as exc:
                    msg = str(exc).lower()
                    if "zero" in msg or "identical" in msg:
                        return ToolResult(
                            success=False,
                            error=ToolError(
                                type="zero_variance",
                                message=(
                                    "Wilcoxon unavailable: paired differences are all zero "
                                    "(identical repeats / cache collision suspected)."
                                ),
                            ),
                        )
                    raise
            stat, p = float(r.statistic), float(r.pvalue)
        elif test_l == "bootstrap":
            rng = np.random.default_rng(42)
            n_boot = 4000
            obs = np.mean(a) - np.mean(b)
            pooled = np.concatenate([a, b])
            cnt = 0
            for _ in range(n_boot):
                a_ = rng.choice(pooled, size=len(a), replace=True)
                b_ = rng.choice(pooled, size=len(b), replace=True)
                if np.mean(a_) - np.mean(b_) >= obs:
                    cnt += 1
            # `cnt` counts the UPPER tail (resampled diff >= observed). Pick the
            # tail matching the alternative — for "less" it is the LOWER tail,
            # NOT `cnt/n_boot` (that inverted the p-value for one-sided-less).
            frac_ge = cnt / n_boot
            if alt == "two_sided":
                p = 2 * min(frac_ge, 1 - frac_ge)
            elif alt == "less":
                p = 1 - frac_ge
            else:  # "greater"
                p = frac_ge
            stat = obs
        elif test_l == "mcnemar":
            return ToolResult(success=False, error=ToolError(type="validation_error", message="mcnemar requires paired binary outcomes — not supported in v1."))
        else:
            return ToolResult(success=False, error=ToolError(type="validation_error", message=f"Unknown test: {test}"))
        sig = p < float(alpha)
        # 95% CI for mean difference (normal approx)
        diff_se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
        z = 1.96
        mean_diff = float(np.mean(a) - np.mean(b))
        ci = [mean_diff - z * diff_se, mean_diff + z * diff_se]
        rec = "Treat difference as statistically supported." if sig else "Difference may be due to chance; collect more runs."
        return ToolResult(
            success=True,
            output={
                "p_value": round(p, 6),
                "statistic": round(stat, 6),
                "significant": sig,
                "effect_size": round(d, 4),
                "effect_magnitude": _effect_label(d),
                "test_used": used,
                "confidence_interval": [round(ci[0], 6), round(ci[1], 6)],
                "recommendation": rec,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
