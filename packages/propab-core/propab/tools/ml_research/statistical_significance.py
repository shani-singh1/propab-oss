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


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def statistical_significance(
    results_a: list,
    results_b: list,
    test: str = "auto",
    alpha: float = 0.05,
    alternative: str = "two_sided",
) -> ToolResult:
    try:
        a = np.array([float(x) for x in results_a], dtype=np.float64)
        b = np.array([float(x) for x in results_b], dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Need at least 2 samples per group."))
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
                r = stats.wilcoxon(a, b, alternative=scipy_alt)
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
            p = 2 * min(cnt / n_boot, 1 - cnt / n_boot) if alt == "two_sided" else cnt / n_boot
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
