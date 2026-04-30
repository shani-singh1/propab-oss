from __future__ import annotations

import numpy as np
from scipy import stats

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "literature_baseline_compare",
    "domain": "ml_research",
    "significance_capable": True,
    "description": "Compare repeated experimental results to a literature baseline (t-test vs baseline mean). Produces p_value and effect_size.",
    "params": {
        "our_results": {"type": "list[float]", "required": True},
        "baseline_value": {"type": "float", "required": True},
        "baseline_std": {"type": "float", "required": False},
        "metric_direction": {"type": "str", "required": False, "default": "lower_is_better"},
        "claim": {"type": "str", "required": False},
    },
    "output": {
        "our_mean": "float",
        "our_ci": "list[float]",
        "improvement_pct": "float",
        "p_value": "float",
        "significant": "bool",
        "conclusion": "str",
    },
    "example": {
        "params": {
            "our_results": [0.42, 0.44, 0.41],
            "baseline_value": 0.5,
            "metric_direction": "lower_is_better",
        },
        "output": {},
    },
}


def literature_baseline_compare(
    our_results: list[float],
    baseline_value: float,
    baseline_std: float | None = None,
    metric_direction: str = "lower_is_better",
    claim: str | None = None,
) -> ToolResult:
    try:
        x = np.asarray(our_results, dtype=float).ravel()
        if x.size < 2:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Need at least two values in our_results."),
            )
        b = float(baseline_value)
        md = str(metric_direction).lower()
        if md not in ("lower_is_better", "higher_is_better"):
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="metric_direction must be lower_is_better or higher_is_better."),
            )
        mean = float(np.mean(x))
        sem = float(stats.sem(x))
        df = int(x.size - 1)
        t_crit = float(stats.t.ppf(0.975, df))
        margin = t_crit * sem
        our_ci = [float(mean - margin), float(mean + margin)]
        if baseline_std is not None and float(baseline_std) > 0:
            z = (mean - b) / (float(baseline_std) + 1e-12)
            p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        else:
            res = stats.ttest_1samp(x, popmean=b, alternative="two-sided")
            p_value = float(res.pvalue)
        significant = p_value < 0.05
        if md == "lower_is_better":
            improvement_pct = float((b - mean) / (abs(b) + 1e-12) * 100.0)
        else:
            improvement_pct = float((mean - b) / (abs(b) + 1e-12) * 100.0)
        direction_word = "better" if improvement_pct > 0 else "worse"
        ctext = f" (claim: {claim})" if claim else ""
        conclusion = (
            f"Mean {mean:.4g} vs baseline {b:.4g}; ~{abs(improvement_pct):.1f}% {direction_word} than baseline"
            f"{ctext}; {'statistically significant' if significant else 'not significant'} at α=0.05 (p={p_value:.3g})."
        )
        return ToolResult(
            success=True,
            output={
                "our_mean": mean,
                "our_ci": our_ci,
                "improvement_pct": improvement_pct,
                "p_value": p_value,
                "significant": significant,
                "conclusion": conclusion,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
