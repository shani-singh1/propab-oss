from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "bootstrap_confidence",
    "domain": "ml_research",
    "description": "Bootstrap CI for mean, median, std, min, max, or percentile of a metric vector.",
    "params": {
        "values": {"type": "list[float]", "required": True},
        "metric": {"type": "str", "required": False, "default": "mean"},
        "percentile": {"type": "float", "required": False, "default": 50.0},
        "n_bootstrap": {"type": "int", "required": False, "default": 10000},
        "ci": {"type": "float", "required": False, "default": 0.95},
    },
    "output": {
        "point_estimate": "float",
        "ci_lower": "float",
        "ci_upper": "float",
        "ci_width": "float",
        "std_error": "float",
    },
    "example": {"params": {"values": [0.1, 0.2, 0.15, 0.18]}, "output": {}},
}


def bootstrap_confidence(
    values: list,
    metric: str = "mean",
    percentile: float = 50.0,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> ToolResult:
    try:
        x = np.array([float(v) for v in values], dtype=np.float64)
        if len(x) < 2:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Need at least 2 values."))
        n_boot = max(500, min(int(n_bootstrap), 50_000))
        ci = float(ci)
        if not 0.5 < ci < 1.0:
            ci = 0.95
        rng = np.random.default_rng(42)
        m = str(metric).lower()

        def stat(arr: np.ndarray) -> float:
            if m == "mean":
                return float(np.mean(arr))
            if m == "median":
                return float(np.median(arr))
            if m == "std":
                return float(np.std(arr, ddof=1))
            if m == "max":
                return float(np.max(arr))
            if m == "min":
                return float(np.min(arr))
            if m == "percentile":
                return float(np.percentile(arr, percentile))
            return float(np.mean(arr))

        point = stat(x)
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(x, size=len(x), replace=True)
            boots.append(stat(sample))
        boots_arr = np.array(boots)
        low_q = (1 - ci) / 2
        hi_q = 1 - low_q
        lo, hi = float(np.quantile(boots_arr, low_q)), float(np.quantile(boots_arr, hi_q))
        se = float(np.std(boots_arr, ddof=1))
        return ToolResult(
            success=True,
            output={
                "point_estimate": round(point, 6),
                "ci_lower": round(lo, 6),
                "ci_upper": round(hi, 6),
                "ci_width": round(hi - lo, 6),
                "std_error": round(se, 6),
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
