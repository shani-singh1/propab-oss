from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "numerical_stability_test",
    "domain": "algorithm_optimization",
    "description": "Stress linear solves across dtypes vs a float64 reference (v1; user code param ignored for safety).",
    "params": {
        "code": {"type": "str", "required": True},
        "input_range": {"type": "list[float]", "required": False, "default": [1e-4, 1e4]},
        "dtypes": {"type": "list[str]", "required": False, "default": ["float32", "float64"]},
        "condition_number_check": {"type": "bool", "required": False, "default": True},
    },
    "output": {
        "stable_range": "list",
        "precision_loss_at": "list",
        "condition_number": "float",
        "dtype_comparison": "list",
        "recommendation": "str",
    },
    "example": {"params": {"code": "# ignored in v1"}, "output": {}},
}


def numerical_stability_test(
    code: str,
    input_range: list | None = None,
    dtypes: list | None = None,
    condition_number_check: bool = True,
) -> ToolResult:
    try:
        rng = np.random.default_rng(0)
        lo, hi = 1e-4, 1e4
        if input_range and len(input_range) >= 2:
            lo, hi = float(input_range[0]), float(input_range[1])
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 10)
        diag = np.logspace(np.log10(lo), np.log10(hi), 6)
        u, _, vt = np.linalg.svd(rng.standard_normal((6, 6)))
        a = (u * diag) @ vt
        b = rng.standard_normal(6)
        x64 = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
        cn = float(np.linalg.cond(a.astype(np.float64))) if condition_number_check else 0.0
        dts = dtypes or ["float32", "float64"]
        rows = []
        max_err = 0.0
        for name in dts:
            dt = np.dtype(str(name).lower())
            if dt not in (np.float32, np.float64):
                continue
            xt = np.linalg.solve(a.astype(dt), b.astype(dt)).astype(np.float64)
            rel = float(np.max(np.abs(xt - x64) / (np.abs(x64) + 1e-12)))
            max_err = max(max_err, rel)
            rows.append({"dtype": str(dt), "max_relative_error": round(rel, 6), "stable": rel < 1e-3})
        prec_loss = [float(hi)] if max_err > 1e-2 else []
        rec = "Use float64 for this conditioning pattern." if max_err > 1e-3 else "float32 is adequate for this synthetic system."
        return ToolResult(
            success=True,
            output={
                "stable_range": [float(lo), float(hi)],
                "precision_loss_at": prec_loss,
                "condition_number": round(cn, 4),
                "dtype_comparison": rows,
                "recommendation": rec,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
