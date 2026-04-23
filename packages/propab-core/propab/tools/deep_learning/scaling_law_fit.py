from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "scaling_law_fit",
    "domain": "deep_learning",
    "description": "Fit log–log power law between scale (e.g. parameter count) and loss from tabular points (numpy v1).",
    "params": {
        "data_points": {"type": "list[dict]", "required": True},
        "fit_type": {"type": "str", "required": False, "default": "power_law"},
        "predict_at": {"type": "list[dict]", "required": False},
    },
    "output": {
        "fit_params": "dict",
        "r_squared": "float",
        "law_str": "str",
        "predictions": "list",
        "plot_data": "dict",
    },
    "example": {
        "params": {
            "data_points": [
                {"model_params": 1e6, "loss": 2.5},
                {"model_params": 4e6, "loss": 2.1},
                {"model_params": 16e6, "loss": 1.8},
            ]
        },
        "output": {},
    },
}


def _scale(row: dict) -> float | None:
    for k in ("model_params", "n_params", "N", "params", "flops"):
        if k in row and row[k] is not None:
            try:
                v = float(row[k])
                return v if v > 0 else None
            except (TypeError, ValueError):
                continue
    return None


def scaling_law_fit(
    data_points: list,
    fit_type: str = "power_law",
    predict_at: list | None = None,
) -> ToolResult:
    try:
        if fit_type not in ("power_law", "chinchilla", "neural_scaling"):
            return ToolResult(success=False, error=ToolError(type="validation_error", message="v1 implements power_law-style log–log fit only."))
        ns: list[float] = []
        ls: list[float] = []
        for row in data_points:
            if not isinstance(row, dict):
                continue
            n = _scale(row)
            loss = row.get("loss")
            if n is None or loss is None:
                continue
            try:
                lf = float(loss)
            except (TypeError, ValueError):
                continue
            if lf <= 0:
                continue
            ns.append(n)
            ls.append(lf)
        if len(ns) < 2:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Need at least 2 valid {scale, loss} points with positive values."))
        x = np.log(np.array(ns, dtype=np.float64))
        y = np.log(np.array(ls, dtype=np.float64))
        a, b = np.polyfit(x, y, 1)
        y_hat = a + b * x
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        law_str = f"log(loss) ≈ {a:.4f} + {b:.4f} * log(scale)"
        preds = []
        if predict_at:
            for row in predict_at:
                if not isinstance(row, dict):
                    continue
                n2 = _scale(row)
                if n2 is None:
                    continue
                preds.append({"scale": n2, "predicted_loss": float(np.exp(a + b * np.log(n2)))})
        return ToolResult(
            success=True,
            output={
                "fit_params": {"intercept": round(float(a), 6), "exponent": round(float(b), 6)},
                "r_squared": round(r2, 6),
                "law_str": law_str,
                "predictions": preds,
                "plot_data": {"log_scale": x.tolist(), "log_loss": y.tolist()},
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
