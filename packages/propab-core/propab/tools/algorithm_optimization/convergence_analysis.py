from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "convergence_analysis",
    "domain": "algorithm_optimization",
    "description": "Analyse a 1D loss curve for plateaus and oscillation (numpy).",
    "params": {
        "loss_curve": {"type": "list[float]", "required": True},
        "theoretical_rate": {"type": "float", "required": False},
    },
    "output": {
        "empirical_rate": "float",
        "plateau_steps": "list",
        "oscillation_score": "float",
        "convergence_type": "str",
        "effective_steps": "int",
        "summary": "str",
    },
    "example": {"params": {"loss_curve": [1.0, 0.5, 0.25, 0.2, 0.19, 0.18]}, "output": {}},
}


def convergence_analysis(loss_curve: list, theoretical_rate: float | None = None) -> ToolResult:
    try:
        y = np.array([float(x) for x in loss_curve], dtype=np.float64)
        if len(y) < 3:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="loss_curve too short."))
        dy = np.abs(np.diff(y))
        plateau_steps = [int(i) for i in range(1, len(dy)) if dy[i - 1] < 1e-6][:20]
        osc = float(np.std(np.diff(y)) / (np.mean(np.abs(np.diff(y))) + 1e-9))
        osc = min(1.0, osc / 5.0)
        emp = float(-np.polyfit(np.log(np.arange(2, len(y) + 1)), np.log(y[1:] + 1e-12), 1)[0]) if len(y) > 4 else 0.5
        ctype = "oscillating" if osc > 0.35 else "linear" if emp < 1.2 else "sublinear"
        eff = len(y)
        for i in range(10, len(y)):
            window = y[i - 10 : i]
            if float(np.ptp(window) / (abs(float(y[i])) + 1e-9)) < 0.01:
                eff = i
                break
        return ToolResult(
            success=True,
            output={
                "empirical_rate": round(emp, 4),
                "plateau_steps": plateau_steps,
                "oscillation_score": round(osc, 4),
                "convergence_type": ctype,
                "effective_steps": eff,
                "summary": f"type={ctype}, oscillation={osc:.2f}.",
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
