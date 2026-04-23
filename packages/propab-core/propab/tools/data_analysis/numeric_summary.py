from __future__ import annotations

import math

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "numeric_summary",
    "domain": "data_analysis",
    "description": "Compute mean, sample standard deviation, min, and max for a numeric list.",
    "params": {
        "values": {
            "type": "list[float]",
            "required": True,
            "description": "Numeric measurements",
        },
    },
    "output": {
        "count": "int",
        "mean": "float",
        "std_sample": "float — Bessel-corrected std (0 if count<2)",
        "min": "float",
        "max": "float",
    },
    "example": {
        "params": {"values": [1.0, 2.0, 3.0, 4.0]},
        "output": {"count": 4, "mean": 2.5, "std_sample": 1.291, "min": 1.0, "max": 4.0},
    },
}


def numeric_summary(values: list) -> ToolResult:
    try:
        if not isinstance(values, list) or not values:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="values must be a non-empty list."),
            )
        xs = [float(x) for x in values]
        n = len(xs)
        mean = sum(xs) / n
        if n < 2:
            std = 0.0
        else:
            var = sum((x - mean) ** 2 for x in xs) / (n - 1)
            std = math.sqrt(var)
        return ToolResult(
            success=True,
            output={
                "count": n,
                "mean": round(mean, 6),
                "std_sample": round(std, 6),
                "min": round(min(xs), 6),
                "max": round(max(xs), 6),
            },
        )
    except (TypeError, ValueError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=str(exc)))
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
