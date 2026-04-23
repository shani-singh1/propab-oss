from __future__ import annotations

import tracemalloc

import numpy as np

from propab.tools.types import ToolResult

TOOL_SPEC = {
    "name": "profile_memory",
    "domain": "algorithm_optimization",
    "description": "Peak allocated memory while building large numpy arrays (v1 proxy).",
    "params": {
        "size_mb_target": {"type": "float", "required": False, "default": 8.0},
    },
    "output": {"peak_mb": "float", "current_mb": "float", "summary": "str"},
    "example": {"params": {"size_mb_target": 2.0}, "output": {}},
}


def profile_memory(size_mb_target: float = 8.0) -> ToolResult:
    tracemalloc.start()
    n = max(100_000, int(size_mb_target * 1024 * 1024 / 8))
    _ = np.zeros(n, dtype=np.float64)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ToolResult(
        success=True,
        output={
            "peak_mb": round(peak / (1024 * 1024), 4),
            "current_mb": round(current / (1024 * 1024), 4),
            "summary": "tracemalloc peak during numpy allocation.",
        },
    )
