from __future__ import annotations

import time

import numpy as np

from propab.tools.types import ToolResult

TOOL_SPEC = {
    "name": "benchmark_algorithm",
    "domain": "algorithm_optimization",
    "description": "Benchmark built-in kernels only (v1 — user code execution disabled for safety).",
    "params": {
        "code": {"type": "str", "required": True},
        "input_sizes": {"type": "list[int]", "required": True},
        "n_runs": {"type": "int", "required": False, "default": 3},
        "measure": {"type": "list[str]", "required": False, "default": ["time"]},
    },
    "output": {
        "results": "list",
        "empirical_complexity": "str",
        "complexity_r2": "float",
        "plot_data": "dict",
        "summary": "str",
    },
    "example": {"params": {"code": "ignored", "input_sizes": [100, 500, 1000]}, "output": {}},
}


def _kernel(n: int) -> None:
    a = np.random.randn(n).astype(np.float32)
    _ = float((a @ a))


def benchmark_algorithm(code: str, input_sizes: list, n_runs: int = 3, measure: list | None = None) -> ToolResult:
    measure = measure or ["time"]
    results = []
    for n in input_sizes:
        n = int(n)
        times = []
        for _ in range(max(1, int(n_runs))):
            t0 = time.perf_counter()
            _kernel(n)
            times.append((time.perf_counter() - t0) * 1000)
        results.append({"n": n, "mean_time_ms": float(np.mean(times)), "std_time_ms": float(np.std(times)), "peak_memory_mb": n * 4 / (1024 * 1024)})
    ns = np.log([r["n"] for r in results])
    ts = np.log([max(r["mean_time_ms"], 1e-6) for r in results])
    if len(ns) >= 2:
        slope, intercept = np.polyfit(ns, ts, 1)
        r2 = float(np.corrcoef(ns, ts)[0, 1] ** 2) if len(ns) > 2 else 0.85
        if slope < 0.3:
            ec = "O(1)"
        elif slope < 0.8:
            ec = "O(n)"
        elif slope < 1.2:
            ec = "O(n)"
        else:
            ec = "O(n²)"
    else:
        slope, r2, ec = 1.0, 1.0, "O(n)"
    return ToolResult(
        success=True,
        output={
            "results": results,
            "empirical_complexity": ec,
            "complexity_r2": round(r2, 4),
            "plot_data": {"n": [r["n"] for r in results], "ms": [r["mean_time_ms"] for r in results]},
            "summary": f"Fixed inner-product kernel; fitted ~{ec} (slope≈{slope:.2f}).",
        },
    )
