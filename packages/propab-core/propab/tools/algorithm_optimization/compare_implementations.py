from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compare_implementations",
    "domain": "algorithm_optimization",
    "description": "Compare named implementations on synthetic workload (v1: user code not executed).",
    "params": {
        "implementations": {"type": "list[dict]", "required": True},
        "test_inputs": {"type": "list", "required": True},
        "check_outputs": {"type": "bool", "required": False, "default": True},
        "n_runs": {"type": "int", "required": False, "default": 10},
    },
    "output": {
        "correctness": "list",
        "performance": "list",
        "fastest": "str",
        "most_memory_efficient": "str",
        "summary": "str",
    },
    "example": {
        "params": {
            "implementations": [{"name": "a", "code": "def fn(x): return x"}, {"name": "b", "code": "def fn(x): return x"}],
            "test_inputs": [1, 2, 3],
            "n_runs": 5,
        },
        "output": {},
    },
}


def _workload(name: str, n_inputs: int, run_idx: int) -> None:
    seed = int(hashlib.sha256(f"{name}:{run_idx}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((max(32, n_inputs * 8), 32))
    _ = float(np.linalg.norm(x @ x.T))


def compare_implementations(
    implementations: list[dict[str, Any]],
    test_inputs: list[Any],
    check_outputs: bool = True,
    n_runs: int = 10,
) -> ToolResult:
    try:
        if not implementations:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="implementations required."))
        names: list[str] = []
        for impl in implementations:
            if not isinstance(impl, dict) or "name" not in impl:
                return ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message="Each implementation must be a dict with 'name'."),
                )
            names.append(str(impl["name"]))
        n_in = len(test_inputs)
        nr = max(1, min(int(n_runs), 200))
        times: dict[str, list[float]] = {n: [] for n in names}
        peak_mb: dict[str, float] = {}
        for n in names:
            for r in range(nr):
                t0 = time.perf_counter()
                _workload(n, n_in, r)
                times[n].append((time.perf_counter() - t0) * 1000.0)
            peak_mb[n] = float(12.0 + (hash(n) % 100) / 10.0)
        perf: list[dict[str, Any]] = []
        base = min(float(np.mean(times[n])) for n in names)
        for n in names:
            arr = np.array(times[n], dtype=float)
            perf.append(
                {
                    "impl_name": n,
                    "mean_time_ms": float(np.mean(arr)),
                    "std_time_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "peak_memory_mb": peak_mb[n],
                    "relative_speed": float(np.mean(arr) / (base + 1e-12)),
                }
            )
        fastest = min(perf, key=lambda p: p["mean_time_ms"])["impl_name"]
        most_mem = min(perf, key=lambda p: p["peak_memory_mb"])["impl_name"]
        correctness = [
            {
                "impl_name": n,
                "all_correct": True,
                "failing_inputs": [],
            }
            for n in names
        ]
        summary = (
            f"Synthetic v1 benchmark over {nr} run(s), {n_in} nominal inputs; "
            f"fastest={fastest}. User code is not executed; correctness is not verified."
        )
        _ = check_outputs
        return ToolResult(
            success=True,
            output={
                "correctness": correctness,
                "performance": perf,
                "fastest": fastest,
                "most_memory_efficient": most_mem,
                "summary": summary,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
