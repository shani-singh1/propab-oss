from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "reproduce_result",
    "domain": "ml_research",
    "description": "Compare variance under fixed vs varying RNG seeds (numpy v1; experiment_code ignored).",
    "params": {
        "experiment_code": {"type": "str", "required": True},
        "n_runs": {"type": "int", "required": False, "default": 5},
        "fixed_seed": {"type": "int", "required": False, "default": 42},
    },
    "output": {
        "fixed_seed_results": "list",
        "random_seed_results": "list",
        "fixed_variance": "float",
        "random_variance": "float",
        "reproducibility_score": "float",
        "recommendation": "str",
    },
    "example": {"params": {"experiment_code": "# v1 synthetic", "n_runs": 6}, "output": {}},
}


def reproduce_result(
    experiment_code: str,
    n_runs: int = 5,
    fixed_seed: int = 42,
) -> ToolResult:
    try:
        n = max(2, min(int(n_runs), 200))
        fixed_seed = int(fixed_seed)
        fixed_results: list[float] = []
        for _ in range(n):
            g = np.random.default_rng(fixed_seed)
            fixed_results.append(float(g.normal(0.0, 1.0)))
        random_results: list[float] = []
        for i in range(n):
            g = np.random.default_rng(10_000 + i * 7919)
            random_results.append(float(g.normal(0.0, 1.0)))
        fv = float(np.var(fixed_results, ddof=1)) if n > 1 else 0.0
        rv = float(np.var(random_results, ddof=1)) if n > 1 else 1.0
        denom = fv + rv + 1e-12
        score = float(max(0.0, min(1.0, 1.0 - rv / denom)))
        rec = "Fixed seed yields stable draws; increase runs if random variance stays high." if score > 0.7 else "High run-to-run variance; tighten seeds or reduce noise sources."
        return ToolResult(
            success=True,
            output={
                "fixed_seed_results": [round(x, 6) for x in fixed_results],
                "random_seed_results": [round(x, 6) for x in random_results],
                "fixed_variance": round(fv, 8),
                "random_variance": round(rv, 8),
                "reproducibility_score": round(score, 4),
                "recommendation": rec,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
