from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "run_experiment_grid",
    "domain": "ml_research",
    "description": "Evaluate all combinations of numeric grid axes with a deterministic synthetic score (v1; user code execution disabled).",
    "params": {
        "experiment_code": {"type": "str", "required": True},
        "grid": {"type": "dict", "required": True},
        "n_repeats": {"type": "int", "required": False, "default": 3},
        "maximize": {"type": "bool", "required": False, "default": False},
    },
    "output": {
        "results": "list",
        "best_config": "dict",
        "best_score": "float",
        "interaction_effects": "dict",
        "total_runs": "int",
    },
    "example": {
        "params": {
            "experiment_code": "# reserved",
            "grid": {"lr": [0.001, 0.01], "batch_size": [16, 32]},
            "n_repeats": 2,
        },
        "output": {},
    },
}


def _score_config(cfg: dict[str, Any], maximize: bool) -> float:
    """Deterministic proxy: prefer moderate lr and batch in (24,48) sweet spot."""
    lr = float(cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    bs = float(cfg.get("batch_size", cfg.get("bs", 32)))
    base = 1.0 / (1.0 + abs(np.log10(lr) + 3)) + 1.0 / (1.0 + abs(bs - 32) / 32)
    noise_amp = 0.02
    h = hash(tuple(sorted((str(k), str(v)) for k, v in cfg.items())))
    rng = np.random.default_rng(h & 0xFFFFFFFF)
    jitter = rng.normal(0, noise_amp)
    s = base + jitter
    return s if maximize else -s


def run_experiment_grid(
    experiment_code: str,
    grid: dict,
    n_repeats: int = 3,
    maximize: bool = False,
) -> ToolResult:
    try:
        if not isinstance(grid, dict) or not grid:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="grid must be a non-empty dict of lists."))
        keys = list(grid.keys())
        for k in keys:
            if not isinstance(grid[k], (list, tuple)) or len(grid[k]) == 0:
                return ToolResult(success=False, error=ToolError(type="validation_error", message=f"grid[{k!r}] must be a non-empty list."))
        n_rep = max(1, min(int(n_repeats), 20))
        combos = list(itertools.product(*[grid[k] for k in keys]))
        if len(combos) > 500:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Too many grid combinations (max 500)."))
        results = []
        for combo in combos:
            cfg = {keys[i]: combo[i] for i in range(len(keys))}
            scores = [_score_config(cfg, maximize) for _ in range(n_rep)]
            mean_s, std_s = float(np.mean(scores)), float(np.std(scores, ddof=1)) if n_rep > 1 else 0.0
            results.append({"config": cfg, "mean_score": mean_s, "std_score": std_s, "rank": 0})
        order = sorted(range(len(results)), key=lambda i: results[i]["mean_score"], reverse=maximize)
        for rnk, idx in enumerate(order):
            results[idx]["rank"] = rnk + 1
        best_idx = order[0]
        best = results[best_idx]
        interactions: dict[str, Any] = {}
        if len(keys) >= 2:
            k0, k1 = str(keys[0]), str(keys[1])
            interactions[f"{k0}x{k1}"] = "not estimated in v1 synthetic grid"
        return ToolResult(
            success=True,
            output={
                "results": results,
                "best_config": best["config"],
                "best_score": best["mean_score"],
                "interaction_effects": interactions,
                "total_runs": len(combos) * n_rep,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
