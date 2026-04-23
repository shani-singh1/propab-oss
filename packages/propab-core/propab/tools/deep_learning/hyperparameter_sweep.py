from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "hyperparameter_sweep",
    "domain": "deep_learning",
    "description": "Grid or random search over a numeric hyperparameter space (synthetic scores, numpy v1).",
    "params": {
        "model_type": {"type": "str", "required": True},
        "search_space": {"type": "dict", "required": True},
        "search_type": {"type": "str", "required": False, "default": "random"},
        "n_trials": {"type": "int", "required": False, "default": 10},
        "n_steps": {"type": "int", "required": False, "default": 200},
        "metric": {"type": "str", "required": False, "default": "val_loss"},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "trials": "list",
        "best_config": "dict",
        "best_score": "float",
        "param_importance": "dict",
    },
    "example": {
        "params": {
            "model_type": "mlp",
            "search_space": {"learning_rate": [1e-4, 1e-3], "dropout": [0.0, 0.1]},
            "search_type": "grid",
            "n_trials": 4,
        },
        "output": {},
    },
}


def _normalize_space(raw: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for k, v in raw.items():
        key = str(k)
        if isinstance(v, (list, tuple)):
            out[key] = list(v)
        else:
            out[key] = [v]
        if not out[key]:
            raise ValueError(f"search_space['{key}'] is empty")
    return out


def _score(config: dict[str, Any], model_type: str, dataset: str, metric: str) -> float:
    blob = str(sorted(config.items())) + model_type + dataset + metric
    seed = int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    lr = float(config.get("learning_rate", config.get("lr", 3e-4)))
    dr = float(config.get("dropout", 0.1))
    base = 1.0 + dr * 0.5 - 0.15 * np.log10(max(lr, 1e-9) + 1e-12)
    noise = float(rng.normal(0, 0.04))
    return float(max(0.05, base + noise))


def _param_importance(trials: list[dict]) -> dict[str, float]:
    if len(trials) < 2:
        return {}
    keys = set()
    for t in trials:
        keys.update(t["config"].keys())
    imp: dict[str, float] = {}
    scores = np.array([t["score"] for t in trials], dtype=float)
    for k in sorted(keys):
        vals: list[float] = []
        for t in trials:
            v = t["config"].get(k)
            if v is None:
                vals.append(float("nan"))
            elif isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
            else:
                vals.append(float(hash(str(v)) % 1000) / 1000.0)
        x = np.array(vals, dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 2 or np.std(x[mask]) < 1e-12:
            imp[k] = 0.0
            continue
        c = float(np.corrcoef(x[mask], scores[mask])[0, 1])
        imp[k] = float(np.nan_to_num(c, nan=0.0))
    return imp


def hyperparameter_sweep(
    model_type: str,
    search_space: dict[str, Any],
    search_type: str = "random",
    n_trials: int = 10,
    n_steps: int = 200,
    metric: str = "val_loss",
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        mt = str(model_type).lower()
        if mt not in ("mlp", "transformer"):
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="model_type must be 'mlp' or 'transformer'."),
            )
        st = str(search_type).lower()
        if st not in ("grid", "random"):
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="search_type must be 'grid' or 'random'."),
            )
        space = _normalize_space(search_space)
        keys = list(space.keys())
        trials_raw: list[dict[str, Any]] = []
        if st == "grid":
            values_product = list(itertools.product(*[space[k] for k in keys]))
            for combo in values_product:
                trials_raw.append(dict(zip(keys, combo, strict=True)))
        else:
            nt = max(1, min(int(n_trials), 500))
            blob = str(sorted(space.items())) + mt
            seed = int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)
            rnd = random.Random(seed)
            for _ in range(nt):
                trials_raw.append({k: rnd.choice(space[k]) for k in keys})
        if not trials_raw:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="No trials generated."))
        trials: list[dict[str, Any]] = []
        for cfg in trials_raw:
            sc = _score(cfg, mt, dataset, metric)
            trials.append({"config": cfg, "score": sc})
        trials_sorted = sorted(trials, key=lambda t: t["score"])
        for i, t in enumerate(trials_sorted):
            t["rank"] = i + 1
        best = trials_sorted[0]
        importance = _param_importance(trials_sorted)
        _ = int(n_steps)
        return ToolResult(
            success=True,
            output={
                "trials": trials_sorted,
                "best_config": dict(best["config"]),
                "best_score": float(best["score"]),
                "param_importance": importance,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
