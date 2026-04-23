from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "activation_statistics",
    "domain": "deep_learning",
    "description": "Synthetic per-layer activation stats (v1 numpy; no forward through torch models).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "n_batches": {"type": "int", "required": False, "default": 20},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "by_layer": "list",
        "dead_neuron_pct": "float",
        "saturation_pct": "float",
        "summary": "str",
    },
    "example": {"params": {"model_id": "demo-model", "n_batches": 5}, "output": {}},
}


def activation_statistics(
    model_id: str,
    n_batches: int = 20,
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        seed = int(hashlib.sha256(str(model_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        n_batches = max(1, min(int(n_batches), 200))
        layer_names = ["input", "hidden_1", "hidden_2", "logits"]
        by_layer: list[dict] = []
        all_vals: list[np.ndarray] = []
        for ln in layer_names:
            w = rng.normal(0, 0.5, size=(n_batches, 64))
            act = np.maximum(0, w)
            frac_zero = float(np.mean(act < 1e-6))
            frac_sat = float(np.mean(np.abs(act) > 3.0))
            by_layer.append(
                {
                    "layer_name": ln,
                    "mean": round(float(np.mean(act)), 6),
                    "std": round(float(np.std(act)), 6),
                    "fraction_zero": round(frac_zero, 6),
                    "fraction_saturated": round(frac_sat, 6),
                }
            )
            all_vals.append(act.ravel())
        stacked = np.concatenate(all_vals)
        dead = float(np.mean(stacked < 1e-6))
        sat = float(np.mean(np.abs(stacked) > 3.0))
        return ToolResult(
            success=True,
            output={
                "by_layer": by_layer,
                "dead_neuron_pct": round(100 * dead, 4),
                "saturation_pct": round(100 * sat, 4),
                "summary": f"Synthetic activations for model_id={model_id!r} dataset={dataset} over {n_batches} batches.",
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
