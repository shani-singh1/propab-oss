from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "lr_range_test",
    "domain": "deep_learning",
    "description": "Synthetic LR sweep vs loss valley (numpy v1; use model_id as RNG seed).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "optimizer": {"type": "str", "required": False, "default": "adam"},
        "lr_min": {"type": "float", "required": False, "default": 1e-7},
        "lr_max": {"type": "float", "required": False, "default": 1.0},
        "n_steps": {"type": "int", "required": False, "default": 100},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "lr_loss_curve": "list",
        "suggested_lr": "float",
        "suggested_lr_max": "float",
    },
    "example": {
        "params": {
            "model_id": "seed-model",
            "optimizer": "adam",
            "lr_min": 1e-5,
            "lr_max": 0.1,
            "n_steps": 24,
        },
        "output": {},
    },
}


def lr_range_test(
    model_id: str,
    optimizer: str = "adam",
    lr_min: float = 1e-7,
    lr_max: float = 1.0,
    n_steps: int = 100,
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        seed = int(hashlib.sha256(str(model_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        lo, hi = float(lr_min), float(lr_max)
        if lo <= 0 or hi <= lo:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Need 0 < lr_min < lr_max."))
        n = max(8, min(int(n_steps), 300))
        lrs = np.logspace(np.log10(lo), np.log10(hi), n)
        curve = []
        for lr in lrs:
            x = float(np.log10(lr))
            loss = float(1.0 + (x + 3.2) ** 2 + rng.normal(0, 0.015))
            curve.append({"lr": float(lr), "loss": loss})
        best = min(curve, key=lambda r: r["loss"])
        suggested = float(best["lr"])
        cstep = max(1, len(curve) // 40)
        return ToolResult(
            success=True,
            output={
                "lr_loss_curve": curve[::cstep],
                "suggested_lr": round(suggested, 8),
                "suggested_lr_max": round(min(suggested * 3.0, hi), 8),
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
