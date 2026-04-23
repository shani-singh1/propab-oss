from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "regularization_effect",
    "domain": "algorithm_optimization",
    "description": "Synthetic comparison of regularization strategies on train/val gap (v1 numpy proxy).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "strategies": {"type": "list[str]", "required": True},
        "n_steps": {"type": "int", "required": False, "default": 500},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "lambda_values": {"type": "list[float]", "required": False},
    },
    "output": {
        "comparison": "list",
        "best_strategy": "str",
        "overfit_baseline": "float",
        "summary": "str",
    },
    "example": {
        "params": {
            "model_id": "m1",
            "strategies": ["none", "l2", "dropout"],
            "n_steps": 100,
        },
        "output": {},
    },
}


def regularization_effect(
    model_id: str,
    strategies: list,
    n_steps: int = 500,
    dataset: str = "synthetic",
    lambda_values: list | None = None,
) -> ToolResult:
    try:
        seed = int(hashlib.sha256(str(model_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        lambdas = list(lambda_values) if lambda_values else [0.0, 1e-4, 1e-3]
        rows = []
        for strat in strategies:
            s = str(strat).lower()
            base_train = 1.0 + rng.random() * 0.2
            gap = 0.05 + rng.random() * 0.15
            if s == "none":
                gap += 0.08
            elif s in ("l2", "l1", "weight_decay"):
                lam = float(lambdas[0]) if lambdas else 1e-4
                gap *= max(0.3, 1.0 / (1.0 + lam * 500))
            elif s in ("dropout", "batch_norm", "layer_norm"):
                gap *= 0.75
            elif s == "early_stopping":
                gap *= 0.65
            elif s in ("data_augment",):
                gap *= 0.7
            train_loss = base_train
            val_loss = train_loss + gap
            wnorm = float(1.0 + rng.random())
            if s in ("l2", "weight_decay"):
                wnorm *= 0.85
            rows.append(
                {
                    "strategy": s,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "gen_gap": round(val_loss - train_loss, 4),
                    "weight_norm": round(wnorm, 4),
                    "best_lambda": float(lambdas[0]) if lambdas and s in ("l2", "l1", "weight_decay") else None,
                }
            )
        if not rows:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="strategies required."))
        baseline = next((r["gen_gap"] for r in rows if r["strategy"] == "none"), rows[0]["gen_gap"])
        best = min(rows, key=lambda r: r["gen_gap"])
        return ToolResult(
            success=True,
            output={
                "comparison": rows,
                "best_strategy": best["strategy"],
                "overfit_baseline": round(float(baseline), 4),
                "summary": f"Best synthetic generalization gap: {best['strategy']} (dataset={dataset}, n_steps={n_steps}).",
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
