from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "gradient_noise_scale",
    "domain": "algorithm_optimization",
    "description": "Synthetic gradient SNR vs batch size proxy (numpy v1; model_id seeds RNG).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "batch_sizes": {"type": "list[int]", "required": False, "default": [16, 32, 64, 128]},
        "n_batches": {"type": "int", "required": False, "default": 20},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {
        "noise_scale": "float",
        "optimal_batch_size": "int",
        "by_batch_size": "list",
        "summary": "str",
    },
    "example": {
        "params": {"model_id": "m", "batch_sizes": [8, 16, 32]},
        "output": {},
    },
}


def gradient_noise_scale(
    model_id: str,
    batch_sizes: list | None = None,
    n_batches: int = 20,
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        seed = int(hashlib.sha256(str(model_id).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        bs_list = [int(x) for x in (batch_sizes or [16, 32, 64, 128])]
        if not bs_list:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="batch_sizes required."))
        rows = []
        snrs = []
        for bs in bs_list:
            snr = float(2.0 * np.log1p(bs) + rng.normal(0, 0.05))
            snrs.append(snr)
            rows.append(
                {
                    "batch_size": bs,
                    "gradient_snr": round(snr, 4),
                    "effective_lr_range": f"{1e-4 * np.sqrt(bs):.2e}–{1e-3 * np.sqrt(bs):.2e}",
                }
            )
        best_bs = int(bs_list[int(np.argmax(snrs))])
        noise_scale = float(1.0 / (max(snrs) + 1e-6))
        return ToolResult(
            success=True,
            output={
                "noise_scale": round(noise_scale, 6),
                "optimal_batch_size": best_bs,
                "by_batch_size": rows,
                "summary": f"Synthetic GNS proxy (dataset={dataset}, n_batches={n_batches}); prefer batch≈{best_bs}.",
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
