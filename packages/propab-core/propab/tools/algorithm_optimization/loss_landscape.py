from __future__ import annotations

import numpy as np

from propab.tools.types import ToolResult

TOOL_SPEC = {
    "name": "loss_landscape",
    "domain": "algorithm_optimization",
    "description": "Synthetic 2D loss grid around origin (v1 proxy; no model weights).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "resolution": {"type": "int", "required": False, "default": 12},
        "extent": {"type": "float", "required": False, "default": 1.0},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "n_samples": {"type": "int", "required": False, "default": 256},
    },
    "output": {
        "landscape_grid": "list",
        "sharpness": "float",
        "flatness_score": "float",
        "minimum_type": "str",
        "plot_data": "dict",
        "summary": "str",
    },
    "example": {"params": {"model_id": "any"}, "output": {}},
}


def loss_landscape(
    model_id: str,
    resolution: int = 12,
    extent: float = 1.0,
    dataset: str = "synthetic",
    n_samples: int = 256,
) -> ToolResult:
    r = max(4, min(int(resolution), 48))
    xs = np.linspace(-extent, extent, r)
    ys = np.linspace(-extent, extent, r)
    grid = []
    for x in xs:
        row = []
        for y in ys:
            row.append(float((x * x + y * y) / 2 + 0.1 * np.sin(5 * x) * np.sin(5 * y)))
        grid.append(row)
    arr = np.array(grid)
    sharp = float(arr.max() - arr.min())
    flat = float(np.mean(arr < arr.min() + 0.1 * (arr.max() - arr.min())))
    return ToolResult(
        success=True,
        output={
            "landscape_grid": grid,
            "sharpness": round(sharp, 4),
            "flatness_score": round(flat, 4),
            "minimum_type": "flat_minimum" if flat > 0.4 else "sharp_minimum",
            "plot_data": {"x": xs.tolist(), "y": ys.tolist(), "z": grid},
            "summary": "Synthetic bowl+ripple surface for UI contour tests.",
        },
    )
