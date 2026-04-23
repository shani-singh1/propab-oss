from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "hessian_analysis",
    "domain": "algorithm_optimization",
    "description": "Synthetic Hessian eigen-analysis at a point (numpy v1; keyed by model_id).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "n_samples": {"type": "int", "required": False, "default": 128},
        "top_k_eigenvalues": {"type": "int", "required": False, "default": 10},
    },
    "output": {
        "top_eigenvalues": "list[float]",
        "condition_number": "float",
        "trace": "float",
        "negative_eigenvalues": "int",
        "critical_point_type": "str",
        "sharpness": "float",
    },
    "example": {
        "params": {"model_id": "m-1", "n_samples": 64, "top_k_eigenvalues": 5},
        "output": {},
    },
}


def hessian_analysis(
    model_id: str,
    dataset: str = "synthetic",
    n_samples: int = 128,
    top_k_eigenvalues: int = 10,
) -> ToolResult:
    try:
        seed = int(hashlib.sha256(f"{model_id}:{dataset}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        ns = max(8, int(n_samples))
        dim = int(np.clip(ns // 16, 4, 32))
        h = rng.standard_normal((dim, dim))
        h = (h + h.T) / 2.0
        w = np.linalg.eigvalsh(h)
        k = max(1, min(int(top_k_eigenvalues), dim))
        top = sorted(w.tolist(), key=lambda x: abs(x), reverse=True)[:k]
        w_min, w_max = float(w.min()), float(w.max())
        cond = float(w_max / (abs(w_min) + 1e-12)) if dim else 1.0
        trace = float(np.sum(w))
        neg_ct = int(np.sum(w < -1e-8))
        pos_ct = int(np.sum(w > 1e-8))
        if neg_ct == 0:
            cpt = "local_minimum"
        elif pos_ct > 0 and neg_ct > 0:
            cpt = "saddle_point"
        else:
            cpt = "local_maximum"
        sharp = float(max(w.max(), -w.min()))
        return ToolResult(
            success=True,
            output={
                "top_eigenvalues": [float(x) for x in top],
                "condition_number": cond,
                "trace": trace,
                "negative_eigenvalues": neg_ct,
                "critical_point_type": cpt,
                "sharpness": sharp,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
