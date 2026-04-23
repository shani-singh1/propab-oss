from __future__ import annotations

import numpy as np

from propab.tools.model_registry import get_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compute_flops",
    "domain": "ml_research",
    "description": "Theoretical forward-pass FLOPs from registry metadata (MLP / transformer v1 estimates).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "input_shape": {"type": "list[int]", "required": True},
        "unit": {"type": "str", "required": False, "default": "GFLOPs"},
    },
    "output": {
        "total_flops": "float",
        "by_operation_type": "dict",
        "by_layer": "list",
        "compute_intensity": "float",
    },
    "example": {"params": {"model_id": "…", "input_shape": [32, 784]}, "output": {}},
}

_UNITS = {"flops": 1.0, "mflops": 1e6, "gflops": 1e9, "tflops": 1e12}


def compute_flops(model_id: str, input_shape: list, unit: str = "GFLOPs") -> ToolResult:
    try:
        info = get_model(str(model_id))
        if not info:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Unknown model_id — build_mlp / build_transformer first in-process."))
        shape = [int(x) for x in input_shape]
        if not shape:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="input_shape required."))
        ukey = str(unit).replace(" ", "").lower()
        div = _UNITS.get(ukey, 1e9)
        kind = str(info.get("kind", ""))
        by_op: dict[str, float] = {"matmul": 0.0, "conv": 0.0, "activation": 0.0, "norm": 0.0, "other": 0.0}
        by_layer: list[dict] = []
        total = 0.0
        if kind == "mlp":
            dims: list[int] = info["dims"]
            batch = shape[0] if len(shape) >= 1 else 1
            flops = 0.0
            for i in range(len(dims) - 1):
                f = 2.0 * batch * dims[i] * dims[i + 1]
                flops += f
                by_layer.append({"layer_name": f"linear_{i}", "flops": f, "pct_of_total": 0.0})
            act = batch * sum(dims[1:-1]) * 0.1
            by_op["matmul"] = flops
            by_op["activation"] = act
            total = flops + act
        elif kind == "transformer":
            batch = shape[0] if len(shape) >= 1 else 1
            seq = shape[1] if len(shape) >= 2 else int(info.get("max_seq_len", 128))
            hint = float(info.get("flops_hint", 1.0))
            total = hint * batch * max(seq, 1) / max(info.get("max_seq_len", 512), 1)
            by_op["matmul"] = total * 0.85
            by_op["activation"] = total * 0.1
            by_op["norm"] = total * 0.05
            by_layer.append({"layer_name": "transformer_stack", "flops": total, "pct_of_total": 100.0})
        else:
            pc = int(info.get("param_count", 0))
            total = float(2 * pc * max(shape[0], 1))
            by_op["matmul"] = total
            by_layer.append({"layer_name": "unknown", "flops": total, "pct_of_total": 100.0})
        for row in by_layer:
            row["pct_of_total"] = round(100.0 * row["flops"] / (total + 1e-12), 2)
        bytes_est = float(info.get("param_count", 0)) * 4.0 * max(shape[0], 1)
        intensity = total / (bytes_est + 1e-6)
        out_unit = total / div
        return ToolResult(
            success=True,
            output={
                "total_flops": round(out_unit, 6),
                "by_operation_type": {k: round(v / div, 6) for k, v in by_op.items()},
                "by_layer": [{**row, "flops": round(row["flops"] / div, 6)} for row in by_layer],
                "compute_intensity": round(intensity, 6),
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
