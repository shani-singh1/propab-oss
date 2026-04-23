from __future__ import annotations

from uuid import uuid4

from propab.tools.model_registry import put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "build_mlp",
    "domain": "deep_learning",
    "description": "Analytical MLP parameter count and architecture summary (v1; trains via train_model).",
    "params": {
        "input_dim": {"type": "int", "required": True},
        "hidden_dims": {"type": "list[int]", "required": True},
        "output_dim": {"type": "int", "required": True},
        "activation": {
            "type": "str",
            "required": False,
            "default": "relu",
            "enum": ["relu", "gelu", "tanh", "sigmoid", "silu"],
        },
        "dropout": {"type": "float", "required": False, "default": 0.0},
        "batch_norm": {"type": "bool", "required": False, "default": False},
    },
    "output": {
        "param_count": "int",
        "layer_shapes": "list",
        "architecture_str": "str",
        "model_id": "str",
    },
    "example": {
        "params": {"input_dim": 8, "hidden_dims": [16, 8], "output_dim": 2},
        "output": {},
    },
}


def build_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
) -> ToolResult:
    try:
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        if any(d <= 0 for d in dims):
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Dimensions must be positive."))
        pc = 0
        layer_shapes: list[dict] = []
        for i in range(len(dims) - 1):
            pc += dims[i] * dims[i + 1] + dims[i + 1]
            layer_shapes.append({"in": dims[i], "out": dims[i + 1]})
            if batch_norm:
                pc += 2 * dims[i + 1]
        arch = f"MLP({'->'.join(str(d) for d in dims)}, act={activation}, dropout={dropout}, bn={batch_norm})"
        mid = str(uuid4())
        put_model(
            mid,
            {
                "kind": "mlp",
                "param_count": pc,
                "dims": dims,
                "activation": activation,
            },
        )
        return ToolResult(
            success=True,
            output={
                "param_count": pc,
                "layer_shapes": layer_shapes,
                "architecture_str": arch,
                "model_id": mid,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
