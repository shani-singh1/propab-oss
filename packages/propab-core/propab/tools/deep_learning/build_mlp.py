from __future__ import annotations

from uuid import uuid4

from propab.tools.model_registry import put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "build_mlp",
    "domain": "deep_learning",
    "description": "Analytical MLP parameter count and architecture summary (v1; trains via train_model).",
    "params": {
        "input_dim": {"type": "int", "required": False, "default": 16,
                      "description": "Input feature dimension. Default 16."},
        "hidden_dims": {"type": "list[int]", "required": False, "default": [64, 32],
                        "description": "Hidden layer sizes. Default [64, 32]."},
        "output_dim": {"type": "int", "required": False, "default": 2,
                       "description": "Output dimension / number of classes. Default 2."},
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
    input_dim: int = 16,
    hidden_dims: list | None = None,
    output_dim: int = 2,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
    # Aliases for common LLM naming variations
    hidden_layers: list | None = None,
    layers: list | None = None,
    num_classes: int | None = None,
    n_classes: int | None = None,
    input_size: int | None = None,
    output_size: int | None = None,
) -> ToolResult:
    # Resolve aliases
    if hidden_dims is None:
        hidden_dims = hidden_layers or layers or [64, 32]
    if num_classes is not None:
        output_dim = int(num_classes)
    if n_classes is not None:
        output_dim = int(n_classes)
    if input_size is not None:
        input_dim = int(input_size)
    if output_size is not None:
        output_dim = int(output_size)
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
