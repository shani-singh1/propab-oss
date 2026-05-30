from __future__ import annotations

from propab.tools.model_registry import resolve_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "count_parameters",
    "domain": "deep_learning",
    "description": "Return parameter counts for a model_id from build_mlp / build_transformer.",
    "params": {"model_id": {"type": "str", "required": True}},
    "output": {
        "total_params": "int",
        "trainable_params": "int",
        "frozen_params": "int",
        "by_layer": "list",
        "size_mb": "float",
    },
    "example": {"params": {"model_id": "dummy"}, "output": {}},
}


def count_parameters(model_id: str) -> ToolResult:
    info = resolve_model(str(model_id))
    if not info:
        return ToolResult(
            success=False,
            error=ToolError(
                type="unknown_model",
                message=f"Unknown model_id '{model_id}' — run build_mlp first, or pass a model_id/trained_model_id from a prior tool.",
            ),
        )
    pc = int(info.get("param_count", 0))
    return ToolResult(
        success=True,
        output={
            "total_params": pc,
            "trainable_params": pc,
            "frozen_params": 0,
            "by_layer": [],
            "size_mb": round(pc * 4 / (1024 * 1024), 4),
        },
    )
