from __future__ import annotations

from propab.tools.model_registry import resolve_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "profile_model",
    "domain": "deep_learning",
    "description": "Approximate parameter memory and FLOPs for a registered model.",
    "params": {"model_id": {"type": "str", "required": True}},
    "output": {"param_count": "int", "approx_memory_mb": "float", "flops_hint": "int", "summary": "str"},
    "example": {"params": {"model_id": "x"}, "output": {}},
}


def profile_model(model_id: str) -> ToolResult:
    info = resolve_model(str(model_id))
    if not info:
        return ToolResult(
            success=False,
            error=ToolError(
                type="unknown_model",
                message=f"Unknown model_id '{model_id}'. Run build_mlp first, or pass a model_id/trained_model_id from a prior tool.",
            ),
        )
    pc = int(info.get("param_count", 0))
    mem = pc * 4 / (1024 * 1024)
    fh = int(info.get("flops_hint", pc * 2))
    return ToolResult(
        success=True,
        output={
            "param_count": pc,
            "approx_memory_mb": round(mem, 4),
            "flops_hint": fh,
            "summary": f"~{mem:.2f} MB params (fp32), FLOP hint {fh}.",
        },
    )
