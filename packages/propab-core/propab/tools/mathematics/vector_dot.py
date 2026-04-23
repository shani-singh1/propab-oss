from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "vector_dot",
    "domain": "mathematics",
    "description": "Compute the dot product of two equal-length numeric vectors.",
    "params": {
        "a": {"type": "list[float]", "required": True, "description": "First vector"},
        "b": {"type": "list[float]", "required": True, "description": "Second vector"},
    },
    "output": {
        "dot": "float — inner product",
        "length": "int — common dimension",
    },
    "example": {
        "params": {"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 1.0]},
        "output": {"dot": 5.0, "length": 3},
    },
}


def vector_dot(a: list, b: list) -> ToolResult:
    try:
        if len(a) != len(b):
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Vectors must have the same length."),
            )
        if not a:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Vectors must be non-empty."),
            )
        af = [float(x) for x in a]
        bf = [float(x) for x in b]
        dot = sum(x * y for x, y in zip(af, bf))
        return ToolResult(success=True, output={"dot": dot, "length": len(af)})
    except (TypeError, ValueError) as exc:
        return ToolResult(success=False, error=ToolError(type="validation_error", message=str(exc)))
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
