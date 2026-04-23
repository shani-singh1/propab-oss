from __future__ import annotations

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "json_extract",
    "domain": "general_computation",
    "description": "Extract a key from a flat JSON object.",
    "params": {
        "data": {"type": "dict", "required": True, "description": "Input JSON object"},
        "key": {"type": "str", "required": True, "description": "Key to fetch from the object"},
    },
    "output": {
        "value": "Any - value for key",
        "exists": "bool - whether key exists",
    },
    "example": {
        "params": {"data": {"a": 1, "b": 2}, "key": "a"},
        "output": {"value": 1, "exists": True},
    },
}


def json_extract(data: dict, key: str) -> ToolResult:
    try:
        return ToolResult(success=True, output={"value": data.get(key), "exists": key in data})
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
