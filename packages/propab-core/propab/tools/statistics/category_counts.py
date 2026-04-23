from __future__ import annotations

from collections import Counter

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "category_counts",
    "domain": "statistics",
    "description": "Count occurrences of categorical values (strings or numbers).",
    "params": {
        "values": {
            "type": "list",
            "required": True,
            "description": "List of hashable category labels",
        },
        "top_k": {
            "type": "int",
            "required": False,
            "description": "Return only the k most frequent categories",
        },
    },
    "output": {
        "counts": "dict[str, int] — category → count",
        "distinct": "int — number of unique categories",
        "total": "int — number of input values",
    },
    "example": {
        "params": {"values": ["a", "b", "a", "a"], "top_k": 2},
        "output": {"counts": {"a": 3, "b": 1}, "distinct": 2, "total": 4},
    },
}


def category_counts(values: list, top_k: int | None = None) -> ToolResult:
    try:
        if not isinstance(values, list):
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="values must be a list."),
            )
        c = Counter(values)
        if top_k is not None:
            items = c.most_common(max(1, int(top_k)))
        else:
            items = c.most_common()
        counts = {str(k): int(v) for k, v in items}
        return ToolResult(
            success=True,
            output={
                "counts": counts,
                "distinct": len(c),
                "total": sum(c.values()),
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
