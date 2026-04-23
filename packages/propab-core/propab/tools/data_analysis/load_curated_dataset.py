from __future__ import annotations

import csv
import io
from typing import Any

import httpx

from propab.datasets import describe_dataset, synthetic_gaussian_rows
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "load_curated_dataset",
    "domain": "data_analysis",
    "description": "Load a small preview of a curated public dataset (CSV over HTTPS or built-in synthetic).",
    "params": {
        "dataset_id": {"type": "str", "required": True},
        "max_rows": {"type": "int", "required": False, "default": 32},
    },
    "output": {
        "columns": "list[str]",
        "rows": "list[dict]",
        "n_rows": "int",
        "source": "str",
    },
    "example": {"params": {"dataset_id": "synthetic_gaussian", "max_rows": 12}, "output": {}},
}


def load_curated_dataset(dataset_id: str, max_rows: int = 32) -> ToolResult:
    try:
        meta = describe_dataset(dataset_id)
        if meta is None:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message=f"Unknown dataset_id: {dataset_id!r}."),
            )
        mr = max(5, min(int(max_rows), 500))
        if meta.dataset_id == "synthetic_gaussian":
            cols, rows = synthetic_gaussian_rows(n_rows=mr, seed=42)
            return ToolResult(
                success=True,
                output={"columns": cols, "rows": rows[:mr], "n_rows": len(rows[:mr]), "source": meta.source},
            )
        if not meta.url:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="Dataset has no URL."))
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(meta.url)
            r.raise_for_status()
        text = r.text
        reader = csv.DictReader(io.StringIO(text))
        cols = reader.fieldnames or []
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(reader):
            if i >= mr:
                break
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return ToolResult(
            success=True,
            output={
                "columns": list(cols),
                "rows": rows,
                "n_rows": len(rows),
                "source": meta.source,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
