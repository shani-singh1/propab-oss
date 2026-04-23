from __future__ import annotations

import io
from uuid import uuid4

import numpy as np

from propab.storage import put_bytes
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "plot_training_curves",
    "domain": "ml_research",
    "description": "Plot loss curves to PNG and upload to MinIO when configured (matplotlib).",
    "params": {
        "curves": {"type": "list[dict]", "required": True},
        "title": {"type": "str", "required": False, "default": "Training Curves"},
        "ylabel": {"type": "str", "required": False, "default": "Loss"},
        "smoothing": {"type": "float", "required": False, "default": 0.0},
        "log_scale": {"type": "bool", "required": False, "default": False},
    },
    "output": {"figure_id": "str", "figure_url": "str", "plot_data": "dict"},
    "example": {
        "params": {
            "curves": [
                {"label": "train", "steps": [0, 1, 2], "values": [1.0, 0.8, 0.6]},
                {"label": "val", "steps": [0, 1, 2], "values": [1.1, 0.85, 0.7]},
            ]
        },
        "output": {},
    },
}


def _smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or len(y) < 2:
        return y
    out = np.empty_like(y)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


def plot_training_curves(
    curves: list,
    title: str = "Training Curves",
    ylabel: str = "Loss",
    smoothing: float = 0.0,
    log_scale: bool = False,
) -> ToolResult:
    try:
        if not isinstance(curves, list) or not curves:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="curves must be a non-empty list."))
        plot_data: dict = {"series": []}
        for c in curves:
            if not isinstance(c, dict):
                continue
            label = str(c.get("label", "curve"))
            steps = c.get("steps") or []
            vals = c.get("values") or []
            if len(steps) != len(vals) or not steps:
                continue
            y = np.array([float(v) for v in vals], dtype=np.float64)
            y = _smooth(y, float(smoothing))
            plot_data["series"].append({"label": label, "steps": list(steps), "values": y.tolist()})
        if not plot_data["series"]:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="No valid curve entries."))
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return ToolResult(
                success=True,
                output={
                    "figure_id": "",
                    "figure_url": "",
                    "plot_data": plot_data,
                    "note": "matplotlib not installed; plot_data only.",
                },
            )
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in plot_data["series"]:
            ax.plot(s["steps"], s["values"], label=s["label"])
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        data = buf.getvalue()
        oid = f"figures/plots/{uuid4().hex}.png"
        url = put_bytes(object_name=oid, data=data, content_type="image/png")
        return ToolResult(
            success=True,
            output={
                "figure_id": oid,
                "figure_url": url or "",
                "plot_data": plot_data,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
