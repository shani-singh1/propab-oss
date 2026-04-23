"""Inject outputs from one tool step into the next (v1: model_id from architecture builders)."""

from __future__ import annotations

from typing import Any

_BUILDERS = frozenset({"build_mlp", "build_transformer"})

# Tools whose params accept ``model_id`` from the registry / builders.
_ACCEPTS_MODEL_ID = frozenset(
    {
        "count_parameters",
        "train_model",
        "evaluate_model",
        "inspect_gradients",
        "profile_model",
        "compute_flops",
        "activation_statistics",
        "loss_landscape",
        "lr_range_test",
        "gradient_noise_scale",
        "compare_optimizers",
        "hessian_analysis",
    }
)


def refine_next_tool_step(
    first_tool: str,
    first_output: dict[str, Any] | None,
    next_step: dict[str, Any],
) -> dict[str, Any]:
    """
    If the first tool was an architecture builder and returned ``model_id``,
    merge that id into the next tool step when the next tool expects it.
    """
    if first_tool not in _BUILDERS or not isinstance(first_output, dict):
        return next_step
    mid = first_output.get("model_id")
    if not mid or not isinstance(mid, str):
        return next_step
    if next_step.get("type") != "tool":
        return next_step
    name = str(next_step.get("tool", ""))
    if name not in _ACCEPTS_MODEL_ID:
        return next_step
    params = next_step.get("params")
    if not isinstance(params, dict):
        params = {}
    else:
        params = dict(params)
    params["model_id"] = mid
    return {"type": "tool", "tool": name, "params": params}
