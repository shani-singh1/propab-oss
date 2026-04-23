from __future__ import annotations

from propab.tools.model_registry import get_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "inspect_gradients",
    "domain": "deep_learning",
    "description": "Compute gradient norms per parameter group for one backward step.",
    "params": {"model_id": {"type": "str", "required": True}},
    "output": {"per_layer": "list", "total_norm": "float", "summary": "str"},
    "example": {"params": {"model_id": "x"}, "output": {}},
}


def inspect_gradients(model_id: str) -> ToolResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(success=False, error=ToolError(type="missing_dependency", message="PyTorch required."))

    info = get_model(str(model_id))
    if not info or info.get("kind") not in ("mlp", "mlp_trained"):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Unknown model_id."))
    dims: list[int] = info["dims"]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    if info.get("kind") == "mlp_trained" and info.get("state_dict"):
        net.load_state_dict({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in info["state_dict"].items()})
    x = torch.randn(16, dims[0])
    y = torch.randint(0, dims[-1], (16,))
    loss = torch.nn.functional.cross_entropy(net(x), y)
    loss.backward()
    per = []
    total = 0.0
    for n, p in net.named_parameters():
        if p.grad is not None:
            g = float(p.grad.norm().detach())
            per.append({"name": n, "grad_norm": g})
            total += g * g
    total_norm = float(total**0.5)
    return ToolResult(
        success=True,
        output={"per_layer": per, "total_norm": total_norm, "summary": f"Backward OK, total grad L2≈{total_norm:.4f}."},
    )
