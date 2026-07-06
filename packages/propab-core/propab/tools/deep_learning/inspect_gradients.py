from __future__ import annotations

from propab.tools.model_registry import resolve_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "inspect_gradients",
    "domain": "deep_learning",
    "description": (
        "Compute gradient norms per parameter group for one backward step on the "
        "PERSISTED trained weights. Fails closed if the model has no persisted weights "
        "(refuses to report gradients of a random-initialized network)."
    ),
    "params": {"model_id": {"type": "str", "required": True}},
    "output": {"per_layer": "list", "total_norm": "float", "summary": "str"},
    "example": {"params": {"model_id": "x:trained"}, "output": {}},
}


def _act_layer(activation_name: str):
    import torch.nn as nn

    return {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }.get(str(activation_name or "relu").lower(), nn.ReLU())


def inspect_gradients(model_id: str) -> ToolResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(success=False, error=ToolError(type="missing_dependency", message="PyTorch required."))

    info = resolve_model(str(model_id), prefer_trained=True)
    if not info or info.get("kind") not in ("mlp", "mlp_trained"):
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Unknown model_id."))

    # Fail closed when there are no persisted trained weights. Previously this tool
    # rebuilt the net from dims with RANDOM init and reported its gradients as if they
    # were the trained model's — a meaningless (and misleading) metric.
    state_dict = info.get("state_dict")
    if info.get("kind") != "mlp_trained" or not state_dict:
        return ToolResult(
            success=False,
            error=ToolError(
                type="no_trained_weights",
                message=(
                    "Model has no persisted trained weights. Re-run train_model to persist "
                    "the trained state_dict; refusing to report gradients of a "
                    "random-initialized network."
                ),
            ),
        )

    dims: list[int] = info.get("dims") or []
    if not dims or len(dims) < 2:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Model has no stored dims."))

    activation_name = info.get("activation", "relu")
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(_act_layer(activation_name))
    net = nn.Sequential(*layers)
    try:
        loaded = {
            k: (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
            for k, v in state_dict.items()
        }
        net.load_state_dict(loaded)
    except Exception as exc:
        return ToolResult(
            success=False,
            error=ToolError(
                type="weight_load_error",
                message=f"Persisted weights do not match model architecture: {exc}",
            ),
        )

    # Prefer the persisted held-out eval batch so the gradient reflects real data;
    # fall back to a fixed random probe batch only for the forward/backward shape.
    eval_data = info.get("eval_data") or {}
    loss_kind = str(eval_data.get("loss_kind") or "cross_entropy")
    if eval_data.get("x"):
        x = torch.tensor(eval_data["x"], dtype=torch.float32)
        if loss_kind == "cross_entropy":
            y = torch.tensor(eval_data["y"], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(net(x), y)
        elif loss_kind == "bce":
            y = torch.tensor(eval_data["y"], dtype=torch.float32)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x), y)
        else:
            y = torch.tensor(eval_data["y"], dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(net(x), y)
    else:
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
