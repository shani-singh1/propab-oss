from __future__ import annotations

from propab.tools.model_registry import get_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "evaluate_model",
    "domain": "deep_learning",
    "description": "Evaluate a trained MLP from train_model on a synthetic batch.",
    "params": {
        "model_id": {"type": "str", "required": True},
        "task": {"type": "str", "required": True, "enum": ["classification", "regression", "autoencoding", "language_modeling"]},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "batch_size": {"type": "int", "required": False, "default": 64},
    },
    "output": {"metrics": "dict", "loss": "float", "summary": "str"},
    "example": {"params": {"model_id": "x", "task": "classification"}, "output": {}},
}


def evaluate_model(model_id: str, task: str, dataset: str = "synthetic", batch_size: int = 64) -> ToolResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(success=False, error=ToolError(type="missing_dependency", message="PyTorch required."))

    info = get_model(str(model_id))
    if not info or info.get("kind") != "mlp_trained":
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Use trained_model_id from train_model output."))
    dims: list[int] = info["dims"]
    sd = info.get("state_dict")
    if not sd:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="Missing weights."))
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    net.load_state_dict({k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in sd.items()})
    net.eval()
    bs = max(8, min(int(batch_size), 512))
    torch.manual_seed(2)
    x = torch.randn(bs, dims[0])
    y = torch.randint(0, dims[-1], (bs,)) if task == "classification" else torch.randn(bs, dims[-1])
    fn = torch.nn.CrossEntropyLoss() if task == "classification" else torch.nn.MSELoss()
    with torch.no_grad():
        out = net(x)
        loss = float(fn(out, y).detach())
    return ToolResult(
        success=True,
        output={"metrics": {"loss": loss}, "loss": loss, "summary": f"Eval loss={loss:.4f} on synthetic batch."},
    )
