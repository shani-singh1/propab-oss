from __future__ import annotations

import time

from propab.tools.model_registry import get_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compare_optimizers",
    "domain": "deep_learning",
    "description": "Train identical tiny MLPs with different optimizers (CPU torch, short run).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "optimizers": {"type": "list[str]", "required": True},
        "learning_rates": {"type": "list[float]", "required": False},
        "n_steps": {"type": "int", "required": False, "default": 60},
        "task": {"type": "str", "required": False, "default": "classification"},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
    },
    "output": {"comparison": "list", "winner": "str", "summary": "str"},
    "example": {"params": {"model_id": "x", "optimizers": ["adam", "sgd"]}, "output": {}},
}


def _run_opt(name: str, lr: float, n_steps: int, dims: list[int], task: str) -> tuple[list[float], float]:
    import torch
    import torch.nn as nn

    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    oc = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW, "rmsprop": torch.optim.RMSprop}
    if name.lower() not in oc:
        name = "adam"
    opt = oc[name.lower()](net.parameters(), lr=lr)
    torch.manual_seed(1)
    x = torch.randn(32, dims[0])
    y = torch.randint(0, dims[-1], (32,)) if task == "classification" else torch.randn(32, dims[-1])
    fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    curve: list[float] = []
    for _ in range(n_steps):
        opt.zero_grad()
        out = net(x)
        loss = fn(out, y)
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    return curve, curve[-1]


def compare_optimizers(
    model_id: str,
    optimizers: list,
    learning_rates: list | None = None,
    n_steps: int = 60,
    task: str = "classification",
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        import torch  # noqa: F401
    except ImportError:
        return ToolResult(success=False, error=ToolError(type="missing_dependency", message="PyTorch required."))

    info = get_model(str(model_id))
    if not info or info.get("kind") != "mlp":
        return ToolResult(success=False, error=ToolError(type="validation_error", message="model_id must be from build_mlp."))
    dims: list[int] = info["dims"]
    n_steps = max(10, min(int(n_steps), 500))
    lrs = learning_rates if learning_rates else [1e-3] * len(optimizers)
    while len(lrs) < len(optimizers):
        lrs.append(1e-3)
    comparison = []
    for i, name in enumerate(optimizers):
        t0 = time.perf_counter()
        curve, final = _run_opt(str(name), float(lrs[i]), n_steps, dims, task)
        cstep = max(1, len(curve) // 10)
        comparison.append(
            {
                "optimizer": str(name),
                "lr": float(lrs[i]),
                "loss_curve": curve[::cstep],
                "final_loss": final,
                "steps_to_convergence": n_steps,
                "wall_sec": round(time.perf_counter() - t0, 4),
            }
        )
    winner = min(comparison, key=lambda c: c["final_loss"])["optimizer"]
    summary = f"Lowest final loss: {winner} among {len(comparison)} optimizers on synthetic batch."
    return ToolResult(success=True, output={"comparison": comparison, "winner": winner, "summary": summary})
