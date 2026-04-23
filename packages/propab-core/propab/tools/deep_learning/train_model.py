from __future__ import annotations

import math

from propab.tools.model_registry import get_model, put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "train_model",
    "domain": "deep_learning",
    "description": "Train a tiny MLP from build_mlp on synthetic data (CPU torch).",
    "params": {
        "model_id": {"type": "str", "required": True},
        "task": {"type": "str", "required": True, "enum": ["classification", "regression", "autoencoding", "language_modeling"]},
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "n_steps": {"type": "int", "required": False, "default": 80},
        "batch_size": {"type": "int", "required": False, "default": 32},
        "optimizer": {"type": "str", "required": False, "default": "adam"},
        "learning_rate": {"type": "float", "required": False, "default": 1e-3},
        "lr_schedule": {"type": "str", "required": False, "default": "none"},
        "weight_decay": {"type": "float", "required": False, "default": 0.0},
        "record_every": {"type": "int", "required": False, "default": 10},
    },
    "output": {
        "loss_curve": "list",
        "gradient_norms": "list",
        "final_train_loss": "float",
        "final_val_loss": "float",
        "final_metric": "float",
        "total_time_sec": "float",
        "steps_per_sec": "float",
        "trained_model_id": "str",
    },
    "example": {"params": {"model_id": "x", "task": "classification"}, "output": {}},
}


def train_model(
    model_id: str,
    task: str,
    dataset: str = "synthetic",
    n_steps: int = 80,
    batch_size: int = 32,
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    lr_schedule: str = "none",
    weight_decay: float = 0.0,
    record_every: int = 10,
) -> ToolResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(
            success=False,
            error=ToolError(type="missing_dependency", message="PyTorch required — install propab[dl] on the worker."),
        )

    info = get_model(str(model_id))
    if not info or info.get("kind") != "mlp":
        return ToolResult(success=False, error=ToolError(type="validation_error", message="model_id must come from build_mlp."))

    dims: list[int] = info["dims"]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    opt_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}.get(optimizer.lower(), torch.optim.Adam)
    opt = opt_cls(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_steps = max(5, min(int(n_steps), 2000))
    bs = max(4, min(int(batch_size), 512))
    torch.manual_seed(0)
    x = torch.randn(bs, dims[0])
    if task == "classification":
        y = torch.randint(0, dims[-1], (bs,))
        loss_fn = nn.CrossEntropyLoss()
    else:
        y = torch.randn(bs, dims[-1])
        loss_fn = nn.MSELoss()

    import time as _t

    t0 = _t.perf_counter()
    curve: list[dict] = []
    grads: list[dict] = []
    last_loss = 0.0
    for step in range(n_steps):
        opt.zero_grad()
        logits = net(x)
        if task == "classification":
            loss = loss_fn(logits, y)
        else:
            loss = loss_fn(logits, y)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0))
        opt.step()
        last_loss = float(loss.detach())
        if step % max(1, int(record_every)) == 0:
            curve.append({"step": step, "train_loss": last_loss, "val_loss": last_loss})
            grads.append({"step": step, "grad_norm": gn})

    dt = _t.perf_counter() - t0
    tid = str(model_id) + ":trained"
    put_model(tid, {"kind": "mlp_trained", "base": model_id, "dims": dims, "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}})
    return ToolResult(
        success=True,
        output={
            "loss_curve": curve,
            "gradient_norms": grads,
            "final_train_loss": last_loss,
            "final_val_loss": last_loss,
            "final_metric": last_loss,
            "total_time_sec": round(dt, 4),
            "steps_per_sec": round(n_steps / max(dt, 1e-6), 2),
            "trained_model_id": tid,
        },
    )
