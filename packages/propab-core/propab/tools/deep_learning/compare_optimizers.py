from __future__ import annotations

import time

from propab.tools.model_registry import get_model, put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compare_optimizers",
    "domain": "deep_learning",
    "description": (
        "Train identical MLPs with different optimizers and compare convergence. "
        "model_id is optional — if omitted a default MLP is built automatically. "
        "Tracks loss curves and picks the winner by lowest final loss."
    ),
    "params": {
        "optimizers": {
            "type": "list[str]",
            "required": True,
            "description": "List of optimizer names: adam, sgd, adamw, rmsprop, adagrad.",
        },
        "model_id": {
            "type": "str",
            "required": False,
            "default": "auto",
            "description": "model_id from build_mlp. If omitted, a default MLP [16,64,32,2] is used.",
        },
        "learning_rates": {
            "type": "list[float]",
            "required": False,
            "description": "Per-optimizer LRs. Defaults to 1e-3 for all.",
        },
        "learning_rate": {
            "type": "float",
            "required": False,
            "description": "Single LR applied to all optimizers (alias).",
        },
        "n_steps": {
            "type": "int",
            "required": False,
            "default": 200,
            "description": "Training steps per optimizer. Use >=200 for meaningful convergence comparison.",
        },
        "task": {
            "type": "str",
            "required": False,
            "default": "classification",
        },
        "dataset": {
            "type": "str",
            "required": False,
            "default": "synthetic",
        },
    },
    "output": {"comparison": "list", "winner": "str", "summary": "str", "val_losses": "list"},
    "example": {
        "params": {"optimizers": ["adam", "sgd", "adamw"], "n_steps": 200},
        "output": {},
    },
}

_OPTIMIZER_MAP = {
    "adam": "Adam",
    "sgd": "SGD",
    "adamw": "AdamW",
    "rmsprop": "RMSprop",
    "adagrad": "Adagrad",
    "adadelta": "Adadelta",
    "nadam": "NAdam",
}


def _build_default_mlp() -> dict:
    """Auto-build a small MLP config when no model_id is provided."""
    from uuid import uuid4
    dims = [16, 64, 32, 2]
    model_id = str(uuid4())
    info = {"kind": "mlp", "dims": dims, "activation": "relu", "param_count": sum(dims[i] * dims[i+1] + dims[i+1] for i in range(len(dims)-1))}
    put_model(model_id, info)
    return model_id, info


def _run_opt(name: str, lr: float, n_steps: int, dims: list[int], task: str) -> tuple[list[float], list[float], float]:
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    # Build model
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)

    # Select optimizer
    opt_name = name.lower()
    opt_cls_name = _OPTIMIZER_MAP.get(opt_name)
    if opt_cls_name is None:
        opt_cls_name = "Adam"
    opt_cls = getattr(torch.optim, opt_cls_name)
    opt = opt_cls(net.parameters(), lr=lr)

    # Learnable synthetic data
    n_train, input_dim = 64, dims[0]
    X_train = torch.randn(n_train, input_dim)
    W_true = torch.randn(input_dim) * 0.5
    Y_raw = X_train @ W_true + torch.randn(n_train) * 0.3

    n_classes = dims[-1]
    if task == "classification":
        if n_classes == 1:
            Y_train = (Y_raw > 0).float().unsqueeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
            is_class = True
        else:
            n_classes = max(2, n_classes)
            thresholds = torch.quantile(Y_raw, torch.linspace(0, 1, n_classes + 1)[1:-1])
            Y_train = torch.zeros(n_train, dtype=torch.long)
            for i, t in enumerate(thresholds):
                Y_train[Y_raw > t] = i + 1
            loss_fn = nn.CrossEntropyLoss()
            is_class = True
    else:
        Y_train = Y_raw.unsqueeze(1).expand(-1, n_classes) if n_classes > 1 else Y_raw.unsqueeze(1)
        loss_fn = nn.MSELoss()
        is_class = False

    curve: list[float] = []
    val_losses: list[float] = []
    # Use every 10th step as "val" (different batch)
    X_val = torch.randn(32, input_dim)
    Y_raw_val = X_val @ W_true + torch.randn(32) * 0.3
    if task == "classification":
        if dims[-1] == 1:
            Y_val = (Y_raw_val > 0).float().unsqueeze(1)
        else:
            Y_val = torch.zeros(32, dtype=torch.long)
            for i, t in enumerate(thresholds):
                Y_val[Y_raw_val > t] = i + 1
    else:
        Y_val = Y_raw_val.unsqueeze(1).expand(-1, n_classes) if n_classes > 1 else Y_raw_val.unsqueeze(1)

    for step in range(n_steps):
        opt.zero_grad()
        out = net(X_train)
        loss = loss_fn(out, Y_train)
        if torch.isnan(loss) or torch.isinf(loss):
            curve.append(float("inf"))
            val_losses.append(float("inf"))
            return curve, val_losses, float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        curve.append(float(loss.detach()))
        if step % max(1, n_steps // 20) == 0:
            with torch.no_grad():
                vout = net(X_val)
                vloss = float(loss_fn(vout, Y_val).detach())
                if torch.isnan(torch.tensor(vloss)) or torch.isinf(torch.tensor(vloss)):
                    vloss = float("inf")
                val_losses.append(vloss)

    return curve, val_losses, curve[-1]


def compare_optimizers(
    optimizers: list,
    model_id: str = "auto",
    learning_rates: list | None = None,
    learning_rate: float | None = None,
    n_steps: int = 200,
    task: str = "classification",
    dataset: str = "synthetic",
) -> ToolResult:
    try:
        import torch  # noqa: F401
    except ImportError:
        return ToolResult(success=False, error=ToolError(type="missing_dependency", message="PyTorch required."))

    # Resolve model architecture
    if not model_id or model_id in ("auto", "x", "", "none", "None"):
        model_id, info = _build_default_mlp()
    else:
        info = get_model(str(model_id))
        if not info or info.get("kind") != "mlp":
            # Build default rather than failing
            model_id, info = _build_default_mlp()

    dims: list[int] = info["dims"]
    n_steps = max(50, min(int(n_steps), 1000))

    # Resolve learning rates
    if learning_rates is None or not learning_rates:
        base_lr = float(learning_rate) if learning_rate is not None else 1e-3
        lrs = [base_lr] * len(optimizers)
    else:
        lrs = [float(lr) for lr in learning_rates]
    while len(lrs) < len(optimizers):
        lrs.append(1e-3)

    comparison = []
    all_final_losses: list[float] = []
    for i, name in enumerate(optimizers):
        t0 = time.perf_counter()
        try:
            curve, val_losses, final = _run_opt(str(name), float(lrs[i]), n_steps, dims, task)
            cstep = max(1, len(curve) // 20)
            comparison.append(
                {
                    "optimizer": str(name),
                    "lr": float(lrs[i]),
                    "loss_curve": curve[::cstep],
                    "val_losses": val_losses,
                    "final_loss": final,
                    "wall_sec": round(time.perf_counter() - t0, 4),
                }
            )
            all_final_losses.append(final)
        except Exception as exc:
            comparison.append(
                {
                    "optimizer": str(name),
                    "lr": float(lrs[i]),
                    "error": str(exc),
                    "final_loss": float("inf"),
                }
            )
            all_final_losses.append(float("inf"))

    winner = min(comparison, key=lambda c: c["final_loss"])["optimizer"]
    summary = (
        f"Winner: {winner} (lowest final loss) among {len(comparison)} optimizers "
        f"after {n_steps} steps on {dims} MLP."
    )
    # Expose flat val_losses list for statistical_significance chaining
    flat_val = comparison[0].get("val_losses", []) if comparison else []

    return ToolResult(
        success=True,
        output={
            "comparison": comparison,
            "winner": winner,
            "summary": summary,
            "val_losses": flat_val,
            "final_losses": all_final_losses,
        },
    )
