from __future__ import annotations

import time as _t

from propab.tools.model_registry import get_model, put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "train_model",
    "domain": "deep_learning",
    "description": (
        "Train a registered MLP (from build_mlp) on synthetic data using real PyTorch. "
        "Returns val_losses list for significance testing and final_val_loss."
    ),
    "params": {
        "model_id": {"type": "str", "required": False, "default": "auto",
                     "description": "model_id from build_mlp. Use 'auto' to auto-build a default MLP."},
        "task": {
            "type": "str",
            "required": False,
            "default": "classification",
            "enum": ["classification", "regression", "autoencoding", "language_modeling"],
        },
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "n_steps": {"type": "int", "required": False, "default": 120,
                    "description": "Number of gradient steps (also accepted as 'epochs' or 'num_steps')"},
        "batch_size": {"type": "int", "required": False, "default": 32},
        "optimizer": {"type": "str", "required": False, "default": "adam"},
        "learning_rate": {"type": "float", "required": False, "default": 1e-3},
        "lr_schedule": {"type": "str", "required": False, "default": "none"},
        "weight_decay": {"type": "float", "required": False, "default": 0.0},
        "noise_level": {"type": "float", "required": False, "default": 0.0},
        "record_every": {"type": "int", "required": False, "default": 10},
        "epochs": {"type": "int", "required": False, "default": None,
                   "description": "Alias for n_steps — pass epochs OR n_steps, not both"},
    },
    "output": {
        "loss_curve": "list[dict]",
        "val_losses": "list[float]",
        "gradient_norms": "list[dict]",
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
    model_id: str = "auto",
    task: str = "classification",
    dataset: str = "synthetic",
    n_steps: int = 120,
    batch_size: int = 32,
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    lr_schedule: str = "none",
    weight_decay: float = 0.0,
    noise_level: float = 0.0,
    record_every: int = 10,
    epochs: int | None = None,  # alias for n_steps
    num_steps: int | None = None,  # alias for n_steps
    num_epochs: int | None = None,  # alias for n_steps
    lr: float | None = None,  # alias for learning_rate
) -> ToolResult:
    # Resolve parameter aliases from LLM variations
    if epochs is not None:
        n_steps = int(epochs)
    if num_steps is not None:
        n_steps = int(num_steps)
    if num_epochs is not None:
        n_steps = int(num_epochs)
    if lr is not None:
        learning_rate = float(lr)
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(
            success=False,
            error=ToolError(
                type="missing_dependency",
                message="PyTorch required — install propab[dl] on the worker.",
            ),
        )

    # Auto-build a default MLP if no model_id provided or model not found
    if str(model_id).lower() in ("auto", "", "none", "null") or get_model(str(model_id)) is None:
        from propab.tools.deep_learning.build_mlp import build_mlp as _build
        _br = _build(input_dim=16, hidden_dims=[64, 32], output_dim=2, activation="relu")
        if not _br.success:
            return ToolResult(success=False, error=ToolError(
                type="auto_build_error",
                message=f"Auto-build MLP failed: {_br.error}",
            ))
        model_id = _br.output["model_id"]

    info = get_model(str(model_id))
    if not info or info.get("kind") != "mlp":
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"model_id '{model_id}' not found or not an MLP. Call build_mlp first, or use model_id='auto'.",
            ),
        )

    dims: list[int] = info["dims"]
    activation_name: str = info.get("activation", "relu")

    def _act_layer() -> nn.Module:
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }.get(activation_name.lower(), nn.ReLU())

    # Build the network
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(_act_layer())
    net = nn.Sequential(*layers)

    # Initialize optimizer
    opt_name = str(optimizer).lower()
    opt_map = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    opt_cls = opt_map.get(opt_name, torch.optim.Adam)
    try:
        opt = opt_cls(net.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    except TypeError:
        # SGD doesn't accept weight_decay in all versions
        opt = opt_cls(net.parameters(), lr=float(learning_rate))

    # Generate a deterministic synthetic dataset keyed to model_id
    # Different model_ids → different data, ensuring varied outcomes
    seed = hash(str(model_id)) & 0x7FFFFFFF
    torch.manual_seed(seed)

    n_train, n_val = 300, 80
    n_steps = max(20, min(int(n_steps), 400))
    bs = max(8, min(int(batch_size), 256))
    rec_every = max(1, int(record_every))

    X_train = torch.randn(n_train, dims[0])
    X_val = torch.randn(n_val, dims[0])

    if task in ("classification", "language_modeling"):
        # Linearly separable data so architectures/configs show real performance differences
        W_true = torch.randn(dims[0]) * 0.5
        Y_raw_train = X_train @ W_true + torch.randn(n_train) * max(0.1, float(noise_level))
        Y_raw_val = X_val @ W_true + torch.randn(n_val) * max(0.1, float(noise_level))
        n_classes = dims[-1]
        if n_classes == 1:
            Y_train = (Y_raw_train > 0).float().unsqueeze(1)
            Y_val = (Y_raw_val > 0).float().unsqueeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            n_classes = max(2, n_classes)
            thresholds = torch.quantile(Y_raw_train, torch.linspace(0, 1, n_classes + 1)[1:-1])
            def _quantize(vals, thr):
                labels = torch.zeros(len(vals), dtype=torch.long)
                for i, t in enumerate(thr):
                    labels[vals > t] = i + 1
                return labels
            Y_train = _quantize(Y_raw_train, thresholds)
            Y_val = _quantize(Y_raw_val, thresholds)
            loss_fn = nn.CrossEntropyLoss()
        is_class = True
    else:
        # regression / autoencoding
        W_true = torch.randn(dims[0], dims[-1]) * 0.3
        Y_train = X_train @ W_true + torch.randn(n_train, dims[-1]) * max(0.0, float(noise_level) + 0.1)
        Y_val = X_val @ W_true + torch.randn(n_val, dims[-1]) * max(0.0, float(noise_level) + 0.1)
        loss_fn = nn.MSELoss()
        is_class = False

    # Optional LR warmup / cosine schedule
    scheduler = None
    lr_sched_name = str(lr_schedule).lower()
    if lr_sched_name == "warmup":
        warmup_steps = max(5, n_steps // 10)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
    elif lr_sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    t0 = _t.perf_counter()
    loss_curve: list[dict] = []
    gradient_norms: list[dict] = []
    val_losses: list[float] = []
    last_train_loss = 0.0

    for step in range(n_steps):
        # Random minibatch each step
        idx = torch.randint(0, n_train, (bs,))
        x_b = X_train[idx]
        y_b = Y_train[idx]

        opt.zero_grad()
        logits = net(x_b)
        loss = loss_fn(logits, y_b)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0))
        opt.step()
        if scheduler is not None:
            scheduler.step()

        last_train_loss = float(loss.detach())

        if step % rec_every == 0:
            with torch.no_grad():
                val_logits = net(X_val)
                val_loss = float(loss_fn(val_logits, Y_val).detach())
            val_losses.append(round(val_loss, 6))
            loss_curve.append({
                "step": step,
                "train_loss": round(last_train_loss, 6),
                "val_loss": round(val_loss, 6),
            })
            gradient_norms.append({"step": step, "grad_norm": round(gn, 6)})

    dt = _t.perf_counter() - t0
    final_val = val_losses[-1] if val_losses else round(last_train_loss, 6)
    tid = f"{model_id}:trained"
    put_model(tid, {
        "kind": "mlp_trained",
        "base": model_id,
        "dims": dims,
        "val_losses": val_losses,
        "final_val_loss": final_val,
    })

    return ToolResult(
        success=True,
        output={
            "loss_curve": loss_curve,
            "val_losses": val_losses,
            "gradient_norms": gradient_norms,
            "final_train_loss": round(last_train_loss, 6),
            "final_val_loss": final_val,
            "final_metric": final_val,
            "total_time_sec": round(dt, 4),
            "steps_per_sec": round(n_steps / max(dt, 1e-6), 2),
            "trained_model_id": tid,
        },
    )
