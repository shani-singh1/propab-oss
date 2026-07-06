from __future__ import annotations

from propab.tools.model_registry import resolve_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "evaluate_model",
    "domain": "deep_learning",
    "description": (
        "Evaluate a trained MLP (from train_model) by running REAL evaluation passes "
        "on the persisted trained weights against the held-out eval split. Returns an "
        "eval_losses list with genuine across-pass variance (bootstrap over the held-out "
        "set) for use with statistical_significance. Fails closed if the model was not "
        "trained with persisted weights + eval data. Use trained_model_id from train_model."
    ),
    "params": {
        "model_id": {"type": "str", "required": True},
        "task": {
            "type": "str",
            "required": True,
            "enum": ["classification", "regression", "autoencoding", "language_modeling"],
        },
        "dataset": {"type": "str", "required": False, "default": "synthetic"},
        "batch_size": {"type": "int", "required": False, "default": 64},
        "n_eval_passes": {"type": "int", "required": False, "default": 5},
    },
    "output": {
        "metrics": "dict",
        "loss": "float",
        "eval_losses": "list[float]",
        "summary": "str",
    },
    "example": {"params": {"model_id": "x:trained", "task": "classification"}, "output": {}},
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


def evaluate_model(
    model_id: str,
    task: str,
    dataset: str = "synthetic",
    batch_size: int = 64,
    n_eval_passes: int = 5,
) -> ToolResult:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return ToolResult(
            success=False,
            error=ToolError(type="missing_dependency", message="PyTorch required."),
        )

    info = resolve_model(str(model_id), prefer_trained=True)
    if not info:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message=f"model_id '{model_id}' not found. Use trained_model_id from train_model output.",
            ),
        )

    # ── Fail-closed guards ────────────────────────────────────────────────────
    # Honest evaluation requires the *actual trained weights* and a held-out eval
    # split. If either is missing (e.g. the model was registered by an older
    # train_model that only stored the loss curve, or was never trained), we must
    # NOT fabricate eval_losses — that is exactly what let a DL hypothesis "confirm"
    # on manufactured variance. Return success=False and emit no eval_losses so the
    # statistical_significance tool has nothing fake to consume.
    dims: list[int] = info.get("dims") or []
    state_dict = info.get("state_dict")
    eval_data = info.get("eval_data")

    if info.get("kind") != "mlp_trained" or not state_dict:
        return ToolResult(
            success=False,
            error=ToolError(
                type="no_trained_weights",
                message=(
                    "Model has no persisted trained weights. Re-run train_model to persist "
                    "the trained state_dict; refusing to fabricate eval metrics from a "
                    "random-initialized network."
                ),
            ),
        )
    if not eval_data or not eval_data.get("x") or eval_data.get("y") is None:
        return ToolResult(
            success=False,
            error=ToolError(
                type="no_eval_data",
                message=(
                    "Model has no persisted held-out eval split. Re-run train_model; "
                    "refusing to fabricate eval metrics without real evaluation data."
                ),
            ),
        )
    if not dims or len(dims) < 2:
        return ToolResult(
            success=False,
            error=ToolError(
                type="validation_error",
                message="Model has no stored dims. Rebuild with build_mlp + train_model.",
            ),
        )

    # ── Rebuild the trained network and load the REAL weights ─────────────────
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
    except Exception as exc:  # weights don't match the architecture — fail closed.
        return ToolResult(
            success=False,
            error=ToolError(
                type="weight_load_error",
                message=f"Persisted weights do not match model architecture: {exc}",
            ),
        )
    net.eval()

    # ── Reconstruct the held-out eval tensors ─────────────────────────────────
    loss_kind = str(eval_data.get("loss_kind") or "cross_entropy")
    x_eval = torch.tensor(eval_data["x"], dtype=torch.float32)
    if loss_kind == "cross_entropy":
        y_eval = torch.tensor(eval_data["y"], dtype=torch.long)
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    elif loss_kind == "bce":
        y_eval = torch.tensor(eval_data["y"], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss()
    else:  # mse / regression
        y_eval = torch.tensor(eval_data["y"], dtype=torch.float32)
        loss_fn = nn.MSELoss()

    n_eval = int(x_eval.shape[0])
    if n_eval < 2:
        return ToolResult(
            success=False,
            error=ToolError(
                type="no_eval_data",
                message="Persisted eval split is too small (<2) for a variance estimate.",
            ),
        )

    # ── REAL evaluation passes: bootstrap over the held-out set ───────────────
    # Each pass runs the trained network on an independent bootstrap resample of the
    # held-out eval data. Because the eval loss genuinely varies across which held-out
    # examples are sampled, the resulting eval_losses reflect real evaluation
    # uncertainty — NOT a single stored number with cosmetic ±2% jitter. This yields
    # an honest sampling distribution for statistical_significance to consume.
    n_pass = max(2, min(int(n_eval_passes), 20))
    bs = max(1, min(int(batch_size), n_eval))
    seed = abs(hash(str(model_id) + "eval")) & 0x7FFFFFFF
    gen = torch.Generator().manual_seed(seed)

    eval_losses: list[float] = []
    with torch.no_grad():
        for _ in range(n_pass):
            # Bootstrap resample (with replacement) of the held-out eval set.
            idx = torch.randint(0, n_eval, (bs,), generator=gen)
            xb = x_eval[idx]
            yb = y_eval[idx]
            loss_val = float(loss_fn(net(xb), yb).detach())
            eval_losses.append(round(loss_val, 6))

    loss = round(sum(eval_losses) / len(eval_losses), 6)
    return ToolResult(
        success=True,
        output={
            "metrics": {
                "loss": loss,
                "n_eval_examples": n_eval,
                "n_passes": n_pass,
                "eval_method": "bootstrap_over_heldout",
            },
            "loss": loss,
            "eval_losses": eval_losses,
            "summary": (
                f"Eval loss={loss:.4f} over {n_pass} real bootstrap passes on {n_eval} "
                f"held-out examples (trained weights). eval_losses={eval_losses} — "
                f"use with statistical_significance."
            ),
        },
    )
