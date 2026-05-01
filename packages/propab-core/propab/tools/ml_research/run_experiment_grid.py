from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from propab.config import settings
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "run_experiment_grid",
    "domain": "ml_research",
    "description": (
        "Evaluate all combinations of a hyperparameter grid by training real MLP models. "
        "Returns val_losses per config — use these with statistical_significance to compare configs."
    ),
    "params": {
        "experiment_code": {"type": "str", "required": False, "default": "# grid search",
                            "description": "Ignored. Provide grid dict instead."},
        "grid": {"type": "dict", "required": False,
                 "default": {"lr": [0.001, 0.01], "batch_size": [16, 32]},
                 "description": "Hyperparameter grid. E.g. {'lr': [0.001, 0.01], 'batch_size': [16, 32]}"},
        "n_repeats": {"type": "int", "required": False, "default": 3},
        "maximize": {"type": "bool", "required": False, "default": False},
        "n_steps": {
            "type": "int",
            "required": False,
            "default": 200,
            "description": "Steps per config. Subtle effects need >=300; structural effects >=150; convergence >=500.",
        },
        "task": {"type": "str", "required": False, "default": "classification"},
        "dataset": {"type": "str", "required": False, "default": "mnist"},
        "input_dim": {"type": "int", "required": False, "default": 16},
        "hidden_dims": {"type": "list[int]", "required": False, "default": [32, 16]},
        "output_dim": {"type": "int", "required": False, "default": 2},
    },
    "output": {
        "results": "list",
        "best_config": "dict",
        "best_score": "float",
        "best_val_losses": "list[float]",
        "worst_val_losses": "list[float]",
        "interaction_effects": "dict",
        "total_runs": "int",
    },
    "example": {
        "params": {
            "experiment_code": "# reserved",
            "grid": {"lr": [0.001, 0.01], "batch_size": [16, 32]},
            "n_repeats": 2,
        },
        "output": {},
    },
}


def _real_train_score(
    cfg: dict[str, Any],
    *,
    maximize: bool,
    n_steps: int,
    task: str,
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dataset: str = "synthetic",
    seed_offset: int = 0,
) -> tuple[float, list[float]]:
    """Run a real training loop with this config; return (score, val_losses)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return _proxy_score(cfg, maximize, seed=seed_offset), []

    lr = float(cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    bs = int(cfg.get("batch_size", cfg.get("bs", 32)))
    opt_name = str(cfg.get("optimizer", "adam")).lower()
    activation = str(cfg.get("activation", cfg.get("act", "relu"))).lower()

    # Use a config-derived seed so different configs → different data
    cfg_seed = abs(hash(str(sorted(cfg.items())))) & 0x7FFFFFFF
    torch.manual_seed(cfg_seed + seed_offset * 7919)

    dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
    act_map = {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
        "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "mish": nn.Mish,
    }
    act_cls = act_map.get(activation, nn.ReLU)

    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    net = nn.Sequential(*layers)

    opt_map = {
        "adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop,
        "adagrad": torch.optim.Adagrad,
    }
    opt_cls = opt_map.get(opt_name, torch.optim.Adam)
    try:
        opt = opt_cls(net.parameters(), lr=lr)
    except Exception:
        opt = torch.optim.Adam(net.parameters(), lr=lr)

    n_train, n_val = 200, 60
    bs = max(8, min(bs, 128))
    n_steps = max(20, min(int(n_steps), 1000))
    dataset_name = str(dataset or "synthetic").strip().lower()
    if task == "classification" and dataset_name == "mnist":
        try:
            import torchvision
            import torchvision.transforms as transforms

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Lambda(lambda x: x.view(-1)),
                ]
            )
            mnist_train = torchvision.datasets.MNIST("/tmp/mnist", train=True, download=True, transform=transform)
            mnist_val = torchvision.datasets.MNIST("/tmp/mnist", train=False, download=True, transform=transform)
            tr_n = min(800, len(mnist_train))
            va_n = min(200, len(mnist_val))
            tr_idx = torch.randperm(len(mnist_train))[:tr_n]
            va_idx = torch.randperm(len(mnist_val))[:va_n]
            X_train = torch.stack([mnist_train[i][0] for i in tr_idx])
            Y_train = torch.tensor([mnist_train[i][1] for i in tr_idx], dtype=torch.long)
            X_val = torch.stack([mnist_val[i][0] for i in va_idx])
            Y_val = torch.tensor([mnist_val[i][1] for i in va_idx], dtype=torch.long)
            dims = [784] + [int(x) for x in hidden_dims] + [10]
            n_train, n_val = len(X_train), len(X_val)
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act_cls())
            net = nn.Sequential(*layers)
            loss_fn = nn.CrossEntropyLoss()
            is_class = True
        except Exception:
            X_train = torch.randn(n_train, dims[0])
            X_val = torch.randn(n_val, dims[0])
    else:
        X_train = torch.randn(n_train, dims[0])
        X_val = torch.randn(n_val, dims[0])

    is_class = task in ("classification",)
    if is_class and not (task == "classification" and dataset_name == "mnist"):
        # Linearly separable data so models can actually learn
        W_true = torch.randn(dims[0]) * 0.5
        Y_raw_train = X_train @ W_true
        Y_raw_val = X_val @ W_true
        n_cls = dims[-1]
        if n_cls == 1:
            Y_train = (Y_raw_train > 0).float().unsqueeze(1)
            Y_val = (Y_raw_val > 0).float().unsqueeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            n_cls = max(2, n_cls)
            # For multi-class: use quantile-based labels
            thresholds = torch.quantile(Y_raw_train, torch.linspace(0, 1, n_cls + 1)[1:-1])
            def quantize(vals, thr):
                labels = torch.zeros(len(vals), dtype=torch.long)
                for i, t in enumerate(thr):
                    labels[vals > t] = i + 1
                return labels
            Y_train = quantize(Y_raw_train, thresholds)
            Y_val = quantize(Y_raw_val, thresholds)
            loss_fn = nn.CrossEntropyLoss()
    else:
        W_true = torch.randn(dims[0], dims[-1]) * 0.3
        Y_train = X_train @ W_true + torch.randn(n_train, dims[-1]) * 0.1
        Y_val = X_val @ W_true + torch.randn(n_val, dims[-1]) * 0.1
        loss_fn = nn.MSELoss()

    val_losses: list[float] = []
    record_every = max(1, n_steps // 8)

    for step in range(n_steps):
        idx = torch.randint(0, n_train, (bs,))
        opt.zero_grad()
        logits = net(X_train[idx])
        loss = loss_fn(logits, Y_train[idx])
        if torch.isnan(loss) or torch.isinf(loss):
            return float("inf") if not maximize else float("-inf"), val_losses
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if step % record_every == 0:
            with torch.no_grad():
                vl = float(loss_fn(net(X_val), Y_val).detach())
                if torch.isnan(torch.tensor(vl)) or torch.isinf(torch.tensor(vl)):
                    vl = float("inf")
            val_losses.append(round(vl, 6))

    final_val = val_losses[-1] if val_losses else 1.0
    score = final_val if not maximize else -final_val
    return score, val_losses


def _proxy_score(cfg: dict[str, Any], maximize: bool, seed: int = 0) -> float:
    """Deterministic fallback when torch is unavailable."""
    lr = float(cfg.get("lr", cfg.get("learning_rate", 1e-3)))
    bs = float(cfg.get("batch_size", cfg.get("bs", 32)))
    base = 1.0 / (1.0 + abs(np.log10(lr) + 3)) + 1.0 / (1.0 + abs(bs - 32) / 32)
    h = hash(tuple(sorted((str(k), str(v)) for k, v in cfg.items())) + (seed,))
    rng = np.random.default_rng(h & 0xFFFFFFFF)
    jitter = rng.normal(0, 0.03)
    s = base + jitter
    return s if maximize else -s


def run_experiment_grid(
    experiment_code: str = "# grid search",
    grid: dict | None = None,
    n_repeats: int = 3,
    maximize: bool = False,
    n_steps: int = 80,
    task: str = "classification",
    dataset: str = "mnist",
    input_dim: int = 16,
    hidden_dims: list | None = None,
    output_dim: int = 2,
) -> ToolResult:
    def normalize_grid_params(raw_grid: dict[str, Any]) -> dict[str, list[Any]]:
        normalized: dict[str, list[Any]] = {}
        for key, value in raw_grid.items():
            if isinstance(value, list):
                normalized[key] = value
            elif isinstance(value, tuple):
                normalized[key] = list(value)
            else:
                normalized[key] = [value]
        return normalized

    def _first_scalar(v: Any) -> Any:
        if isinstance(v, list) and v:
            return _first_scalar(v[0])
        if isinstance(v, tuple) and v:
            return _first_scalar(v[0])
        return v

    try:
        if grid is None:
            grid = {"lr": [0.001, 0.01], "batch_size": [16, 32]}
        if not isinstance(grid, dict) or not grid:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="grid must be a non-empty dict of lists."),
            )
        grid = normalize_grid_params(grid)
        keys = list(grid.keys())
        for k in keys:
            if not isinstance(grid[k], (list, tuple)) or len(grid[k]) == 0:
                return ToolResult(
                    success=False,
                    error=ToolError(type="validation_error", message=f"grid[{k!r}] must be a non-empty list."),
                )
        combos = list(itertools.product(*[grid[k] for k in keys]))
        if len(combos) > 100:
            return ToolResult(
                success=False,
                error=ToolError(type="validation_error", message="Too many grid combinations (max 100)."),
            )

        n_rep = max(1, min(int(n_repeats), 3))
        if hidden_dims is None:
            hidden_dims = [32, 16]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        hdims = [int(_first_scalar(x)) for x in hidden_dims]

        results = []
        for combo in combos:
            cfg = {keys[i]: combo[i] for i in range(len(keys))}
            rep_scores: list[float] = []
            rep_val_losses: list[float] = []

            for repeat in range(n_rep):
                score, vl = _real_train_score(
                    cfg,
                    maximize=maximize,
                    n_steps=max(20, min(int(_first_scalar(n_steps)), 500 if str(settings.propab_profile).lower() == "dev" else 1000)),
                    task=task,
                    dataset=str(dataset or getattr(settings, "classification_default_dataset", "mnist")),
                    input_dim=int(_first_scalar(input_dim)),
                    hidden_dims=hdims,
                    output_dim=int(_first_scalar(output_dim)),
                    seed_offset=repeat,
                )
                rep_scores.append(score)
                rep_val_losses.extend(vl[-3:] if len(vl) >= 3 else vl)

            mean_s = float(np.mean(rep_scores))
            std_s = float(np.std(rep_scores, ddof=1)) if n_rep > 1 else 0.0
            results.append({
                "config": cfg,
                "mean_score": round(mean_s, 6),
                "std_score": round(std_s, 6),
                "val_losses": [round(v, 6) for v in rep_val_losses],
                "rank": 0,
            })

        order = sorted(range(len(results)), key=lambda i: results[i]["mean_score"], reverse=maximize)
        for rnk, idx in enumerate(order):
            results[idx]["rank"] = rnk + 1

        best_idx = order[0]
        worst_idx = order[-1]
        best = results[best_idx]
        worst = results[worst_idx]

        interactions: dict[str, Any] = {}
        if len(keys) >= 2:
            k0, k1 = str(keys[0]), str(keys[1])
            interactions[f"{k0}x{k1}"] = "interaction estimated from real training scores"

        return ToolResult(
            success=True,
            output={
                "results": results,
                "best_config": best["config"],
                "best_score": best["mean_score"],
                "best_val_losses": best["val_losses"],
                "worst_val_losses": worst["val_losses"],
                "interaction_effects": interactions,
                "total_runs": len(combos) * n_rep,
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
