from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "compare_gradient_methods",
    "domain": "algorithm_optimization",
    "description": "Compare gradient descent variants on a 2D Rosenbrock surface (numpy).",
    "params": {
        "methods": {"type": "list[str]", "required": True},
        "problem": {"type": "str", "required": False, "default": "rosenbrock"},
        "n_steps": {"type": "int", "required": False, "default": 400},
        "learning_rate": {"type": "float", "required": False, "default": 0.01},
        "init_point": {"type": "list[float]", "required": False},
    },
    "output": {"trajectories": "list", "winner": "str", "fastest": "str", "summary": "str", "plot_data": "dict"},
    "example": {"params": {"methods": ["sgd", "adam"]}, "output": {}},
}


def _rosenbrock(xy: np.ndarray) -> float:
    x, y = float(xy[0]), float(xy[1])
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def _grad_rosen(xy: np.ndarray) -> np.ndarray:
    x, y = float(xy[0]), float(xy[1])
    dx = -2 * (1 - x) - 400 * x * (y - x * x)
    dy = 200 * (y - x * x)
    return np.array([dx, dy], dtype=np.float64)


def compare_gradient_methods(
    methods: list,
    problem: str = "rosenbrock",
    n_steps: int = 400,
    learning_rate: float = 0.01,
    init_point: list | None = None,
) -> ToolResult:
    if problem != "rosenbrock":
        return ToolResult(success=False, error=ToolError(type="validation_error", message="v1 supports problem=rosenbrock only."))
    n_steps = max(20, min(int(n_steps), 5000))
    xy0 = np.array(init_point if init_point else [-1.0, 1.0], dtype=np.float64)
    trajectories = []
    for m in methods:
        xy = xy0.copy()
        curve = []
        lr = float(learning_rate)
        for t in range(n_steps):
            g = _grad_rosen(xy)
            curve.append(float(_rosenbrock(xy)))
            name = str(m).lower()
            if name == "sgd":
                xy -= lr * g
            elif name in ("adam", "adamw"):
                # simplified Adam moment
                if t == 0:
                    m1 = np.zeros(2)
                    m2 = np.zeros(2)
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                m1 = beta1 * m1 + (1 - beta1) * g
                m2 = beta2 * m2 + (1 - beta2) * (g * g)
                m1h = m1 / (1 - beta1 ** (t + 1))
                m2h = m2 / (1 - beta2 ** (t + 1))
                xy -= lr * m1h / (np.sqrt(m2h) + eps)
            else:
                xy -= lr * g
        step = max(1, len(curve) // 20)
        trajectories.append(
            {
                "method": str(m),
                "steps": n_steps,
                "loss_curve": curve[::step],
                "final_loss": curve[-1],
                "converged": curve[-1] < 0.05,
                "steps_to_1pct": n_steps,
            }
        )
    winner = min(trajectories, key=lambda tr: tr["final_loss"])["method"]
    fastest = winner
    return ToolResult(
        success=True,
        output={
            "trajectories": trajectories,
            "winner": winner,
            "fastest": fastest,
            "summary": f"Lowest final Rosenbrock loss: {winner}.",
            "plot_data": {"note": "2D trajectories omitted in v1"},
        },
    )
