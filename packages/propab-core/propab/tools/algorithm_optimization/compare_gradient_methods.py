from __future__ import annotations

import math

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


def _adam_bias_denom(beta: float, t1: int, eps: float = 1e-8) -> float:
    """Stable 1 - beta**t1 for bias correction (avoids float overflow in beta**(t+1))."""
    if t1 <= 0 or not (0.0 < beta < 1.0):
        return max(eps, 1.0 - beta if beta <= 1.0 else eps)
    lx = t1 * math.log(beta)
    # For large t1, beta**t1 -> 0 and 1-beta**t1 -> 1; -expm1(lx) == 1 - exp(lx) == 1 - beta**t1
    return max(float(-math.expm1(lx)), eps)


def compare_gradient_methods(
    methods: list,
    problem: str = "rosenbrock",
    n_steps: int = 400,
    learning_rate: float = 0.01,
    init_point: list | None = None,
) -> ToolResult:
    if problem != "rosenbrock":
        return ToolResult(success=False, error=ToolError(type="validation_error", message="v1 supports problem=rosenbrock only."))
    if not methods:
        return ToolResult(success=False, error=ToolError(type="validation_error", message="methods must be a non-empty list."))
    n_steps = max(20, min(int(n_steps), 5000))
    xy0 = np.array(init_point if init_point else [-1.0, 1.0], dtype=np.float64)
    trajectories = []
    try:
        for m in methods:
            xy = xy0.copy()
            curve = []
            lr = float(learning_rate)
            m1 = np.zeros(2, dtype=np.float64)
            m2 = np.zeros(2, dtype=np.float64)
            for t in range(n_steps):
                if not np.all(np.isfinite(xy)):
                    break
                g = _grad_rosen(xy)
                if not np.all(np.isfinite(g)):
                    break
                curve.append(float(_rosenbrock(xy)))
                if not math.isfinite(curve[-1]):
                    break
                name = str(m).lower()
                if name == "sgd":
                    xy -= lr * g
                elif name in ("adam", "adamw"):
                    beta1, beta2, eps = 0.9, 0.999, 1e-8
                    m1 = beta1 * m1 + (1.0 - beta1) * g
                    m2 = beta2 * m2 + (1.0 - beta2) * (g * g)
                    tp1 = t + 1
                    d1 = _adam_bias_denom(beta1, tp1, eps)
                    d2 = _adam_bias_denom(beta2, tp1, eps)
                    m1h = m1 / d1
                    m2h = m2 / d2
                    upd = lr * m1h / (np.sqrt(m2h) + eps)
                    if not np.all(np.isfinite(upd)):
                        break
                    xy -= upd
                else:
                    xy -= lr * g
            if not curve:
                curve.append(float(_rosenbrock(xy0)))
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
    except (FloatingPointError, OverflowError, ValueError) as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=f"optimization diverged: {exc}"))
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
