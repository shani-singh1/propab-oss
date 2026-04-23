from __future__ import annotations

import hashlib

import numpy as np

from propab.tools.types import ToolError, ToolResult

_VALID = frozenset({"standard", "causal", "sliding_window", "linear", "random_sparse"})

TOOL_SPEC = {
    "name": "compare_attention_variants",
    "domain": "deep_learning",
    "description": "Compare attention variants on synthetic speed/memory/accuracy (numpy v1).",
    "params": {
        "variants": {"type": "list[str]", "required": True},
        "seq_lengths": {"type": "list[int]", "required": True},
        "d_model": {"type": "int", "required": False, "default": 128},
        "n_heads": {"type": "int", "required": False, "default": 4},
        "task": {"type": "str", "required": False, "default": "sequence_classification"},
    },
    "output": {
        "comparison": "list",
        "pareto_front": "list[str]",
        "summary": "str",
    },
    "example": {
        "params": {
            "variants": ["standard", "linear"],
            "seq_lengths": [32, 64],
            "d_model": 64,
            "n_heads": 4,
        },
        "output": {},
    },
}


def _pareto_front(rows: list[dict]) -> list[str]:
    """Maximize accuracy, minimize wall_time_ms — keep non-dominated variant names (best per variant across seq)."""
    by_variant: dict[str, list[dict]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    scores: dict[str, tuple[float, float]] = {}
    for v, lst in by_variant.items():
        acc = float(np.mean([x["accuracy"] for x in lst]))
        t = float(np.mean([x["wall_time_ms"] for x in lst]))
        scores[v] = (acc, t)
    names = list(scores.keys())
    front: list[str] = []
    for i, vi in enumerate(names):
        ai, ti = scores[vi]
        dominated = False
        for j, vj in enumerate(names):
            if i == j:
                continue
            aj, tj = scores[vj]
            if (aj >= ai and tj <= ti) and (aj > ai or tj < ti):
                dominated = True
                break
        if not dominated:
            front.append(vi)
    return sorted(front)


def compare_attention_variants(
    variants: list,
    seq_lengths: list,
    d_model: int = 128,
    n_heads: int = 4,
    task: str = "sequence_classification",
) -> ToolResult:
    try:
        bad = [v for v in variants if str(v) not in _VALID]
        if bad:
            return ToolResult(
                success=False,
                error=ToolError(
                    type="validation_error",
                    message=f"Unknown variants (allowed: {_VALID}): {bad}",
                ),
            )
        if not seq_lengths:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="seq_lengths required."))
        dm = int(d_model)
        nh = max(1, int(n_heads))
        seed = int(hashlib.sha256((task + str(variants)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        comparison: list[dict] = []
        for variant in variants:
            v = str(variant)
            for s in seq_lengths:
                s = int(s)
                if s <= 0:
                    continue
                if v == "linear":
                    flops = float(s * dm * dm)
                    mem = float(s * dm * 4 / 1e6)
                    wall = float(0.5 + 0.001 * s + rng.normal(0, 0.02))
                elif v in ("random_sparse", "sliding_window"):
                    flops = float(s * (dm**1.2) * nh)
                    mem = float(s * dm * 6 / 1e6)
                    wall = float(1.2 + 0.008 * s + rng.normal(0, 0.03))
                else:
                    flops = float(s * s * dm)
                    mem = float(s * s * dm * 4 / 1e6)
                    wall = float(2.0 + 0.00002 * (s * s) + rng.normal(0, 0.04))
                acc = float(
                    0.55
                    + 0.35 / (1.0 + wall / 50.0)
                    + (0.02 if v == "standard" else 0.0)
                    + rng.normal(0, 0.008)
                )
                acc = float(np.clip(acc, 0.0, 1.0))
                comparison.append(
                    {
                        "variant": v,
                        "seq_len": s,
                        "flops": flops,
                        "memory_mb": mem,
                        "accuracy": acc,
                        "wall_time_ms": max(0.05, wall),
                    }
                )
        pf = _pareto_front(comparison)
        summary = (
            f"Synthetic {task} comparison for {len(variants)} variant(s) × {len(seq_lengths)} seq lengths; "
            f"Pareto set: {', '.join(pf) or 'n/a'}."
        )
        return ToolResult(success=True, output={"comparison": comparison, "pareto_front": pf, "summary": summary})
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
