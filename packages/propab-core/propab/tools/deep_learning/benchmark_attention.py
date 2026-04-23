from __future__ import annotations

import numpy as np

from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "benchmark_attention",
    "domain": "deep_learning",
    "description": "Numpy softmax-attention timing and memory proxy (no torch required).",
    "params": {
        "seq_len": {"type": "int", "required": True},
        "d_model": {"type": "int", "required": True},
        "n_heads": {"type": "int", "required": True},
        "n_trials": {"type": "int", "required": False, "default": 3},
    },
    "output": {"mean_time_ms": "float", "peak_memory_mb": "float", "flops_estimate": "int", "summary": "str"},
    "example": {"params": {"seq_len": 64, "d_model": 128, "n_heads": 4}, "output": {}},
}


def benchmark_attention(seq_len: int, d_model: int, n_heads: int, n_trials: int = 3) -> ToolResult:
    try:
        import time

        seq_len = int(seq_len)
        d_model = int(d_model)
        n_heads = int(n_heads)
        if d_model % n_heads != 0:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="d_model must divide n_heads."))
        d_h = d_model // n_heads
        flops = 2 * seq_len * seq_len * d_model  # QK^T + @V order of magnitude
        times = []
        for _ in range(max(1, int(n_trials))):
            q = np.random.randn(seq_len, d_model).astype(np.float32)
            k = np.random.randn(seq_len, d_model).astype(np.float32)
            v = np.random.randn(seq_len, d_model).astype(np.float32)
            t0 = time.perf_counter()
            attn = q @ k.T / np.sqrt(float(d_model))
            attn = np.exp(attn - attn.max(axis=1, keepdims=True))
            attn /= attn.sum(axis=1, keepdims=True) + 1e-9
            _ = attn @ v
            times.append((time.perf_counter() - t0) * 1000)
        mem_mb = (seq_len * d_model * 3 * 4) / (1024 * 1024) * 3  # rough peak
        return ToolResult(
            success=True,
            output={
                "mean_time_ms": round(float(np.mean(times)), 4),
                "peak_memory_mb": round(mem_mb, 4),
                "flops_estimate": int(flops),
                "summary": f"Softmax-attention proxy seq={seq_len}, d={d_model}, H={n_heads}.",
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
