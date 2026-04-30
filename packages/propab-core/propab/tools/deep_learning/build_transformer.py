from __future__ import annotations

from uuid import uuid4

from propab.tools.model_registry import put_model
from propab.tools.types import ToolError, ToolResult

TOOL_SPEC = {
    "name": "build_transformer",
    "domain": "deep_learning",
    "description": "Analytical transformer parameter / FLOP summary (v1 encoder-style estimate).",
    "params": {
        "model_type": {"type": "str", "required": False, "default": "encoder",
                       "enum": ["encoder", "decoder", "encoder_decoder"]},
        "d_model": {"type": "int", "required": False, "default": 128},
        "n_heads": {"type": "int", "required": False, "default": 4},
        "n_layers": {"type": "int", "required": False, "default": 2},
        "d_ff": {"type": "int", "required": False, "default": None},
        "max_seq_len": {"type": "int", "required": False, "default": 512},
        "dropout": {"type": "float", "required": False, "default": 0.1},
        "vocab_size": {"type": "int", "required": False, "default": None},
    },
    "output": {
        "param_count": "int",
        "layer_shapes": "dict",
        "architecture_str": "str",
        "model_id": "str",
        "attention_flops_per_token": "int",
    },
    "example": {
        "params": {"model_type": "encoder", "d_model": 128, "n_heads": 4, "n_layers": 2},
        "output": {},
    },
}


def build_transformer(
    model_type: str = "encoder",
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int | None = None,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    vocab_size: int | None = None,
) -> ToolResult:
    try:
        d_model = int(d_model)
        n_heads = int(n_heads)
        n_layers = int(n_layers)
        if d_model % n_heads != 0:
            return ToolResult(success=False, error=ToolError(type="validation_error", message="d_model must divide n_heads."))
        d_ff = int(d_ff) if d_ff is not None else 4 * d_model
        vs = int(vocab_size) if vocab_size is not None else max(256, d_model)
        # Rough param estimate: embeddings + n_layers * (self_attn + ff) blocks
        emb = vs * d_model
        per_layer = 4 * d_model * d_model + 2 * d_model * d_ff  # attn projections + ff (very rough)
        pc = emb + n_layers * per_layer + d_model * vs  # tied lm head approx
        flops = 2 * n_layers * max_seq_len * d_model * d_model  # order-of-magnitude matmul proxy
        mid = str(uuid4())
        arch = f"Transformer({model_type}, d={d_model}, L={n_layers}, H={n_heads}, ff={d_ff})"
        put_model(
            mid,
            {
                "kind": "transformer",
                "param_count": pc,
                "d_model": d_model,
                "n_layers": n_layers,
                "flops_hint": flops,
                "max_seq_len": int(max_seq_len),
            },
        )
        return ToolResult(
            success=True,
            output={
                "param_count": pc,
                "layer_shapes": {"d_model": d_model, "n_layers": n_layers, "d_ff": d_ff},
                "architecture_str": arch,
                "model_id": mid,
                "attention_flops_per_token": int(flops // max(max_seq_len, 1)),
            },
        )
    except Exception as exc:
        return ToolResult(success=False, error=ToolError(type="execution_error", message=str(exc)))
