from propab.tool_chain import refine_next_tool_step


def test_refine_injects_model_id_after_build_mlp() -> None:
    first_out = {"model_id": "abc-123", "param_count": 42}
    nxt = {"type": "tool", "tool": "count_parameters", "params": {"model_id": "dummy"}}
    out = refine_next_tool_step("build_mlp", first_out, nxt)
    assert out["params"]["model_id"] == "abc-123"


def test_refine_noop_for_unrelated_first_tool() -> None:
    nxt = {"type": "tool", "tool": "numeric_summary", "params": {"values": [1, 2, 3]}}
    out = refine_next_tool_step("json_extract", {"exists": True}, nxt)
    assert out == nxt


def test_refine_skips_non_tool_next() -> None:
    nxt = {"type": "code", "code": "pass"}
    out = refine_next_tool_step("build_mlp", {"model_id": "x"}, nxt)
    assert out == nxt


def test_refine_injects_model_id_for_hessian_analysis() -> None:
    nxt = {"type": "tool", "tool": "hessian_analysis", "params": {"dataset": "synthetic"}}
    out = refine_next_tool_step("build_transformer", {"model_id": "mid-99"}, nxt)
    assert out["params"]["model_id"] == "mid-99"
