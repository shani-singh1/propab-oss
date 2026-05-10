from services.worker.sub_agent_loop import _primary_metric_from_tool_output


def test_primary_metric_percentage_in_nested_accuracy_key_path() -> None:
    fv = _primary_metric_from_tool_output({"run": {"val_accuracy_pct": 96.25}})
    assert fv is not None
    assert abs(fv - 0.9625) < 1e-6
