from propab.paper_gate import (
    merit_from_ledger,
    merit_from_trace_rows,
    output_suggests_metrics,
    short_circuit_merits_paper,
)


def test_merit_from_ledger_confirmed() -> None:
    ok, r = merit_from_ledger({"confirmed": ["a"], "refuted": [], "inconclusive": []})
    assert ok and r == "confirmed_hypotheses"


def test_merit_from_ledger_refuted() -> None:
    ok, r = merit_from_ledger({"confirmed": [], "refuted": ["b"], "inconclusive": []})
    assert ok and r == "refuted_hypotheses"


def test_merit_from_trace_metric_blob() -> None:
    rows = [
        {
            "step_type": "tool_call",
            "input_json": {"tool": "train_model", "params": {}},
            "output_json": {"train_loss": [0.5, 0.2], "epochs": 3, "baseline": 0.4},
            "error_json": None,
        }
    ]
    ok, r = merit_from_trace_rows(rows)
    assert ok and "metric" in r


def test_merit_from_trace_two_substantive_tools() -> None:
    rows = [
        {
            "step_type": "tool_call",
            "input_json": {"tool": "evaluate_model", "params": {}},
            "output_json": {"accuracy": 0.91},
            "error_json": None,
        },
        {
            "step_type": "tool_call",
            "input_json": {"tool": "loss_landscape", "params": {}},
            "output_json": {"summary": "flat region"},
            "error_json": None,
        },
    ]
    ok, r = merit_from_trace_rows(rows)
    assert ok and "substantive" in r


def test_output_suggests_metrics_false_on_empty() -> None:
    assert output_suggests_metrics({}) is False


def test_short_circuit_merits_default_false(monkeypatch) -> None:
    from propab import config

    monkeypatch.setattr(config.settings, "paper_policy", "substantive")
    ok, r = short_circuit_merits_paper()
    assert not ok and "literature" in r
