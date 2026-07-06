"""Tests for significance-input provenance tracking in the worker.

These cover the helpers that classify whether a significance tool's numeric
inputs TRACE to a prior real tool/sandbox output ("computed") versus being
agent-typed literals with no upstream source ("agent_literal").
"""
from __future__ import annotations

from services.worker.sub_agent_loop import (
    _array_traces_to_prior,
    _classify_stat_input_provenance,
    _collect_numeric_arrays,
)


def test_collect_numeric_arrays_flat_and_nested():
    payload = {
        "val_losses": [0.3, 0.28, 0.31],
        "meta": {"scores": [0.9, 0.8]},
        "results": [{"mean_score": 0.5}, {"vals": [1.0, 2.0, 3.0]}],
        "scalar": 0.42,  # ignored: not a list
        "mixed": [1, "x"],  # ignored: not homogeneous numeric
    }
    arrays = _collect_numeric_arrays(payload)
    assert [0.3, 0.28, 0.31] in arrays
    assert [0.9, 0.8] in arrays
    assert [1.0, 2.0, 3.0] in arrays
    assert all(len(a) >= 2 for a in arrays)


def test_array_traces_exact_match():
    prior = [[0.3, 0.28, 0.31, 0.29]]
    assert _array_traces_to_prior([0.3, 0.28, 0.31, 0.29], prior) is True


def test_array_traces_subset_slice():
    # Agent passes val_losses[:2] of a longer computed array.
    prior = [[0.30, 0.28, 0.31, 0.29, 0.27]]
    assert _array_traces_to_prior([0.28, 0.30], prior) is True


def test_array_does_not_trace_when_unrelated():
    prior = [[0.30, 0.28, 0.31, 0.29]]
    assert _array_traces_to_prior([0.99, 0.98, 0.97], prior) is False


def test_provenance_computed_when_inputs_from_prior_output():
    # Two prior real training outputs, agent passes their val_losses verbatim.
    prior_outputs = [
        {"val_losses": [0.30, 0.28, 0.31, 0.29]},
        {"val_losses": [0.22, 0.24, 0.21, 0.23]},
    ]
    params = {
        "results_a": [0.30, 0.28, 0.31, 0.29],
        "results_b": [0.22, 0.24, 0.21, 0.23],
    }
    assert _classify_stat_input_provenance(params, prior_outputs) == "computed"


def test_provenance_agent_literal_when_no_upstream_source():
    # A prior output exists but the agent-typed arrays do NOT trace to it.
    prior_outputs = [{"val_losses": [0.30, 0.28, 0.31, 0.29]}]
    params = {
        "results_a": [0.91, 0.92, 0.93],
        "results_b": [0.80, 0.81, 0.82],
    }
    assert _classify_stat_input_provenance(params, prior_outputs) == "agent_literal"


def test_provenance_agent_literal_when_no_prior_outputs_at_all():
    # No prior computed arrays to corroborate -> cannot be "computed".
    params = {"values": [0.5, 0.6, 0.7]}
    assert _classify_stat_input_provenance(params, []) == "agent_literal"


def test_provenance_mixed_inputs_flagged_agent_literal():
    # results_a is real, results_b is invented -> conservative: agent_literal.
    prior_outputs = [{"val_losses": [0.30, 0.28, 0.31, 0.29]}]
    params = {
        "results_a": [0.30, 0.28, 0.31, 0.29],  # traces
        "results_b": [0.11, 0.12, 0.13, 0.14],  # invented
    }
    assert _classify_stat_input_provenance(params, prior_outputs) == "agent_literal"


def test_provenance_unknown_when_no_numeric_array_inputs():
    # e.g. a significance call with only scalar params (nothing to trace).
    prior_outputs = [{"val_losses": [0.30, 0.28, 0.31]}]
    params = {"baseline_value": 0.5}
    assert _classify_stat_input_provenance(params, prior_outputs) == "unknown"
