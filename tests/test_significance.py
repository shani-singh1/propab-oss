from __future__ import annotations

import pytest

from services.worker.significance import (
    SignificanceResult,
    any_significance_tool_ran,
    check_significance,
    fisher_combine_p_values,
)


def test_gate_passes_on_p_value():
    results = [{"p_value": 0.02, "statistic": 2.5, "effect_size": 0.4}]
    sig = check_significance(results)
    assert sig.gate_passed is True
    assert sig.p_value == pytest.approx(0.02)
    assert sig.method == "p_value"


def test_gate_passes_on_effect_size_no_p():
    results = [{"effect_size": 0.55}]
    sig = check_significance(results)
    assert sig.gate_passed is True
    assert sig.method == "effect_size"


def test_gate_passes_on_confidence_interval():
    results = [{"confidence_interval": [0.05, 0.40]}]
    sig = check_significance(results)
    assert sig.gate_passed is True
    assert sig.method == "confidence_interval"


def test_gate_passes_ci_lower_upper_keys():
    results = [{"ci_lower": 0.10, "ci_upper": 0.50}]
    sig = check_significance(results)
    assert sig.gate_passed is True


def test_gate_not_passed_high_p():
    results = [{"p_value": 0.42, "effect_size": 0.03}]
    sig = check_significance(results)
    assert sig.gate_passed is False
    assert sig.gate_definitively_failed is True


def test_gate_pending_no_stat_evidence():
    results = [{"loss": 0.35, "accuracy": 0.82}]
    sig = check_significance(results)
    assert sig.gate_passed is False
    assert sig.gate_definitively_failed is False  # no stat test ran


def test_gate_not_passed_negligible_effect():
    results = [{"effect_size": 0.05}]
    sig = check_significance(results)
    assert sig.gate_passed is False


def test_empty_results():
    sig = check_significance([])
    assert sig.gate_passed is False
    assert sig.gate_definitively_failed is False


def test_multiple_outputs_takes_best_p():
    results = [{"p_value": 0.15}, {"p_value": 0.03}]
    sig = check_significance(results)
    assert sig.gate_passed is True
    assert sig.p_value == pytest.approx(0.03)


def test_any_significance_tool_ran_true():
    assert any_significance_tool_ran(["train_model", "statistical_significance"]) is True


def test_any_significance_tool_ran_false():
    assert any_significance_tool_ran(["train_model", "build_transformer"]) is False


def test_fisher_combine():
    # Two independently significant results should combine to a more significant p
    combined = fisher_combine_p_values([0.04, 0.03])
    assert combined < 0.04


def test_fisher_combine_empty():
    assert fisher_combine_p_values([]) == 1.0


def test_fisher_combine_single():
    combined = fisher_combine_p_values([0.04])
    assert 0.0 < combined < 0.15
