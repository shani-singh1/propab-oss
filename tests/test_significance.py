from __future__ import annotations

import pytest

from services.worker.significance import (
    SignificanceResult,
    any_significance_tool_ran,
    check_significance,
    classify_verdict,
    fisher_combine_p_values,
)


def _evidence(**kw):
    base = {
        "n_metric_steps": 2,
        "relevance_score": 0.5,
        "p_value": 0.01,
        "delta": -0.1,
        "delta_pct": -5.0,
        "effect_size": 0.6,
    }
    base.update(kw)
    return base


def test_classify_verdict_confirmed_when_replicated():
    sig = SignificanceResult(gate_passed=True, p_value=0.01, effect_size=0.6)
    verdict, _ = classify_verdict(_evidence(n_metric_steps=2), sig)
    assert verdict == "confirmed"


def test_classify_verdict_unreplicated_downgraded_to_inconclusive():
    sig = SignificanceResult(gate_passed=True, p_value=0.01, effect_size=0.6)
    verdict, reason = classify_verdict(_evidence(n_metric_steps=1), sig)
    assert verdict == "inconclusive"
    assert "unreplicated" in reason


def test_classify_verdict_single_step_confirmable_when_bar_is_one():
    sig = SignificanceResult(gate_passed=True, p_value=0.01, effect_size=0.6)
    verdict, _ = classify_verdict(
        _evidence(n_metric_steps=1), sig, min_metric_steps_for_confirm=1
    )
    assert verdict == "confirmed"


def test_classify_verdict_no_metric_steps_inconclusive():
    sig = SignificanceResult(gate_passed=True)
    verdict, reason = classify_verdict(_evidence(n_metric_steps=0), sig)
    assert verdict == "inconclusive"
    assert "no metric-bearing steps" in reason


def test_classify_verdict_refuted_on_definitive_failure():
    sig = SignificanceResult(gate_passed=False, gate_definitively_failed=True, p_value=0.4)
    verdict, _ = classify_verdict(_evidence(), sig)
    assert verdict == "refuted"


def test_classify_verdict_gate_not_passed_inconclusive():
    sig = SignificanceResult(gate_passed=False)
    verdict, _ = classify_verdict(_evidence(), sig)
    assert verdict == "inconclusive"


def test_classify_verdict_ambiguous_direction_inconclusive():
    sig = SignificanceResult(gate_passed=True, p_value=0.2)
    # p_value present but >= 0.05 → direction not supported
    verdict, reason = classify_verdict(
        _evidence(p_value=0.2, effect_size=None, delta_pct=None), sig
    )
    assert verdict == "inconclusive"
    assert "ambiguous" in reason


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
