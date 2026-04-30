from __future__ import annotations

import pytest

from services.worker.think_act import (
    AgentContext,
    _fallback_significance_action,
    _parse_action,
    should_stop,
)


def _make_ctx(**kwargs) -> AgentContext:
    defaults = dict(
        hypothesis_text="Does pre-norm help under high noise?",
        test_methodology="Compare pre-norm vs post-norm with statistical_significance",
        learned_from=None,
        peer_findings=[],
        results_so_far=[],
        tool_names_run=[],
        steps_taken=0,
        max_steps=10,
        min_steps=3,
    )
    defaults.update(kwargs)
    return AgentContext(**defaults)


def test_should_stop_at_max_steps():
    ctx = _make_ctx(steps_taken=10, max_steps=10)
    assert should_stop(ctx) is True


def test_should_not_stop_before_max():
    ctx = _make_ctx(steps_taken=4, max_steps=10)
    assert should_stop(ctx) is False


def test_parse_action_tool():
    data = {
        "action_type": "tool",
        "tool_name": "statistical_significance",
        "params": {"results_a": [0.9, 0.88], "results_b": [0.8, 0.78]},
        "reasoning": "need p-value",
        "expected_outcome": "p_value < 0.05",
    }
    action = _parse_action(data)
    assert action.action_type == "tool"
    assert action.tool_name == "statistical_significance"
    assert "results_a" in action.params


def test_parse_action_stop():
    data = {"action_type": "stop", "reasoning": "enough evidence"}
    action = _parse_action(data)
    assert action.action_type == "stop"


def test_parse_action_unknown_defaults_to_stop():
    data = {"action_type": "fly", "reasoning": "test"}
    action = _parse_action(data)
    assert action.action_type == "stop"


def test_fallback_significance_action_with_numbers():
    ctx = _make_ctx(results_so_far=[{"loss": 0.3, "accuracy": 0.85}, {"loss": 0.28}])
    action = _fallback_significance_action(ctx)
    assert action.action_type == "tool"
    assert action.tool_name == "bootstrap_confidence"
    assert len(action.params["values"]) >= 2


def test_fallback_significance_action_no_numbers():
    ctx = _make_ctx(results_so_far=[{"status": "ok"}])
    action = _fallback_significance_action(ctx)
    assert action.action_type == "stop"


def test_agent_context_results_summary_empty():
    ctx = _make_ctx()
    assert "No results" in ctx.to_results_summary()


def test_agent_context_results_summary_truncates():
    big_result = {"data": "x" * 2000}
    ctx = _make_ctx(results_so_far=[big_result] * 10)
    # 10 results of ~2000 chars each; full summary would be ~20000 chars.
    summary = ctx.to_results_summary(max_chars=3000)
    # Should contain the truncation marker
    assert "truncated" in summary
    # Should be notably shorter than the untruncated version
    full = ctx.to_results_summary(max_chars=100_000)
    assert len(summary) < len(full)


def test_agent_context_significance_status_no_results():
    ctx = _make_ctx()
    sig = ctx.significance_status()
    assert sig.gate_passed is False


def test_agent_context_significance_status_with_p_value():
    ctx = _make_ctx(results_so_far=[{"p_value": 0.03, "effect_size": 0.45}])
    sig = ctx.significance_status()
    assert sig.gate_passed is True
