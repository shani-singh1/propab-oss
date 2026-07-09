from __future__ import annotations

import time

import pytest

from services.worker.think_act import (
    AgentAction,
    AgentContext,
    _fallback_significance_action,
    _is_spec_example_params,
    _lists_trivially_equal,
    _parse_action,
    _params_match_example,
    _stop_needs_evidence,
    should_stop,
)

# Minimal live-spec fixtures mirroring the real TOOL_SPEC["example"]["params"]
# for the three significance tools. These are what registry.get_cluster_with_significance
# passes into decide_next_action as `specs`.
_SIG_SPECS = [
    {
        "name": "statistical_significance",
        "significance_capable": True,
        "example": {"params": {"results_a": [0.9, 0.88, 0.91], "results_b": [0.82, 0.8, 0.79]}},
    },
    {
        "name": "bootstrap_confidence",
        "significance_capable": True,
        "example": {"params": {"values": [0.1, 0.2, 0.15, 0.18]}},
    },
    {
        "name": "literature_baseline_compare",
        "significance_capable": True,
        "example": {"params": {"our_results": [0.42, 0.44, 0.41], "baseline_value": 0.5}},
    },
    # A non-significance tool with numeric example params must never be treated
    # as a significance example source.
    {
        "name": "compute_flops",
        "significance_capable": False,
        "example": {"params": {"input_shape": [32, 784]}},
    },
]


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


def test_should_stop_when_deadline_elapsed():
    ctx = _make_ctx(steps_taken=0, max_steps=100, deadline_monotonic=time.monotonic() - 1.0)
    assert should_stop(ctx) is True
    assert ctx.time_budget_exceeded is True


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


def test_parse_action_code_captures_real_source():
    src = "import json\nprint(json.dumps({'sandbox': 'ok', 'count': 42}))\n"
    data = {
        "action_type": "code",
        "code_description": "count solutions",
        "code": src,
        "reasoning": "no tool covers this combinatorial search",
    }
    action = _parse_action(data)
    assert action.action_type == "code"
    assert action.code == src.strip()
    assert action.code_description == "count solutions"


def test_parse_action_code_without_source_is_none():
    data = {"action_type": "code", "code_description": "describe only"}
    action = _parse_action(data)
    assert action.action_type == "code"
    assert action.code is None


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


# ─── Generalized spec-example fabrication guard ───────────────────────────────


def test_guard_flags_own_spec_example_stat_sig():
    # statistical_significance passed its own spec example -> flagged.
    params = {"results_a": [0.9, 0.88, 0.91], "results_b": [0.82, 0.8, 0.79]}
    assert _is_spec_example_params("statistical_significance", params, _SIG_SPECS) is True


def test_guard_flags_own_spec_example_bootstrap():
    params = {"values": [0.1, 0.2, 0.15, 0.18]}
    assert _is_spec_example_params("bootstrap_confidence", params, _SIG_SPECS) is True


def test_guard_flags_own_spec_example_litbaseline():
    params = {"our_results": [0.42, 0.44, 0.41], "baseline_value": 0.5}
    assert _is_spec_example_params("literature_baseline_compare", params, _SIG_SPECS) is True


def test_guard_flags_cross_tool_copied_example():
    # KEY NEW BEHAVIOR: agent copies statistical_significance's example array into
    # bootstrap_confidence's `values`. The old 3-array denylist keyed by tool name
    # would MISS this; the value-based guard catches it.
    params = {"values": [0.9, 0.88, 0.91]}
    assert _is_spec_example_params("bootstrap_confidence", params, _SIG_SPECS) is True


def test_guard_flags_trivially_scaled_example():
    # Example multiplied by a constant (10x) — a trivial derivation, still fake.
    params = {"values": [1.0, 2.0, 1.5, 1.8]}
    assert _is_spec_example_params("bootstrap_confidence", params, _SIG_SPECS) is True


def test_guard_flags_reordered_example():
    params = {"results_a": [0.88, 0.91, 0.9], "results_b": [0.79, 0.8, 0.82]}
    assert _is_spec_example_params("statistical_significance", params, _SIG_SPECS) is True


def test_guard_accepts_genuine_values():
    # Real measurements that do not match any spec example -> accepted.
    params = {
        "results_a": [0.312, 0.298, 0.305, 0.301],
        "results_b": [0.277, 0.281, 0.269, 0.274],
    }
    assert _is_spec_example_params("statistical_significance", params, _SIG_SPECS) is False


def test_guard_does_not_use_nonsignificance_tool_example():
    # compute_flops' input_shape [32, 784] is a non-sig tool example. An agent
    # legitimately passing [32, 784] as `values` should NOT be flagged from it.
    params = {"values": [32.0, 784.0]}
    assert _is_spec_example_params("bootstrap_confidence", params, _SIG_SPECS) is False


def test_guard_legacy_floor_without_specs():
    # No specs supplied (e.g. correction re-check) -> legacy hardcoded floor still
    # catches the three known examples, so we never regress below the old behavior.
    assert _is_spec_example_params(
        "statistical_significance",
        {"results_a": [0.9, 0.88, 0.91], "results_b": [0.82, 0.8, 0.79]},
        None,
    ) is True
    assert _is_spec_example_params(
        "bootstrap_confidence", {"values": [0.1, 0.2, 0.15, 0.18]}, None
    ) is True


def test_params_match_example_value_based():
    assert _params_match_example({"x": [0.1, 0.2, 0.15, 0.18]}, {"values": [0.1, 0.2, 0.15, 0.18]})
    assert not _params_match_example({"x": [0.5, 0.6, 0.7]}, {"values": [0.1, 0.2, 0.15, 0.18]})


def test_lists_trivially_equal_variants():
    base = [0.1, 0.2, 0.15, 0.18]
    assert _lists_trivially_equal(base, base)
    assert _lists_trivially_equal([0.2, 0.1, 0.18, 0.15], base)  # reorder
    assert _lists_trivially_equal([1.0, 2.0, 1.5, 1.8], base)  # 10x scale
    assert _lists_trivially_equal([1.1, 1.2, 1.15, 1.18], base)  # +1 offset
    assert not _lists_trivially_equal([0.3, 0.5, 0.9, 0.1], base)  # unrelated


# ─── Evidence stop-gate (significance OR verification satisfies it) ────────────

# The three sig specs plus two deterministic certifiers (verification_capable).
_GATE_SPECS = _SIG_SPECS + [
    {"name": "extremal_set_search", "verification_capable": True, "example": {"params": {"n": 7}}},
    {"name": "certify_b3_record", "verification_capable": True, "example": {"params": {"n": 2}}},
]


def _stop() -> AgentAction:
    return AgentAction(action_type="stop")


def test_stop_gate_needs_evidence_when_nothing_ran():
    ctx = _make_ctx(tool_names_run=[], steps_taken=3, min_steps=3)
    assert _stop_needs_evidence(_stop(), ctx, _GATE_SPECS) is True


def test_stop_gate_satisfied_by_significance_tool():
    ctx = _make_ctx(tool_names_run=["statistical_significance"], steps_taken=3, min_steps=3)
    assert _stop_needs_evidence(_stop(), ctx, _GATE_SPECS) is False


def test_stop_gate_satisfied_by_verification_tool():
    # A certified witness is verification-grade evidence -> stop is allowed, never
    # forced through the ML significance path (the S0 math-discovery fix).
    ctx = _make_ctx(tool_names_run=["extremal_set_search"], steps_taken=3, min_steps=3)
    assert _stop_needs_evidence(_stop(), ctx, _GATE_SPECS) is False


def test_stop_gate_not_enforced_before_min_steps():
    ctx = _make_ctx(tool_names_run=[], steps_taken=1, min_steps=3)
    assert _stop_needs_evidence(_stop(), ctx, _GATE_SPECS) is False


def test_stop_gate_only_applies_to_stop_action():
    ctx = _make_ctx(tool_names_run=[], steps_taken=3, min_steps=3)
    assert _stop_needs_evidence(AgentAction(action_type="tool"), ctx, _GATE_SPECS) is False


def test_stop_gate_is_spec_driven_not_name_driven():
    # If the specs do NOT mark the tool verification_capable, running it does not
    # satisfy the gate — the concept is spec-driven, not a hardcoded tool name.
    plain_specs = [{"name": "extremal_set_search", "example": {"params": {"n": 7}}}]
    ctx = _make_ctx(tool_names_run=["extremal_set_search"], steps_taken=3, min_steps=3)
    assert _stop_needs_evidence(_stop(), ctx, plain_specs) is True
