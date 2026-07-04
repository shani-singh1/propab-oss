"""Belief trend promotion for math_combinatorics (fixes.md A1)."""
from __future__ import annotations

import json

from propab.belief_promotion import (
    apply_trend_promotion_to_beliefs,
    is_consistent_trend,
    try_trend_promotion,
)
from propab.belief_state import CampaignBeliefState, BeliefObject
from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin


def _confirmed_node(nid: str, n: int, ratio: float) -> dict:
    return {
        "id": nid,
        "verdict": "confirmed",
        "text": f"Greedy Sidon at n={n} ratio {ratio}",
        "finding": {
            "metric_name": "sidon_ratio_to_sqrt_n",
            "metric_value": ratio,
            "n": n,
            "sweep": [{"n": n, "ratio_to_sqrt_n": ratio}],
        },
    }


def test_trend_promotion_from_json_evidence_summary() -> None:
    """Live campaigns store metrics in evidence_summary JSON string, not finding."""
    nodes = {
        "a": {
            "id": "a",
            "verdict": "confirmed",
            "text": "Greedy Sidon sweep",
            "finding": {"claim": "D(n) decreases"},
            "evidence_summary": json.dumps({
                "metric_name": "sidon_ratio_to_sqrt_n",
                "metric_value": 0.939,
                "max_n": 500,
                "sweep": [{"n": 500, "ratio_to_sqrt_n": 0.939}],
            }),
        },
        "b": {
            "id": "b",
            "verdict": "confirmed",
            "text": "Greedy Sidon sweep",
            "finding": {"claim": "D(n) decreases"},
            "evidence_summary": json.dumps({
                "metric_name": "sidon_ratio_to_sqrt_n",
                "metric_value": 0.885,
                "max_n": 1000,
                "sweep": [{"n": 1000, "ratio_to_sqrt_n": 0.885}],
            }),
        },
        "c": {
            "id": "c",
            "verdict": "confirmed",
            "text": "Greedy Sidon sweep",
            "finding": {"claim": "D(n) decreases"},
            "evidence_summary": json.dumps({
                "metric_name": "sidon_ratio_to_sqrt_n",
                "metric_value": 0.900,
                "max_n": 10000,
                "sweep": [
                    {"n": 100, "ratio_to_sqrt_n": 1.0},
                    {"n": 10000, "ratio_to_sqrt_n": 0.827},
                ],
            }),
        },
    }
    threshold = MathCombinatoricsPlugin().belief_promotion_threshold()
    ids = try_trend_promotion(
        "greedy F(n)/sqrt(n) is monotonically decreasing for n in [500, 10000]",
        nodes,
        threshold,
    )
    assert ids is not None
    assert len(ids) >= 3
    nodes = {
        "a": _confirmed_node("a", 500, 0.939),
        "b": _confirmed_node("b", 1000, 0.885),
        "c": _confirmed_node("c", 2000, 0.827),
    }
    threshold = MathCombinatoricsPlugin().belief_promotion_threshold()
    ids = try_trend_promotion(
        "greedy F(n)/sqrt(n) is monotonically decreasing for n in [500, 10000]",
        nodes,
        threshold,
    )
    assert ids is not None
    assert len(ids) >= 3


def test_apply_synthesis_preserves_trend_promoted_beliefs() -> None:
    state = CampaignBeliefState()
    state.active_beliefs.append(BeliefObject(
        statement="carried promoted belief",
        confidence="weak",
        supporting_nodes=["a", "b", "c"],
    ))
    state.apply_synthesis_beliefs([
        {"statement": "new ungrounded belief", "confidence": "weak", "status": "active"},
    ], allow_ungrounded=True)
    assert any(b.statement == "carried promoted belief" for b in state.active_beliefs)
    assert len(state.active_beliefs[0].supporting_nodes) == 3


def test_refresh_active_belief_grows_support() -> None:
    from propab.belief_promotion import refresh_active_belief_trend_support

    state = CampaignBeliefState()
    state.active_beliefs.append(BeliefObject(
        statement="greedy F(n)/sqrt(n) is monotonically decreasing",
        confidence="weak",
        supporting_nodes=["a", "b", "c"],
    ))
    nodes = {
        "a": _confirmed_node("a", 500, 0.939),
        "b": _confirmed_node("b", 1000, 0.885),
        "c": _confirmed_node("c", 2000, 0.827),
        "d": _confirmed_node("d", 5000, 0.721),
        "e": _confirmed_node("e", 10000, 0.670),
    }
    thresh = MathCombinatoricsPlugin().belief_promotion_threshold()
    n = refresh_active_belief_trend_support(state, nodes, thresh)
    assert n == 1
    assert len(state.active_beliefs[0].supporting_nodes) >= 4


def test_trend_promotion_three_monotone_nodes() -> None:
    nodes = {
        "a": _confirmed_node("a", 500, 0.939),
        "b": _confirmed_node("b", 1000, 0.885),
    }
    threshold = MathCombinatoricsPlugin().belief_promotion_threshold()
    assert try_trend_promotion("monotonically decreasing Sidon ratio", nodes, threshold) is None


def test_non_monotonic_trend_rejected() -> None:
    assert is_consistent_trend([0.939, 0.950, 0.827], decreasing=True) is False


def test_apply_trend_promotion_moves_to_active() -> None:
    state = CampaignBeliefState()
    state.proposed_ungrounded_beliefs.append(BeliefObject(
        statement="greedy F(n)/sqrt(n) is monotonically decreasing",
        confidence="weak",
    ))
    nodes = {
        "a": _confirmed_node("a", 500, 0.939),
        "b": _confirmed_node("b", 1000, 0.885),
        "c": _confirmed_node("c", 2000, 0.827),
    }
    n = apply_trend_promotion_to_beliefs(
        state, nodes, MathCombinatoricsPlugin().belief_promotion_threshold(),
    )
    assert n == 1
    assert len(state.active_beliefs) == 1
    assert len(state.active_beliefs[0].supporting_nodes) >= 3
