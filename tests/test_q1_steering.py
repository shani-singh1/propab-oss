"""Evidence-based threshold compounding (no deterministic hypothesis injection)."""
from __future__ import annotations

from propab.hypothesis_tree import HypothesisTree
from propab.numerical_seeds import (
    crossing_from_node,
    extract_math_combinatorics_seeds,
    refresh_lifetime_context_with_crossings,
)

Q1 = "[domain_profile:math_combinatorics] Where does greedy F(n)/sqrt(n) first fall below 0.60 for n in [10000, 50000]?"


def test_extract_from_threshold_search_structured() -> None:
    node = {
        "id": "ts-1",
        "verdict": "confirmed",
        "finding": {
            "metric_name": "sidon_ratio_to_sqrt_n",
            "metric_value": 0.598,
            "n": 35000,
            "threshold_search": {
                "target_ratio": 0.60,
                "crossing_n": 35000,
                "crossing_ratio": 0.598,
                "prev_n": 34500,
                "prev_ratio": 0.601,
            },
        },
    }
    assert crossing_from_node(node) is not None
    seeds = extract_math_combinatorics_seeds([node])
    assert any(s["finding_type"] == "threshold_crossing" for s in seeds)
    assert seeds[0]["parameters"]["crossing_n"] == 35000


def test_in_campaign_context_reports_crossing_fact_only() -> None:
    tree = HypothesisTree()
    nodes = tree.add_seeds([{
        "text": "Greedy Sidon threshold crossing claim",
        "test_methodology": "greedy Sidon threshold crossing sweep",
    }], generation=0)
    tree.update_node(
        nodes[0].id,
        "confirmed",
        0.95,
        'evidence={"threshold_search":{"target_ratio":0.60,"crossing_n":35000,"crossing_ratio":0.598,"prev_n":30000}};',
    )
    ctx = refresh_lifetime_context_with_crossings("", tree, Q1)
    assert "35000" in ctx
    assert "In-campaign finding:" in ctx
    assert "MUST narrow" not in ctx
    assert "Next experiments" not in ctx
