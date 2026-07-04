"""Numerical seed extraction (fixes.md A2)."""
from __future__ import annotations

from propab.knowledge_graph import KnowledgeGraph
from propab.numerical_seeds import extract_math_combinatorics_seeds, format_seeds_for_question, classify_hypothesis_bucket
from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin


def _sweep_node(ratios: list[float], ns: list[int]) -> dict:
    sweep = [{"n": n, "ratio_to_sqrt_n": r} for n, r in zip(ns, ratios)]
    return {
        "id": "sweep-1",
        "verdict": "confirmed",
        "finding": {
            "metric_name": "sidon_ratio_to_sqrt_n",
            "metric_value": sum(ratios) / len(ratios),
            "sweep": sweep,
            "notes": f"Sidon sweep over n={ns}: ratios={ratios}",
        },
    }


def test_extract_monotone_and_thresholds() -> None:
    ns = [500, 1000, 2000, 5000, 10000]
    ratios = [0.939, 0.885, 0.827, 0.721, 0.670]
    seeds = extract_math_combinatorics_seeds([_sweep_node(ratios, ns)])
    types = {s["finding_type"] for s in seeds}
    assert "monotonic_trend" in types
    assert "threshold_crossing" in types
    crossing_70 = [s for s in seeds if s.get("parameters", {}).get("threshold") == 0.70]
    assert crossing_70
    assert crossing_70[0]["parameters"]["crossing_n"] == 10000


def test_extract_from_hypothesis_text_when_evidence_empty() -> None:
    node = {
        "id": "text-1",
        "verdict": "confirmed",
        "text": (
            "For n=50000, the greedy Sidon efficiency ratio F(n)/sqrt(n) will fall below 0.650"
        ),
    }
    node2 = {
        "id": "text-2",
        "verdict": "confirmed",
        "text": (
            "In the greedy Sidon construction, the efficiency ratio F(20000)/sqrt(20000) "
            "will be strictly greater than 0.661"
        ),
    }
    seeds = extract_math_combinatorics_seeds([node, node2])
    assert seeds
    assert any(s["finding_type"] == "threshold_crossing" for s in seeds)


def test_classify_sidon_not_cap_set_when_scope_boilerplate_present() -> None:
    text = (
        "Population: Greedy Sidon for n in {10000, 20000}. Claim: F(n)/sqrt(n) falls below 0.65\n"
        "Population: Integers {1,...,n} or vector space F_3^n\n"
        "Distribution: All admissible combinatorial structures in the domain"
    )
    bucket = classify_hypothesis_bucket(text, "greedy Sidon threshold sweep")
    assert bucket["problem_type"] == "sidon"


def test_claim_has_numeric_falsifier_rejects_pure_structural() -> None:
    from propab.numerical_seeds import claim_has_numeric_falsifier

    assert not claim_has_numeric_falsifier(
        "Population: Greedy Sidon sets. Claim: The ratio is strictly monotonic decreasing",
        "structural variance analysis",
    )
    assert claim_has_numeric_falsifier(
        "For n=50000, greedy Sidon F(n)/sqrt(n) falls below 0.60",
        "threshold sweep",
    )


def test_plugin_and_knowledge_graph_storage() -> None:
    plugin = MathCombinatoricsPlugin()
    ns = [500, 1000, 2000]
    ratios = [0.939, 0.885, 0.827]
    seeds = plugin.extract_numerical_seeds([_sweep_node(ratios, ns)])
    assert seeds
    g = KnowledgeGraph()
    g.store_numerical_seeds("math_combinatorics", "camp-1", seeds)
    loaded = g.get_numerical_seeds("math_combinatorics")
    assert loaded
    text = format_seeds_for_question(loaded)
    assert "prior campaigns" in text.lower()
