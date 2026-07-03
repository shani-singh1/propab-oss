"""Tests for the math_combinatorics domain plugin."""
from __future__ import annotations

import pytest

from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
from propab.domain_modules.math_combinatorics.verifier import run_combinatorics_experiment
from propab.verdict_pipeline import run_verdict_pipeline


@pytest.fixture
def plugin() -> MathCombinatoricsPlugin:
    return MathCombinatoricsPlugin()


def test_preflight_passes(plugin: MathCombinatoricsPlugin) -> None:
    result = plugin.preflight()
    assert result.passed, f"Preflight failed: {result.reason}"


def test_available_features_nonempty(plugin: MathCombinatoricsPlugin) -> None:
    features = plugin.available_features()
    assert len(features) > 0
    assert "sidon_set_density" in features


def test_confirmation_criteria_deterministic(plugin: MathCombinatoricsPlugin) -> None:
    criteria = plugin.confirmation_criteria()
    assert criteria["verification_type"] == "deterministic"
    assert criteria["requires_holdout"] is False
    assert criteria["min_metric_steps_for_confirm"] >= 1


def test_trivial_single_n_sidon_is_not_confirmed(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": "A Sidon set of size approximately sqrt(n) exists in {1,...,100}",
        "test_methodology": "greedy search",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence.get("trivial_rediscovery") is True
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "inconclusive"


def test_sidon_sweep_asymptotic_can_confirm(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "For n in {100, 200, 500, 1000, 2000}, the ratio F(n)/sqrt(n) converges "
            "to a constant below 1.0 as n grows"
        ),
        "test_methodology": "multi-n greedy Sidon sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence.get("sweep")
    assert evidence.get("max_n", 0) >= 500
    verdict, _, confidence = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict in ("confirmed", "refuted", "inconclusive")
    if verdict == "confirmed":
        assert evidence.get("discovery_worthy")
        assert confidence >= 0.90


def test_sidon_counterexample_refutes() -> None:
    hypothesis = {
        "statement": "No Sidon set of size 3 exists in {1,...,100}",
        "test_methodology": "exhaustive search",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence["verified_false_steps"] >= 1


def test_verdict_pipeline_respects_discovery_worthy() -> None:
    evidence = {
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "deterministic": True,
        "discovery_worthy": True,
        "verification_method": "combinatorial_computation",
        "metric_value": 0.98,
    }
    verdict, confidence, _ = run_verdict_pipeline(
        evidence,
        campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "confirmed"
    assert confidence >= 0.90


def test_hypothesis_on_topic_rejects_path_garbage(plugin: MathCombinatoricsPlugin) -> None:
    bad = (
        "Search the system PATH for binaries related to result submission "
        "(e.g., 'propab-submit', 'submit-results')"
    )
    assert plugin.hypothesis_on_topic(bad) is False


def test_hypothesis_on_topic_accepts_open_problem_claim(plugin: MathCombinatoricsPlugin) -> None:
    good = (
        "For n in {500, 1000, 2000}, the ratio F(n)/sqrt(n) converges to a constant "
        "below 1.0 for maximum Sidon sets"
    )
    assert plugin.hypothesis_on_topic(good) is True


def test_domain_registration() -> None:
    from propab.domain_modules.registry import get_domain_plugin

    registered = get_domain_plugin("math_combinatorics")
    assert registered is not None
    assert registered.domain_id == "math_combinatorics"
