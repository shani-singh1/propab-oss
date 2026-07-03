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


def test_sidon_verification_finds_set() -> None:
    hypothesis = {
        "statement": "A Sidon set of size approximately sqrt(n) exists in {1,...,100}",
        "test_methodology": "exhaustive search for maximum Sidon set",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence["deterministic"] is True
    assert evidence["verified_true_steps"] >= 1
    assert evidence["metric_value"] > 0
    assert evidence["max_sidon_size"] >= 8


def test_sidon_counterexample_detection() -> None:
    hypothesis = {
        "statement": "No Sidon set of size 3 exists in {1,...,100}",
        "test_methodology": "exhaustive search",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence["verified_false_steps"] >= 1


def test_verdict_pipeline_confirms_deterministic_math() -> None:
    evidence = {
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "deterministic": True,
        "verification_method": "combinatorial_computation",
        "metric_value": 0.095,
    }
    verdict, confidence, reason = run_verdict_pipeline(
        evidence,
        campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "confirmed", f"Expected confirmed, got {verdict}: {reason}"
    assert confidence >= 0.90


def test_hypothesis_on_topic_rejects_path_garbage(plugin: MathCombinatoricsPlugin) -> None:
    bad = (
        "Search the system PATH for binaries related to result submission "
        "(e.g., 'propab-submit', 'submit-results')"
    )
    assert plugin.hypothesis_on_topic(bad) is False


def test_hypothesis_on_topic_accepts_sidon_claim(plugin: MathCombinatoricsPlugin) -> None:
    good = (
        "A Sidon set of size at least 1.05*sqrt(n) exists in {1,...,200} "
        "with greedy construction"
    )
    assert plugin.hypothesis_on_topic(good) is True


def test_domain_registration() -> None:
    from propab.domain_modules.registry import get_domain_plugin

    registered = get_domain_plugin("math_combinatorics")
    assert registered is not None
    assert registered.domain_id == "math_combinatorics"
