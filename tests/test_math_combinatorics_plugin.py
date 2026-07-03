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


def test_cap_set_claim_at_least_300_refuted(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": "The cap set size a_3(7) is at least 300 in F_3^7",
        "test_methodology": "best-known cap-set table lookup",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["cap_set_size"])
    assert evidence["cap_set_size"] == 236
    assert evidence["verified_false_steps"] >= 1
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"


def test_bose_chowla_produces_valid_sidon_set() -> None:
    from propab.domain_modules.math_combinatorics.constructors import bose_chowla_sidon, is_sidon_set

    for q in (3, 5, 7):
        s = bose_chowla_sidon(q)
        assert len(s) == q
        assert is_sidon_set(s)


def test_greedy_vs_bose_chowla_comparison(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "For n in {500, 1000, 2000}, Bose-Chowla Sidon construction exceeds "
            "greedy construction in size"
        ),
        "test_methodology": "greedy vs Bose-Chowla multi-n sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    assert evidence.get("comparison_sweep")
    assert evidence.get("max_n", 0) >= 500
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict in ("confirmed", "refuted", "inconclusive")


def test_methodology_rejects_sat_solver(plugin: MathCombinatoricsPlugin) -> None:
    bad = (
        "Find maximum cap set in F_3^7 using SAT solver encoding of AP-free constraints"
    )
    assert plugin.hypothesis_on_topic(bad, methodology="SAT-based cap-set search") is False


def test_methodology_accepts_scope_json_without_greedy_keyword(
    plugin: MathCombinatoricsPlugin,
) -> None:
    text = (
        "For n in {100, 500, 1000}, ratio F(n)/sqrt(n) converges below 1.0 for maximum Sidon sets"
    )
    scope_json = '{"methodology": "scoped_verification", "population": "integers 1..n"}'
    assert plugin.hypothesis_on_topic(text, methodology=scope_json) is True


def test_methodology_accepts_greedy_sweep(plugin: MathCombinatoricsPlugin) -> None:
    good = (
        "For n in {100, 500, 1000}, ratio F(n)/sqrt(n) converges below 1.0 for greedy Sidon sets"
    )
    assert plugin.hypothesis_on_topic(good, methodology="multi-n greedy Sidon sweep") is True


def test_methodology_rejects_tabu_search_at_synthesis(plugin: MathCombinatoricsPlugin) -> None:
    text = "For n=5000 find maximum Sidon set density in {1,...,n}"
    assert plugin.hypothesis_on_topic(text, methodology="tabu search with restarts") is False


def test_methodology_rejects_mcmc(plugin: MathCombinatoricsPlugin) -> None:
    text = "Maximum Sidon sets for n in {2000, 5000, 10000}"
    assert plugin.hypothesis_on_topic(
        text, methodology="Markov Chain Monte Carlo sampling of Sidon sets",
    ) is False


def test_interval_band_claim_refuted_not_confirmed(plugin: MathCombinatoricsPlugin) -> None:
    """da855131 false positive: [0.90, 0.95] band vs actual greedy ratios up to 1.20."""
    hypothesis = {
        "statement": (
            "For n in {100, 200, 500, 1000, 2000, 5000}, the ratio F(n)/sqrt(n) for maximum "
            "Sidon sets stays within the interval [0.90, 0.95] without showing a monotonic "
            "trend toward 1.0."
        ),
        "test_methodology": "multi-n greedy Sidon sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"
    assert evidence.get("verified_false_steps", 0) >= 1


def test_cap_set_hypothesis_routes_to_cap_metric(plugin: MathCombinatoricsPlugin) -> None:
    """Cap-set claims must not confirm on Sidon-ratio evidence."""
    hypothesis = {
        "statement": (
            "In F_3^n for n in {3, 4, 5, 6, 7}, the sequence c_n = |A_max(n)|^(1/n) is strictly "
            "increasing (c_n < c_{n+1}), with c_7 > 2.25, narrowing the gap to the CLP bound (2.756)."
        ),
        "test_methodology": "cap-set best-known table sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["cap_set_size"])
    assert str(evidence.get("metric_name") or "").startswith("cap_set")
    assert evidence.get("metric_mismatch") is not True


def test_structural_poisson_claim_refuted(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "The normalized gaps between consecutive elements in maximum Sidon sets for n >= 1000 "
            "follow a Poisson distribution (exp(-x))."
        ),
        "test_methodology": "multi-n greedy Sidon sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"


def test_decile_density_claim_refuted(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "There is no statistically significant difference in the local density of elements "
            "between the first decile [1, 0.1n] and the last decile [0.9n, n] of a maximum Sidon "
            "set for n >= 2000."
        ),
        "test_methodology": "scoped_verification",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"


def test_asymptotic_below_one_refuted_when_ratios_exceed_band(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "For n in {100, 200, 500, 1000, 2000, 5000, 10000}, the ratio F(n)/sqrt(n) converges "
            "to a constant below 1.0 as n grows"
        ),
        "test_methodology": "multi-n greedy Sidon sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"


def test_bose_chowla_exceeds_greedy_refuted(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": (
            "For n in {500, 1000, 2000}, Bose-Chowla Sidon construction exceeds "
            "greedy construction in size"
        ),
        "test_methodology": "greedy vs Bose-Chowla multi-n sweep",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["sidon_set_density"])
    verdict, _, _ = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "refuted"
