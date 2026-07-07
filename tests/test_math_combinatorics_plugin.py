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


# --- Real cap-set computation (F_3^n): validity, witness, honest sizes --------


def test_cap_third_point_completes_line() -> None:
    from propab.domain_modules.math_combinatorics.verifier import cap_third_point

    # In F_3, a line through distinct a, b has third point c with a+b+c=0 mod 3.
    a, b = (0, 0), (1, 2)
    c = cap_third_point(a, b)
    assert all((a[k] + b[k] + c[k]) % 3 == 0 for k in range(2))


def test_is_valid_cap_accepts_known_cap_and_rejects_collinear() -> None:
    from propab.domain_modules.math_combinatorics.verifier import is_valid_cap

    # {(0,0), (0,1), (1,0)} is a cap in F_3^2 (no third point completes a line).
    ok, _ = is_valid_cap([(0, 0), (0, 1), (1, 0)], 2)
    assert ok is True

    # {(0,0), (1,1), (2,2)} is a full line in F_3^2 -> NOT a cap.
    bad, detail = is_valid_cap([(0, 0), (1, 1), (2, 2)], 2)
    assert bad is False
    assert detail["reason"] == "collinear_triple"

    # Duplicate points are rejected.
    dup, ddetail = is_valid_cap([(0, 0), (0, 0)], 2)
    assert dup is False
    assert ddetail["reason"] == "duplicate_point"


def test_compute_cap_set_returns_validated_cap_with_matching_size() -> None:
    from propab.domain_modules.math_combinatorics.verifier import compute_cap_set, is_valid_cap

    for n in (2, 3, 6, 7):
        r = compute_cap_set(n)
        # Reported size must equal the actual computed set size, independently re-checked.
        assert r["cap_valid"] is True
        assert r["cap_set_size"] == r["computed_size"]
        pts = r["witness"].get("cap_points")
        if pts is not None:
            # Full witness: re-validate from scratch and confirm the reported size.
            revalid, _ = is_valid_cap([tuple(p) for p in pts], n)
            assert revalid is True
            assert len(pts) == r["cap_set_size"]
        assert r["construction_source"] == "computed"


def test_compute_cap_set_reproduces_f3_4_max_20() -> None:
    """Near-exhaustive branch-and-bound reproduces the known max cap in F_3^4 = 20."""
    from propab.domain_modules.math_combinatorics.verifier import (
        compute_cap_set,
        is_valid_cap,
    )

    r = compute_cap_set(4)
    assert r["cap_set_size"] == 20
    assert r["cap_valid"] is True
    assert r["vs_best_known"] == "matches_best_known"
    pts = [tuple(p) for p in r["witness"]["cap_points"]]
    ok, _ = is_valid_cap(pts, 4)
    assert ok is True
    assert len(pts) == 20


def test_max_cap_exhaustive_optimal_small_dim() -> None:
    """Exhaustive search proves the maximum cap in F_3^2 is 4 (complete search)."""
    from propab.domain_modules.math_combinatorics.verifier import (
        is_valid_cap,
        max_cap_exhaustive,
    )

    cap, complete = max_cap_exhaustive(2, time_limit=5.0)
    assert complete is True  # whole tree searched -> provably optimal
    assert len(cap) == 4
    ok, _ = is_valid_cap(cap, 2)
    assert ok is True


def test_product_construction_yields_valid_larger_cap() -> None:
    from propab.domain_modules.math_combinatorics.verifier import (
        _base_cap,
        _cap_product,
        is_valid_cap,
    )

    a = _base_cap(2)
    b = _base_cap(3)
    prod = _cap_product(a, b)
    assert len(prod) == len(a) * len(b)
    ok, _ = is_valid_cap(prod, 5)
    assert ok is True


def test_compute_cap_set_reports_honest_gap_below_best_known() -> None:
    from propab.domain_modules.math_combinatorics.verifier import compute_cap_set

    # F_3^8 best-known is 512; our product construction is genuinely smaller.
    r = compute_cap_set(8)
    assert r["best_known_size"] == 512
    assert r["cap_set_size"] < 512
    assert r["vs_best_known"] == "below_best_known"
    assert r["gap_to_best_known"] == 512 - r["cap_set_size"]
    assert r["cap_valid"] is True


def test_large_cap_uses_certificate_witness_with_validated_factors() -> None:
    from propab.domain_modules.math_combinatorics.verifier import compute_cap_set

    r = compute_cap_set(10)  # 3^10 = 59049 points; full set too large to list.
    assert r["cap_valid"] is True
    assert r["witness"]["kind"] == "certificate"
    # Every base factor of the product is fully validated.
    factors = r["witness"]["factor_validity"]
    assert factors and all(f["valid"] for f in factors)
    # Certificate reports a checkable size and sample.
    assert r["cap_set_size"] == r["computed_size"]
    assert len(r["witness"]["sample_points"]) > 0


def test_cap_compute_path_not_flagged_rediscovery(plugin: MathCombinatoricsPlugin) -> None:
    hypothesis = {
        "statement": "Construct a cap set in F_3^6 and report its size.",
        "test_methodology": "greedy construction",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["cap_set_size"])
    assert evidence["construction_source"] == "computed"
    assert evidence.get("trivial_rediscovery") is False
    assert evidence["cap_valid"] is True
    assert evidence["cap_set_size"] > 0


def test_cap_table_lookup_still_flagged_rediscovery(plugin: MathCombinatoricsPlugin) -> None:
    """DISC2 guard: a best-known TABLE lookup must stay trivial_rediscovery."""
    hypothesis = {
        "statement": "Report the best-known cap set size in F_3^6.",
        "test_methodology": "best-known table lookup",
    }
    evidence = run_combinatorics_experiment(hypothesis, ["cap_set_size"])
    assert evidence["cap_set_size"] == 112  # tabulated best-known value
    assert evidence.get("trivial_rediscovery") is True
    assert evidence["construction_source"] == "best_known_table"
    verdict, _, confidence = plugin.classify_verdict(hypothesis["statement"], evidence)
    assert verdict == "inconclusive"
    assert confidence <= 0.5


def test_cap_size_never_reported_from_table_as_computed() -> None:
    """The compute path must not surface a CAP_SET_BEST_KNOWN value as 'computed'."""
    from propab.domain_modules.math_combinatorics.constructors import CAP_SET_BEST_KNOWN
    from propab.domain_modules.math_combinatorics.verifier import compute_cap_set

    # For a dim where best-known > our real construction, sizes must differ.
    for n in (6, 7, 8):
        r = compute_cap_set(n)
        assert r["construction_source"] == "computed"
        assert r["cap_set_size"] != CAP_SET_BEST_KNOWN[n]
        assert r["cap_valid"] is True
