"""Tests for the coding_theory domain plugin (binary linear error-correcting codes).

Honesty guarantees under test:
- A constructed code's computed minimum distance is independently recomputed and
  matches the emitted witness.
- The [7,4,3] Hamming code is reproduced exactly (min distance 3).
- A table-lookup path is flagged trivial_rediscovery (never a discovery).
- An honest below/meets-known result is reported honestly, not oversold.
- Evidence routes as "deterministic" only with a real proof method + witness.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from propab.domain_modules.coding_theory.constructors import (
    best_known_distance,
    compute_min_distance,
    extended_hamming_code,
    hamming_code,
    is_table_lookup_evidence,
    parse_code_params,
    recompute_distance_of_witness,
    reed_muller_rm1,
    repetition_code,
    simplex_code,
    trivial_rediscovery,
)
from propab.domain_modules.coding_theory.plugin import CodingTheoryPlugin
from propab.domain_modules.coding_theory.verifier import run_coding_experiment
from propab.verdict_pipeline import classify_evidence_type, run_verdict_pipeline


@pytest.fixture
def plugin() -> CodingTheoryPlugin:
    return CodingTheoryPlugin()


# --------------------------------------------------------------------------- #
# Contract basics
# --------------------------------------------------------------------------- #
def test_preflight_passes(plugin: CodingTheoryPlugin) -> None:
    result = plugin.preflight()
    assert result.passed, f"Preflight failed: {result.reason}"


def test_available_features_nonempty(plugin: CodingTheoryPlugin) -> None:
    features = plugin.available_features()
    assert len(features) > 0
    assert "code_minimum_distance" in features


def test_uses_synthetic_data_is_false(plugin: CodingTheoryPlugin) -> None:
    assert plugin.uses_synthetic_data() is False


def test_confirmation_criteria_deterministic(plugin: CodingTheoryPlugin) -> None:
    criteria = plugin.confirmation_criteria()
    assert criteria["verification_type"] == "deterministic"
    assert criteria["requires_holdout"] is False
    assert criteria["min_metric_steps_for_confirm"] >= 1


def test_domain_registration() -> None:
    from propab.domain_modules.registry import get_domain_plugin

    registered = get_domain_plugin("coding_theory")
    assert registered is not None
    assert registered.domain_id == "coding_theory"


def test_resolve_by_question() -> None:
    from propab.domain_modules.registry import resolve_domain_plugin

    resolved = resolve_domain_plugin(
        question="Build a binary linear code and compute its minimum distance"
    )
    assert resolved is not None
    assert resolved.domain_id == "coding_theory"


def test_literature_profile_has_real_fields(plugin: CodingTheoryPlugin) -> None:
    prof = plugin.literature_profile()
    assert prof["search_terms"]
    assert prof["novelty_criteria"]
    assert prof["tabulation_sources"]
    assert prof["open_problem_sources"]
    # codetables.de must be referenced as the tabulation source.
    blob = str(prof["tabulation_sources"]).lower()
    assert "codetables" in blob or "grassl" in blob or "brouwer" in blob


# --------------------------------------------------------------------------- #
# Real computation: witness correctness
# --------------------------------------------------------------------------- #
def test_hamming_7_4_min_distance_is_3() -> None:
    g = hamming_code(3)
    assert g.shape == (4, 7)
    dist = compute_min_distance(g)
    assert dist["min_distance"] == 3
    assert dist["generator_valid"] is True
    assert dist["witness_codeword"] is not None


def test_witness_recomputation_matches_claimed_distance() -> None:
    """Independently recompute the min distance and confirm it equals the witness weight."""
    g = hamming_code(3)
    dist = compute_min_distance(g)
    recheck = recompute_distance_of_witness(dist["generator_matrix"], dist["witness_message"])
    assert recheck["ok"] is True
    assert recheck["weight"] == dist["min_distance"]
    assert recheck["recomputed_codeword"] == dist["witness_codeword"]


def test_witness_weight_equals_brute_force_over_all_codewords() -> None:
    """Fully independent recomputation: brute-force min weight over 2^k-1 codewords."""
    g = hamming_code(3).astype(np.int64)
    k, n = g.shape
    brute_min = min(
        int(((np.array(msg) @ g) % 2).sum())
        for msg in itertools.product((0, 1), repeat=k)
        if any(msg)
    )
    dist = compute_min_distance(g)
    assert dist["min_distance"] == brute_min == 3


def test_extended_hamming_8_4_min_distance_is_4() -> None:
    g = extended_hamming_code(3)
    assert g.shape == (4, 8)
    assert compute_min_distance(g)["min_distance"] == 4


def test_simplex_code_all_nonzero_weights_equal() -> None:
    """Simplex [7,3,4]: every nonzero codeword has weight exactly 2^(r-1)=4."""
    g = simplex_code(3).astype(np.int64)
    k, n = g.shape
    weights = {
        int(((np.array(msg) @ g) % 2).sum())
        for msg in itertools.product((0, 1), repeat=k)
        if any(msg)
    }
    assert weights == {4}
    assert compute_min_distance(g)["min_distance"] == 4


def test_repetition_code_distance_equals_n() -> None:
    for n in (3, 5, 6):
        assert compute_min_distance(repetition_code(n))["min_distance"] == n


def test_reed_muller_rm1_distance() -> None:
    """RM(1, m) is [2^m, m+1, 2^(m-1)]. m=3 -> [8,4,4]."""
    g = reed_muller_rm1(3)
    assert g.shape == (4, 8)
    assert compute_min_distance(g)["min_distance"] == 4


def test_invalid_generator_reports_no_distance() -> None:
    dist = compute_min_distance([[1, 1, 0], [1, 1, 0]])  # rank 1, k=2 -> dependent
    assert dist["generator_valid"] is False
    assert dist["min_distance"] is None


# --------------------------------------------------------------------------- #
# Rediscovery rejection
# --------------------------------------------------------------------------- #
def test_hamming_reproduces_known_and_is_rediscovery(plugin: CodingTheoryPlugin) -> None:
    hyp = {
        "statement": "Construct the [7,4,3] Hamming code and verify its minimum distance",
        "test_methodology": "hamming exhaustive enumeration",
    }
    ev = run_coding_experiment(hyp)
    assert ev["computed_min_distance"] == 3
    assert ev["best_known_distance"] == 3
    assert ev["witness_recheck_ok"] is True
    # Meets (does not beat) best-known => rediscovery, not a discovery.
    assert ev["trivial_rediscovery"] is True
    assert ev["discovery_worthy"] is False
    verdict, _, _ = plugin.classify_verdict(hyp["statement"], ev)
    assert verdict == "inconclusive"


def test_table_lookup_evidence_is_flagged_rediscovery() -> None:
    table_ev = {"construction_source": "best_known_table", "min_distance": 3, "n": 7, "k": 4}
    assert is_table_lookup_evidence(table_ev) is True
    assert trivial_rediscovery(table_ev, 7, 4, 3) is True


def test_distance_without_witness_is_table_lookup() -> None:
    """A distance with no achieving witness codeword must not count as a real computation."""
    ev = {"min_distance": 5, "witness_codeword": None}
    assert is_table_lookup_evidence(ev) is True


def test_meets_known_bound_is_rediscovery_not_discovery(plugin: CodingTheoryPlugin) -> None:
    hyp = {"statement": "Construct a repetition code with n=6", "test_methodology": "repetition"}
    ev = run_coding_experiment(hyp)
    assert ev["computed_min_distance"] == best_known_distance(6, 1) == 6
    assert ev["trivial_rediscovery"] is True
    assert ev["discovery_worthy"] is False


# --------------------------------------------------------------------------- #
# Honest reporting (no overclaiming)
# --------------------------------------------------------------------------- #
def test_below_known_result_reported_honestly_not_oversold(plugin: CodingTheoryPlugin) -> None:
    """A random systematic [11,4] code has small d; must not be sold as a discovery."""
    hyp = {"statement": "A binary linear [11,4] code", "test_methodology": "generator matrix"}
    ev = run_coding_experiment(hyp)
    known = best_known_distance(11, 4)
    assert ev["computed_min_distance"] < known
    assert ev["discovery_worthy"] is False
    verdict, _, _ = plugin.classify_verdict(hyp["statement"], ev)
    assert verdict == "inconclusive"


def test_claim_exceeding_true_distance_is_refuted(plugin: CodingTheoryPlugin) -> None:
    """Claiming d>=4 for the [7,4] Hamming code (true d=3) is refuted by real computation."""
    hyp = {
        "statement": "The [7,4] Hamming code has minimum distance d >= 4",
        "test_methodology": "hamming",
    }
    ev = run_coding_experiment(hyp)
    assert ev["computed_min_distance"] == 3
    assert ev["verified_false_steps"] >= 1
    verdict, _, _ = plugin.classify_verdict(hyp["statement"], ev)
    assert verdict == "refuted"


def test_supported_claim_meeting_known_is_not_confirmed(plugin: CodingTheoryPlugin) -> None:
    """Claim d>=3 for [7,4] is SUPPORTED but only meets known => rediscovery, not confirmed."""
    hyp = {
        "statement": "The [7,4,3] Hamming code has minimum distance d >= 3",
        "test_methodology": "hamming",
    }
    ev = run_coding_experiment(hyp)
    assert ev["verified_true_steps"] == 1
    assert ev["discovery_worthy"] is False
    assert ev["trivial_rediscovery"] is True
    verdict, _, _ = plugin.classify_verdict(hyp["statement"], ev)
    assert verdict == "inconclusive"


# --------------------------------------------------------------------------- #
# Evidence type / verdict pipeline routing
# --------------------------------------------------------------------------- #
def test_evidence_routes_as_deterministic() -> None:
    hyp = {"statement": "Construct the [7,4,3] Hamming code", "test_methodology": "hamming"}
    ev = run_coding_experiment(hyp)
    assert ev["deterministic"] is True
    assert ev["verification_method"] == "exhaustive_enumeration"
    assert classify_evidence_type(ev) == "deterministic"


def test_discovery_worthy_evidence_confirms_through_pipeline() -> None:
    """A beats-known result routes to 'confirmed' through the full verdict pipeline."""
    ev = {
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "deterministic": True,
        "discovery_worthy": True,
        "trivial_rediscovery": False,
        "verification_method": "exhaustive_enumeration",
        "witness_recheck_ok": True,
        "metric_value": 5,
    }
    verdict, confidence, _ = run_verdict_pipeline(ev, campaign_context={"min_metric_steps": 1})
    assert verdict == "confirmed"
    assert confidence >= 0.90


def test_failed_witness_recheck_is_never_confirmed(plugin: CodingTheoryPlugin) -> None:
    """If the witness fails independent recomputation, verdict must not be confirmed."""
    ev = {
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "discovery_worthy": True,
        "witness_recheck_ok": False,
        "notes": "witness failed",
    }
    verdict, _, _ = plugin.classify_verdict("some [n,k,d] code", ev)
    assert verdict == "inconclusive"


# --------------------------------------------------------------------------- #
# Topic gating
# --------------------------------------------------------------------------- #
def test_on_topic_accepts_coding_claim(plugin: CodingTheoryPlugin) -> None:
    good = "Construct a binary linear [15,7,5] BCH code and compute its minimum distance"
    assert plugin.hypothesis_on_topic(good) is True


def test_off_topic_rejects_path_garbage(plugin: CodingTheoryPlugin) -> None:
    bad = "Search the system PATH for binaries related to result submission (propab-submit)"
    assert plugin.hypothesis_on_topic(bad) is False


def test_rejects_unimplemented_sat_solver(plugin: CodingTheoryPlugin) -> None:
    bad = "Find a good binary linear code minimum distance using a SAT solver encoding"
    assert plugin.hypothesis_on_topic(bad, methodology="SAT-based search") is False


def test_rejects_neural_network_methodology(plugin: CodingTheoryPlugin) -> None:
    text = "Improve the minimum distance of a binary linear [16,8] code"
    assert plugin.hypothesis_on_topic(text, methodology="neural network optimizer") is False


def test_accepts_scope_json_methodology(plugin: CodingTheoryPlugin) -> None:
    text = "Verify the [7,4,3] Hamming code minimum distance over GF(2)"
    scope = '{"methodology": "scoped_verification", "population": "binary linear codes"}'
    assert plugin.hypothesis_on_topic(text, methodology=scope) is True


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_code_params_nkd() -> None:
    p = parse_code_params("Construct the [23, 12, 7] binary Golay code")
    assert p["n"] == 23 and p["k"] == 12 and p["d"] == 7


def test_parse_code_params_nk_only() -> None:
    p = parse_code_params("A binary linear [15, 11] code")
    assert p["n"] == 15 and p["k"] == 11 and p["d"] is None


# =========================================================================== #
# v1: widened reference table integrity, correctness at scale, robustness
# =========================================================================== #


def _griesmer_length(k: int, d: int) -> int:
    """Griesmer bound: minimal n for a binary [n,k,d] code = sum ceil(d/2^i)."""
    return sum(-(-d // (2 ** i)) for i in range(k))


def test_coding_best_known_table_is_griesmer_consistent() -> None:
    """Every widened Brouwer/Grassl entry must satisfy the Griesmer bound (a
    NECESSARY condition) and basic [n,k,d] ranges. A table entry that violated
    Griesmer would be an impossible distance and would poison rediscovery rejection."""
    from propab.domain_modules.coding_theory.constructors import BEST_KNOWN_TABLE

    assert len(BEST_KNOWN_TABLE) >= 120  # widened well beyond the original ~90
    for (n, k), d in BEST_KNOWN_TABLE.items():
        assert 1 <= k <= n, f"[{n},{k}] out of range"
        assert 1 <= d <= n, f"[{n},{k},{d}] distance out of range"
        assert _griesmer_length(k, d) <= n, f"[{n},{k},{d}] violates Griesmer bound"


def test_coding_constructible_optimal_families_match_table_exactly() -> None:
    """Correctness at scale: for every code this module can BUILD, the independently
    computed minimum distance must equal the tabulated best-known d — never exceed
    it (a construction that spuriously 'beats' an optimal family would be a bug and a
    false discovery)."""
    from propab.domain_modules.coding_theory.constructors import (
        best_known_distance,
        compute_min_distance,
        extended_hamming_code,
        hamming_code,
        parity_check_code,
        reed_muller_rm1,
        repetition_code,
        simplex_code,
    )

    cases: list[tuple[np.ndarray, int, int]] = []
    for r in (2, 3, 4):
        cases.append((hamming_code(r), 2 ** r - 1, 2 ** r - 1 - r))
        cases.append((extended_hamming_code(r), 2 ** r, 2 ** r - 1 - r))
        cases.append((simplex_code(r), 2 ** r - 1, r))
    for m in (2, 3, 4):
        cases.append((reed_muller_rm1(m), 2 ** m, m + 1))
    for n in (2, 5, 7, 10, 13, 16):
        cases.append((repetition_code(n), n, 1))
    for k in (2, 4, 7, 11, 14):
        cases.append((parity_check_code(k), k + 1, k))

    for g, n, k in cases:
        d = compute_min_distance(g)["min_distance"]
        bk = best_known_distance(n, k)
        assert bk is not None, f"[{n},{k}] missing from widened table"
        assert d == bk, f"[{n},{k}] computed d={d} but table={bk}"


def test_coding_new_table_anchors_present_and_correct() -> None:
    """Spot-check a handful of the newly-added, theory-anchored entries."""
    from propab.domain_modules.coding_theory.constructors import best_known_distance

    # RM(1,4) = [16,5,8]; simplex r=4 = [15,4,8]; ext-Hamming r=4 = [16,11,4].
    assert best_known_distance(16, 5) == 8
    assert best_known_distance(15, 4) == 8
    assert best_known_distance(16, 11) == 4
    # New n=13/14 rows (Brouwer/Grassl codetables.de).
    assert best_known_distance(13, 4) == 6
    assert best_known_distance(14, 5) == 6
    assert best_known_distance(13, 2) == 8
    assert best_known_distance(14, 2) == 9


def test_coding_wider_table_rejects_more_rediscoveries() -> None:
    """A wider correct table lets the engine flag more meets/below-known results as
    rediscoveries. A random [14,5] code meeting-or-below the known d=6 must not be a
    discovery, and a [13,4] code likewise checks against the newly-added entry."""
    from propab.domain_modules.coding_theory.constructors import trivial_rediscovery

    # Newly-tabulated [14,5] has best-known d=6; any computed d<=6 is a rediscovery.
    ev = {"verification_method": "exhaustive_enumeration", "witness_codeword": [1]}
    assert trivial_rediscovery(ev, 14, 5, 4) is True   # below known -> rediscovery
    assert trivial_rediscovery(ev, 14, 5, 6) is True   # meets known -> rediscovery
    assert trivial_rediscovery(ev, 14, 5, 7) is False  # would beat known -> not


def test_coding_max_exhaustive_k_is_production_bounded() -> None:
    """Robustness: the exhaustive-enumeration cap keeps the worst case feasible. At
    the cap, enumeration completes quickly; beyond it we refuse to certify."""
    import time

    from propab.domain_modules.coding_theory.constructors import (
        MAX_EXHAUSTIVE_K,
        compute_min_distance,
        random_generator,
    )

    assert MAX_EXHAUSTIVE_K <= 16, "cap must bound 2^k enumeration for production"
    # At the cap, a real computation finishes well under a conservative ceiling.
    g = random_generator(MAX_EXHAUSTIVE_K + 8, MAX_EXHAUSTIVE_K, seed=11)
    start = time.time()
    dist = compute_min_distance(g)
    assert time.time() - start < 10.0
    assert dist["enumeration_complete"] is True
    # Above the cap, refuse to certify a distance (no silent lie).
    g_big = random_generator(MAX_EXHAUSTIVE_K + 10, MAX_EXHAUSTIVE_K + 2, seed=11)
    over = compute_min_distance(g_big)
    assert over["min_distance"] is None
    assert over["enumeration_complete"] is False


def test_coding_computation_is_deterministic_under_fixed_seed() -> None:
    """A fixed seed yields byte-identical generator, distance, and witness."""
    from propab.domain_modules.coding_theory.constructors import (
        compute_min_distance,
        random_generator,
    )

    for k in (4, 8, 12):
        g1 = random_generator(k + 6, k, seed=99)
        g2 = random_generator(k + 6, k, seed=99)
        assert np.array_equal(g1, g2)
        d1 = compute_min_distance(g1)
        d2 = compute_min_distance(g2)
        assert d1["min_distance"] == d2["min_distance"]
        assert d1["witness_codeword"] == d2["witness_codeword"]


def test_coding_reproduces_known_hamming_7_4_3_optimum() -> None:
    """Reproduce a known optimum end-to-end: [7,4,3] Hamming, honestly labelled as a
    rediscovery (meets best-known d=3), with an independently re-checked witness."""
    hyp = {
        "statement": "Construct the [7,4,3] Hamming code and verify its minimum distance",
        "test_methodology": "hamming exhaustive enumeration",
    }
    ev = run_coding_experiment(hyp)
    assert ev["computed_min_distance"] == 3
    assert ev["best_known_distance"] == 3
    assert ev["witness_recheck_ok"] is True
    assert ev["trivial_rediscovery"] is True
    assert ev["discovery_worthy"] is False
