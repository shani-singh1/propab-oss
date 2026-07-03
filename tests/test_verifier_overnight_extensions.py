"""Verifier extensions A3, C1, C2, C3 (fixes.md)."""
from __future__ import annotations

import time

from propab.domain_modules.math_combinatorics.verifier import (
    _greedy_sidon_max_legacy,
    _greedy_sidon_max_optimized,
    find_threshold_crossing,
    run_combinatorics_experiment,
)


def test_optimized_matches_legacy_small_n() -> None:
    for n in (100, 500, 1000):
        assert len(_greedy_sidon_max_optimized(n)) == len(_greedy_sidon_max_legacy(n))


def test_threshold_crossing_near_8700_for_070() -> None:
    result = find_threshold_crossing(0.70, 5000, 10000, n_step=200)
    assert result["crossing_n"] is not None
    assert 8000 <= result["crossing_n"] <= 9500


def test_bc_matched_comparison_routes() -> None:
    hyp = {
        "text": "For prime power q, matched Bose-Chowla comparison at n=q^2+q up to 5000",
        "test_methodology": "bose-chowla matched prime q",
    }
    ev = run_combinatorics_experiment(hyp, ["sidon_set_density"])
    assert ev.get("metric_name") == "bc_matched_win_rate"
    assert ev.get("comparison_table")


def test_ap_free_sweep_runs() -> None:
    hyp = {
        "text": "AP-free density trend for n in {100, 500, 1000, 2000}",
        "test_methodology": "greedy ap-free sweep",
    }
    ev = run_combinatorics_experiment(hyp, ["arithmetic_progression_free_density"])
    assert "ap_free" in str(ev.get("metric_name"))


def test_large_n_sweep_under_60s() -> None:
    hyp = {
        "text": "Greedy Sidon ratio for n in {10000, 20000, 30000}",
        "test_methodology": "greedy ratio sweep",
    }
    start = time.time()
    ev = run_combinatorics_experiment(hyp, ["sidon_set_density"])
    elapsed = time.time() - start
    assert ev.get("metric_name") == "sidon_ratio_to_sqrt_n"
    assert elapsed < 60
