"""Verifier extensions A3, C1, C2, C3 (fixes.md)."""
from __future__ import annotations

import time

from propab.domain_modules.math_combinatorics.verifier import (
    _greedy_sidon_max_budgeted,
    _greedy_sidon_max_legacy,
    _greedy_sidon_max_optimized,
    _sidon_point,
    find_threshold_crossing,
    run_combinatorics_experiment,
)


def test_optimized_matches_legacy_small_n() -> None:
    for n in (100, 500, 1000):
        assert len(_greedy_sidon_max_optimized(n)) == len(_greedy_sidon_max_legacy(n))


def test_greedy_sidon_budget_returns_partial_best_so_far() -> None:
    # A tiny budget at large n forces the multi-start search to stop early. It must
    # still return a non-empty, valid (Sidon) best-so-far set rather than nothing.
    best, complete = _greedy_sidon_max_budgeted(100_000, time_budget=0.0001)
    assert complete is False
    assert len(best) > 0
    # Independently confirm the truncated prefix is a genuine Sidon set.
    sums: set[int] = set()
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            s = best[i] + best[j]
            assert s not in sums
            sums.add(s)


def test_greedy_sidon_no_budget_runs_to_completion() -> None:
    # Default (no budget) is unchanged legacy behavior: always complete.
    best, complete = _greedy_sidon_max_budgeted(500)
    assert complete is True
    assert len(best) == len(_greedy_sidon_max_legacy(500))


def test_sidon_point_tags_partial_on_budget(monkeypatch) -> None:
    import propab.domain_modules.math_combinatorics.verifier as vmod

    monkeypatch.setattr(vmod, "SIDON_TIME_BUDGET", 0.0001)
    pt = _sidon_point(100_000, method="greedy")
    assert pt.get("partial") is True
    assert pt.get("partial_reason") == "budget_reached"
    assert pt.get("best_so_far_size", 0) > 0
    assert pt["max_sidon_size"] == pt["best_so_far_size"]


def test_cap_set_tags_partial_on_budget(monkeypatch) -> None:
    import propab.domain_modules.math_combinatorics.verifier as vmod
    from propab.domain_modules.math_combinatorics.verifier import compute_cap_set

    # Force the B&B to hit its budget immediately so the cap is best-so-far.
    monkeypatch.setattr(vmod, "CAP_BB_TIME_BUDGET", 0.0)
    res = compute_cap_set(4)
    assert res.get("partial") is True
    assert res.get("partial_reason") == "budget_reached"
    assert res["cap_valid"] is True  # best-so-far is still an independently-valid cap
    assert res.get("best_so_far_size") == res["cap_set_size"]


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
