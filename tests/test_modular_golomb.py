"""Exact CP-SAT modular Golomb ruler finder (OEIS A004135 / A004136) — sanity gate.

We only exercise tiny n (n<=4, sub-second beyond first-call warmup) so CI stays fast; the
encoding was validated during development against the published OEIS terms through n=6/7
(both variants) via ``min_modular_ruler``. The pure-Python witness verifier
``is_modular_sidon`` is tested without OR-tools.
"""
from __future__ import annotations

import pytest

from propab.domain_modules.math_combinatorics.discovery import (
    A004135_TERMS,
    A004136_TERMS,
    certify_modular_ruler,
    decide_modular_ruler,
    get_record,
    is_modular_sidon,
    min_modular_ruler,
    ortools_available,
)

needs_ortools = pytest.mark.skipif(
    not ortools_available(), reason="OR-tools (ortools) not installed"
)


# --- Pure-Python verifier (no solver): must be paranoid and independent. --------------
def test_is_modular_sidon_positive():
    # OEIS A004136 example: {0,1,3} in Z_7 — 0+0,0+1,0+3,1+1,1+3,3+3 all distinct mod 7.
    assert is_modular_sidon([0, 1, 3], 7, include_repeats=True)
    assert is_modular_sidon([0, 1, 3], 7, include_repeats=False)


def test_is_modular_sidon_negative():
    # {0,1,2,3} in Z_7: distinct-pair sums 0+3=3 and 1+2=3 collide mod 7.
    assert not is_modular_sidon([0, 1, 2, 3], 7, include_repeats=False)
    # A double collision only matters when repeats count: 0+0=0 vs ... use Z_4 {0,2}:
    # incl. repeats 2+2=0 == 0+0 -> not a perfect ruler; distinct-pairs is fine.
    assert not is_modular_sidon([0, 2], 4, include_repeats=True)
    assert is_modular_sidon([0, 2], 4, include_repeats=False)
    # out-of-range and duplicate residues are rejected.
    assert not is_modular_sidon([0, 1, 7], 7, include_repeats=False)
    assert not is_modular_sidon([0, 0, 1], 7, include_repeats=False)


# --- Sanity gate: reproduce known OEIS terms exactly (finder recomputes a(n)). ---------
@needs_ortools
@pytest.mark.parametrize("n", [3, 4])
def test_reproduces_A004135_distinct_pairs(n):
    exp = A004135_TERMS[n]
    r = min_modular_ruler(n, include_repeats=False, k_max=exp + 1, per_k_budget=10, workers=4)
    assert r["outcome"] == "found"
    assert r["k"] == exp  # matches OEIS A004135
    assert r["proven_minimal"] is True  # every smaller k proven UNSAT
    assert is_modular_sidon(r["set"], r["k"], include_repeats=False)  # independent recheck


@needs_ortools
@pytest.mark.parametrize("n", [3, 4])
def test_reproduces_A004136_incl_repeats(n):
    exp = A004136_TERMS[n]
    r = min_modular_ruler(n, include_repeats=True, k_max=exp + 1, per_k_budget=10, workers=4)
    assert r["outcome"] == "found"
    assert r["k"] == exp  # matches OEIS A004136
    assert r["proven_minimal"] is True
    assert is_modular_sidon(r["set"], r["k"], include_repeats=True)


@needs_ortools
def test_decide_below_optimum_is_unsat():
    # a(4)=6 for A004135, so Z_5 must be PROVEN infeasible for a 4-subset (minimality).
    d = decide_modular_ruler(4, 5, include_repeats=False, time_budget=10)
    assert d["outcome"] == "unsat" and d["proven"] is True


# --- Certifier: cheap, deterministic, honest (upper bound only). ----------------------
@needs_ortools
def test_certifier_accepts_valid_ruler_as_upper_bound():
    r = min_modular_ruler(4, include_repeats=True, k_max=A004136_TERMS[4] + 1,
                          per_k_budget=10, workers=4)
    cert = certify_modular_ruler(
        {"k": r["k"], "set": r["set"]}, r["k"],
        include_repeats=True, published_best_k=A004136_TERMS[4], expected_n=4,
    )
    assert cert["certified"] is True
    assert cert["checks"]["is_modular_sidon"] is True
    # k == published best is NOT a strict improvement.
    assert cert["beats_published"] is False


def test_certifier_rejects_bad_witness():
    # A residue-collision witness must never certify (guards the record-claim path).
    cert = certify_modular_ruler([0, 1, 2, 3], 7, include_repeats=False, expected_n=4)
    assert cert["certified"] is False
    assert cert["checks"]["is_modular_sidon"] is False


# --- Registry consistency: sourced terms match the finder's known-term tables. --------
@pytest.mark.parametrize(
    "oeis_id,terms",
    [("A004135", A004135_TERMS), ("A004136", A004136_TERMS)],
)
def test_registry_matches_known_terms(oeis_id, terms):
    for n, val in terms.items():
        rec = get_record(oeis_id, n)
        assert rec is not None
        assert rec["best_known"] == val
        assert rec["status"] == ("open" if val is None else "proven_optimal")
