"""Exact CP-SAT B_3 finder (OR-tools) — encoding-correctness sanity gate.

We only exercise n=4 here (sub-second) so CI stays fast; the encoding was validated
against the proven optima a(4)=6, a(5)=8 (both proven-optimal) and sound UNSAT proofs
for size+1 during development. n>=6 does not close in a CI-sized budget (that scaling
limit is the honest finding — CP-SAT alone does not reach a(7)), so it is not tested here.
"""
from __future__ import annotations

import pytest

from propab.domain_modules.math_combinatorics.discovery import (
    certify_b3_record,
    decide_b3_cpsat,
    is_B3,
    max_b3_cpsat,
    ortools_available,
)

pytestmark = pytest.mark.skipif(
    not ortools_available(), reason="OR-tools (ortools) not installed"
)


def test_cpsat_reproduces_proven_optimum_n4():
    # a(4) = 6 is proven optimal; CP-SAT must both find it AND close the gap.
    r = max_b3_cpsat(4, time_budget=30)
    assert r["size"] == 6
    assert r["proven_optimal"] is True
    assert is_B3([tuple(v) for v in r["set"]])  # independent re-verification


def test_cpsat_proves_plus_one_unsat_n4():
    # No B_3 set of size 7 exists in {0,1}^4 — CP-SAT must PROVE it (a real
    # infeasibility proof), which is what makes "a(n)=k optimal" defensible.
    d = decide_b3_cpsat(4, 7, time_budget=30)
    assert d["outcome"] == "unsat"
    assert d["proven"] is True


def test_cpsat_sat_witness_is_genuinely_b3_n4():
    d = decide_b3_cpsat(4, 6, time_budget=30)
    assert d["outcome"] == "sat"
    assert d["size"] >= 6
    assert is_B3([tuple(v) for v in d["set"]])  # the solver's witness is real


def test_certifier_rejects_a_cpsat_nonrecord():
    # A size-6 cap in {0,1}^4 does not beat any published record here; the certifier
    # must not certify it as a record (guards the record-claim path).
    d = decide_b3_cpsat(4, 6, time_budget=30)
    cert = certify_b3_record({"n": 4, "set": d["set"]}, published_best=6, expected_n=4)
    assert cert["certified"] is False  # 6 is not strictly > 6


# ---------------------------------------------------------------------------
# Lex-leader symmetry breaking (SOUNDNESS GATE).
# The reduction must never remove an optimum: it has to still reproduce the proven
# optima a(4)=6, a(5)=8 and still prove size+1 UNSAT. Default is lex_leader=True, so
# the tests above already exercise it; these pin the contract explicitly and cross-check
# the symmetry-broken model against the origin-only model.
# ---------------------------------------------------------------------------
def test_lex_leader_reproduces_proven_optimum_n4():
    r = max_b3_cpsat(4, time_budget=30, lex_leader=True)
    assert r["size"] == 6 and r["proven_optimal"] is True
    assert is_B3([tuple(v) for v in r["set"]])  # symmetry-broken witness is genuinely B_3
    d = decide_b3_cpsat(4, 7, time_budget=30, lex_leader=True)
    assert d["outcome"] == "unsat" and d["proven"] is True  # +1 still proven infeasible


def test_lex_leader_reproduces_proven_optimum_n5():
    # a(5)=8 proven optimal must survive symmetry breaking (a stronger soundness gate).
    r = max_b3_cpsat(5, time_budget=60, lex_leader=True)
    assert r["size"] == 8 and r["proven_optimal"] is True
    assert is_B3([tuple(v) for v in r["set"]])
    d = decide_b3_cpsat(5, 9, time_budget=60, lex_leader=True)
    assert d["outcome"] == "unsat" and d["proven"] is True


def test_lex_leader_agrees_with_origin_only_n4():
    # Sound symmetry breaking cannot change the optimum: lex-on and lex-off agree.
    on = max_b3_cpsat(4, time_budget=30, lex_leader=True)
    off = max_b3_cpsat(4, time_budget=30, lex_leader=False)
    assert on["size"] == off["size"] == 6
    assert on["proven_optimal"] and off["proven_optimal"]
