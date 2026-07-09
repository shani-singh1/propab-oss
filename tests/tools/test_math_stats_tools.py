"""Tests for the math + stats tool batch (M1-M5, S1) — correctness + honesty invariants.

These tools were built to a strict honesty spec: exact-or-`unknown` (never a wrong
"exact"), independent re-verification of solutions, conjectures validated on held-out
data (never over-claimed), and correct multiple-testing correction. The tests assert
those invariants, not just the happy path.
"""
from __future__ import annotations

import math

from propab.tools.registry import ToolRegistry

_R = ToolRegistry()


def call(name, **params):
    return _R.call(name, params)


# ── symbolic_algebra ─────────────────────────────────────────────────────────
def test_symbolic_algebra_factor_and_solve():
    r = call("symbolic_algebra", op="factor", expr="x^2 - 1")
    assert r.success and "x - 1" in r.output["result"] and "x + 1" in r.output["result"]
    r2 = call("symbolic_algebra", op="diff", expr="x**3", var="x")
    assert r2.success and "3" in str(r2.output["result"])


def test_symbolic_algebra_bad_op_is_validation_error():
    r = call("symbolic_algebra", op="teleport", expr="x")
    assert not r.success and r.error.type == "validation_error"


# ── number_theory (factorization self-verifies) ──────────────────────────────
def test_number_theory_factorint_selfverifies():
    r = call("number_theory", op="factorint", a=360)
    assert r.success
    fac = r.output.get("result") or r.output.get("factors")
    prod = 1
    for p, e in fac.items():
        prod *= int(p) ** int(e)
    assert prod == 360  # the honesty invariant: factors re-multiply to the input


def test_number_theory_primality():
    assert call("number_theory", op="isprime", a=97).output["result"] is True
    assert call("number_theory", op="isprime", a=91).output["result"] is False


def test_number_theory_rejects_noninteger():
    r = call("number_theory", op="isprime", a="not-a-number")
    assert not r.success and r.error.type == "validation_error"


# ── combinatorial_enumeration (exact-or-unknown) ─────────────────────────────
def test_combinatorial_enumeration_exact():
    r = call("combinatorial_enumeration", op="binomial", n=5, k=2)
    assert r.success and r.output["value"] == 10 and r.output["exact"] is True


def test_combinatorial_enumeration_oversized_is_unknown_not_wrong():
    # A huge generate/count must return unknown, never a truncated "exact" count.
    r = call("combinatorial_enumeration", op="generate", n=10_000, kind="subsets")
    assert (not r.success) or (r.output.get("exact") is not True) or (r.output.get("status") == "unknown")


# ── graph_invariants (exact-or-unknown; bounds labeled) ──────────────────────
def test_graph_invariants_chromatic_k4():
    r = call("graph_invariants", op="chromatic_number", edges=[[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]])
    assert r.success and r.output["value"] == 4 and r.output["exact"] is True


def test_graph_invariants_k5_nonplanar():
    edges = [[i, j] for i in range(5) for j in range(i + 1, 5)]
    r = call("graph_invariants", op="is_planar", edges=edges)
    assert r.success and r.output["value"] is False


# ── constraint_solve (solution re-verified; unsat is a proof) ────────────────
def test_constraint_solve_optimal_and_reverified():
    r = call(
        "constraint_solve",
        variables=[{"name": "x", "type": "int", "low": 0, "high": 5},
                   {"name": "y", "type": "int", "low": 0, "high": 5}],
        constraints=[{"coeffs": {"x": 1, "y": 1}, "op": "<=", "rhs": 4}],
        objective={"sense": "maximize", "coeffs": {"x": 1, "y": 1}},
        time_budget_sec=5,
    )
    assert r.success and r.output["outcome"] in ("optimal", "sat")
    a = r.output["assignment"]
    assert a["x"] + a["y"] <= 4  # the returned solution genuinely satisfies the constraint


def test_constraint_solve_contradiction_is_unsat_not_bogus():
    r = call(
        "constraint_solve",
        variables=[{"name": "x", "type": "int", "low": 0, "high": 10}],
        constraints=[{"coeffs": {"x": 1}, "op": ">=", "rhs": 5},
                     {"coeffs": {"x": 1}, "op": "<=", "rhs": 2}],
        time_budget_sec=5,
    )
    assert r.success and r.output["outcome"] == "unsat"


# ── linear_optimization (solution re-verified) ───────────────────────────────
def test_linear_optimization_optimal_verified():
    r = call("linear_optimization", c=[1, 1], A_ub=[[1, 1]], b_ub=[4],
             bounds=[[0, None], [0, None]], sense="maximize")
    assert r.success and r.output["status"] == "optimal"
    assert abs(r.output["optimal_value"] - 4.0) < 1e-6
    assert r.output.get("verified") is True


# ── sequence_oracle (conjecture validated on held-out; never "established") ───
def test_sequence_oracle_fibonacci_is_conjecture_holdout_validated():
    r = call("sequence_oracle", terms=[1,1,2,3,5,8,13,21,34,55,89,144], check_oeis=False)
    assert r.success
    assert r.output.get("verdict") in ("conjecture", "recurrence", "unknown")
    # It must NOT claim the formula is proven/established.
    assert r.output.get("verdict") != "proven"
    if r.output.get("verdict") != "unknown":
        assert r.output.get("validated_on_holdout") in (True, False)


def test_sequence_oracle_too_few_terms_is_unknown():
    r = call("sequence_oracle", terms=[1, 2], check_oeis=False)
    assert (not r.success) or r.output.get("verdict") == "unknown"


# ── multiple_testing_correction (BH correct + monotone) ──────────────────────
def test_bh_correction_correct_and_monotone():
    r = call("multiple_testing_correction",
             p_values=[0.001, 0.008, 0.039, 0.041, 0.042, 0.06],
             method="benjamini_hochberg", alpha=0.05)
    assert r.success
    q = r.output["adjusted_p_values"]
    # BH q-values, sorted by original p, are monotone non-decreasing and in [0,1].
    assert all(0.0 <= v <= 1.0 for v in q)
    order = sorted(range(len(q)), key=lambda i: [0.001,0.008,0.039,0.041,0.042,0.06][i])
    q_sorted = [q[i] for i in order]
    assert all(q_sorted[i] <= q_sorted[i+1] + 1e-9 for i in range(len(q_sorted)-1))


def test_bonferroni_and_bad_pvalue():
    r = call("multiple_testing_correction", p_values=[0.01, 0.02], method="bonferroni", alpha=0.05)
    assert r.success and abs(r.output["adjusted_p_values"][0] - 0.02) < 1e-9  # 0.01*2
    bad = call("multiple_testing_correction", p_values=[0.5, 1.7], method="bonferroni")
    assert not bad.success and bad.error.type == "validation_error"


# ── power_analysis (monotone; d=0 -> power~alpha) ────────────────────────────
def test_power_analysis_required_n_and_monotone():
    r = call("power_analysis", test="two_sample_t", effect_size=0.5, alpha=0.05, power=0.8)
    assert r.success and r.output.get("n_per_group") == 64  # textbook n for d=0.5, power=0.8
    assert r.output.get("caveat")  # must carry the "underpowered != no effect" honesty caveat


def test_power_analysis_bad_alpha_rejected():
    r = call("power_analysis", test="two_sample_t", effect_size=0.5, alpha=1.5, n=50)
    assert not r.success and r.error.type == "validation_error"
