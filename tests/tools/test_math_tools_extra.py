"""Tests for the 5 remaining M1/M5 mathematics tools — correctness + honesty invariants.

These tools are built to a strict honesty spec: an identity is 'proven' only when BOTH a
symbolic simplify==0 AND dense numeric sampling agree (else refuted-with-counterexample
or an honest unknown); linear-algebra / polynomial results carry an independent re-check;
a PSLQ relation is a CONJECTURE re-derived at higher precision (spurious ⇒ not_confirmed);
and a counterexample search self-certifies its witness and never upgrades "none found"
into a universal proof. The tests assert those invariants, not just the happy path.
"""
from __future__ import annotations

import mpmath as mp
import sympy as sp

from propab.tools.registry import ToolRegistry

_R = ToolRegistry()


def call(name, **params):
    return _R.call(name, params)


def _proportional(a, b):
    """True iff integer vectors a, b point in the same/opposite direction (relation identity)."""
    from fractions import Fraction

    if a is None or b is None or len(a) != len(b):
        return False
    ratio = None
    for x, y in zip(a, b):
        if x == 0 and y == 0:
            continue
        if x == 0 or y == 0:
            return False
        r = Fraction(int(y), int(x))
        if ratio is None:
            ratio = r
        elif r != ratio:
            return False
    return ratio is not None


# ── symbolic_verify_identity (proven needs BOTH gates; refuted ships a counterexample) ──
def test_verify_identity_true_is_proven():
    r = call("symbolic_verify_identity", lhs="sin(x)**2 + cos(x)**2", rhs="1")
    assert r.success and r.output["verdict"] == "proven"
    # 'proven' required BOTH the symbolic zero AND numeric agreement — never one alone.
    assert r.output["symbolic_zero"] is True and r.output["numeric_agree"] is True


def test_verify_identity_near_miss_is_refuted_with_counterexample():
    r = call("symbolic_verify_identity", lhs="x**2 - 1", rhs="(x-1)**2")
    assert r.success and r.output["verdict"] == "refuted"
    cx = r.output["counterexample"]
    assert cx is not None and "point" in cx
    # HONESTY: the counterexample is self-certifying — re-substitute it independently.
    subs = {sp.Symbol(k): sp.sympify(v) for k, v in cx["point"].items()}
    lhs, rhs = sp.sympify("x**2 - 1"), sp.sympify("(x-1)**2")
    assert lhs.subs(subs) != rhs.subs(subs)  # lhs really differs from rhs at that point


def test_verify_identity_undecidable_is_unknown_not_overclaimed():
    # sqrt(x**2) == Abs(x) is TRUE over the reals, but simplify cannot prove it without a
    # real assumption. Numeric sampling agrees, yet the tool refuses to claim 'proven'.
    r = call("symbolic_verify_identity", lhs="sqrt(x**2)", rhs="Abs(x)")
    assert r.success and r.output["verdict"] == "unknown"
    assert r.output["counterexample"] is None
    assert r.output["symbolic_zero"] is False  # honestly not proven symbolically


def test_verify_identity_bad_input_is_validation_error():
    r = call("symbolic_verify_identity", lhs="(((", rhs="1")
    assert not r.success and r.error.type == "validation_error"


# ── exact_linear_algebra (exact values; solutions independently re-verified) ─────────────
def test_exact_linear_algebra_det_exact():
    r = call("exact_linear_algebra", op="det", matrix=[[1, 2], [3, 4]])
    assert r.success and r.output["result"] == "-2" and r.output["field"] == "QQ"


def test_exact_linear_algebra_inverse_is_reverified():
    r = call("exact_linear_algebra", op="inverse", matrix=[[4, 7], [2, 6]])
    assert r.success and r.output["verified"] is True
    # HONESTY: independently confirm M * inv == I from the returned entries.
    M = sp.Matrix([[4, 7], [2, 6]])
    inv = sp.Matrix([[sp.sympify(e) for e in row] for row in r.output["result"]])
    assert sp.simplify(M * inv) == sp.eye(2)


def test_exact_linear_algebra_gf_solve_reverified():
    # 2x + y = 3, x + 3y = 5 over GF(7); the returned solution is re-substituted.
    r = call("exact_linear_algebra", op="solve", matrix=[[2, 1], [1, 3]], b=[3, 5], modulus=7)
    assert r.success and r.output["verified"] is True and r.output["field"] == "GF(7)"
    x = r.output["result"]["solution"]
    assert (2 * x[0] + x[1]) % 7 == 3 and (x[0] + 3 * x[1]) % 7 == 5


def test_exact_linear_algebra_bad_input_is_validation_error():
    r = call("exact_linear_algebra", op="teleport", matrix=[[1]])
    assert not r.success and r.error.type == "validation_error"


# ── polynomial_tools (factorization/groebner re-checked; roots exact + complete) ─────────
def test_polynomial_factor_is_reverified():
    r = call("polynomial_tools", op="factor", poly="x**4 - 1")
    assert r.success and r.output["verified"] is True
    # HONESTY: the returned factorization re-expands to the input.
    assert sp.expand(sp.sympify(r.output["result"]) - sp.sympify("x**4 - 1")) == 0


def test_polynomial_roots_exact_and_complete():
    r = call("polynomial_tools", op="roots", poly="x**2 - 5*x + 6")
    assert r.success and r.output["complete"] is True and r.output["verified"] is True
    assert set(r.output["result"]["roots"].keys()) == {"2", "3"}


def test_polynomial_groebner_membership_verified():
    r = call("polynomial_tools", op="groebner", polys=["x**2 + y**2 - 1", "x - y"], var=["x", "y"])
    assert r.success and r.output["verified"] is True
    assert isinstance(r.output["result"]["basis"], list) and r.output["result"]["basis"]


def test_polynomial_bad_input_is_validation_error():
    r = call("polynomial_tools", op="factorize", poly="x**2 - 1")  # 'factorize' is not a valid op
    assert not r.success and r.error.type == "validation_error"


# ── integer_relation (conjecture confirmed at higher precision; spurious ⇒ not_confirmed) ─
def _phi_strings(digits=80):
    """High-precision strings for 1, phi, phi**2, and a near-miss phi**2 + 1e-13."""
    with mp.workdps(digits + 30):
        phi = (1 + mp.sqrt(5)) / 2
        return (mp.nstr(phi, digits),
                mp.nstr(phi * phi, digits),
                mp.nstr(phi * phi + mp.mpf("1e-13"), digits))


def test_integer_relation_true_relation_confirmed():
    # 1 + phi - phi**2 = 0  =>  relation proportional to [1, 1, -1].
    phi, phi2, _ = _phi_strings()
    r = call("integer_relation", values=["1", phi, phi2], precision=25, confirm_precision=60)
    assert r.success and r.output["status"] == "confirmed"
    assert _proportional(r.output["relation"], [1, 1, -1])
    assert r.output["independently_rederived"] is True


def test_integer_relation_spurious_is_not_confirmed():
    # phi**2 perturbed by 1e-13: a relation appears at low precision but dissolves higher up.
    phi, _, phi2_pert = _phi_strings()
    r = call("integer_relation", values=["1", phi, phi2_pert], precision=16, confirm_precision=50)
    assert r.success
    assert r.output["found"] is True          # it WAS found at the working precision
    assert r.output["status"] == "not_confirmed"  # ...but honestly rejected on re-derivation


def test_integer_relation_bad_input_is_validation_error():
    r = call("integer_relation", values=["1"])  # need at least 2 constants
    assert not r.success and r.error.type == "validation_error"


# ── counterexample_search (self-certifying witness; "none found" never a false proof) ────
def test_counterexample_search_finds_real_counterexample():
    # Euler's polynomial n**2 - n + 41 is prime for n=0..40 but composite at n=41 (=41**2).
    r = call("counterexample_search",
             space={"type": "integers", "low": 0, "high": 50, "var": "n"},
             predicate={"type": "is_prime", "expr": "n**2 - n + 41"})
    assert r.success and r.output["found"] is True
    cx = r.output["counterexample"]
    assert cx["point"]["n"] == 41
    # HONESTY: self-certifying — re-evaluation on the witness agrees P is false there.
    assert cx["evaluation"]["is_prime"] is False
    assert cx["recheck_predicate_holds"] is False
    # independent hand-check: 41**2 - 41 + 41 == 41**2, a perfect square (composite).
    assert (41 ** 2 - 41 + 41) == 41 ** 2


def test_counterexample_search_exhausted_true_case_is_not_a_universal_proof():
    # n < 2**n holds for every n in [1,20]; the search exhausts the finite space honestly.
    r = call("counterexample_search",
             space={"type": "integers", "low": 1, "high": 20, "var": "n"},
             predicate={"type": "relation", "lhs": "n", "op": "<", "rhs": "2**n"})
    assert r.success and r.output["found"] is False
    assert r.output["exhausted"] is True and r.output["checked"] == 20
    # HONESTY: a null result is never a 'proven' verdict; the note scopes it to the finite space.
    assert "NOT a proof of the universal statement" in r.output["note"]


def test_counterexample_search_bad_input_is_validation_error():
    r = call("counterexample_search",
             space={"type": "integers", "low": 0, "high": 5, "var": "n"},
             predicate={"type": "teleport", "expr": "n"})
    assert not r.success and r.error.type == "validation_error"
