"""Erdős #143 target — the verifier is the whole product, so it is what gets tested hardest.

Reminder, restated here because it governs what these tests are allowed to assert: a finite search
can NEVER settle #143 (a counterexample is an INFINITE set with a DIVERGENT sum). Nothing below
tests "did we solve it". They test that the verifier decides the separation condition exactly, that
it is safe on garbage, and that the growth harness measures what it claims to measure.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from propab.evolve.targets.erdos143 import (
    ERDOS_PRIMITIVE_CONSTANT,
    Erdos143Problem,
    partial_sum_trend,
    trend_report,
)


@pytest.fixture(scope="module")
def prob() -> Erdos143Problem:
    return Erdos143Problem(n_max=200)


def _primes(n: int) -> list[int]:
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = bytearray(len(sieve[p * p :: p]))
    return [i for i in range(2, n + 1) if sieve[i]]


# --------------------------------------------------------------------------- #
# The separation condition
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [10, 50, 100, 200, 1000])
def test_primes_satisfy_separation(n: int) -> None:
    """The primes are a valid set: |k*p - q| is a positive integer for distinct primes."""
    p = Erdos143Problem(n_max=n)
    v = p.verify(_primes(n))
    assert v.valid, v.detail
    assert v.detail["all_integers"] is True
    assert v.detail["known_construction"].startswith("primes")
    # score is the primes' partial sum, i.e. exactly the sourced baseline
    assert v.score == pytest.approx(p.best_known(), rel=1e-12)


def test_violating_set_is_rejected_with_a_witness(prob: Erdos143Problem) -> None:
    """4 = 2*2, so |2*2 - 4| = 0 < 1. The witness must name the pair and the k."""
    v = prob.verify([2, 4])
    assert not v.valid
    assert v.detail["reason"] == "separation violated"
    assert v.detail["violating_pair"] == ["2", "4"]
    assert v.detail["k"] == 2
    assert Fraction(v.detail["gap"]) == 0


def test_multiples_are_rejected_generally(prob: Erdos143Problem) -> None:
    for cand in ([3, 9], [5, 10, 15], [2, 3, 6], [7, 7 * 13]):
        assert not prob.verify(cand).valid, cand


def test_valid_witness_is_a_quantified_margin(prob: Erdos143Problem) -> None:
    """A valid verdict carries the TIGHTEST constraint in the set — a re-derivable certificate."""
    v = prob.verify(_primes(200))
    w = v.detail["witness"]
    assert Fraction(w["min_gap"]) >= 1
    assert Fraction(w["min_slack"]) == Fraction(w["min_gap"]) - 1
    assert w["cross_checked_against_naive_scan"] is True
    x, y, k = w["tightest_pair_and_k"]
    assert abs(k * Fraction(x) - Fraction(y)) == Fraction(w["min_gap"])


def test_integer_case_is_exactly_primitivity(prob: Erdos143Problem) -> None:
    """L4: on integers, (SEP) <=> no element divides another. Checked against divisibility."""
    rng = random.Random(1143)
    for _ in range(300):
        s = sorted(rng.sample(range(2, 60), rng.randint(2, 6)))
        primitive = all(b % a != 0 for i, a in enumerate(s) for b in s[i + 1 :])
        assert prob.verify(s).valid == primitive, s


# --------------------------------------------------------------------------- #
# Structure: L3 (elements below 2) and the degenerate singleton
# --------------------------------------------------------------------------- #
def test_element_below_2_forces_a_singleton(prob: Erdos143Problem) -> None:
    """L3: if x < 2 its multiples are spaced < 2 apart, so every y > x lands within x/2 < 1."""
    for x in ("3/2", "11/10", "19/10", "1.05"):
        for y in (3, 5, 7, 100, 199):
            assert not prob.verify([x, y]).valid, (x, y)


def test_singleton_is_rejected_as_degenerate(prob: Erdos143Problem) -> None:
    """Without this, {1+eps} scores arbitrarily high and the search 'wins' with junk."""
    v = prob.verify(["101/100"])
    assert not v.valid
    assert "degenerate" in v.detail["reason"]
    # ...and the junk it would otherwise score is enormous, which is the whole point:
    x = 1.01
    assert 1.0 / (x * math.log(x)) > 90


def test_two_forces_odd_integers(prob: Erdos143Problem) -> None:
    """A real at distance >= 1 from every even integer must be the midpoint of a gap: an odd int."""
    assert prob.verify([2, 3]).valid
    assert prob.verify([2, 9]).valid
    assert not prob.verify([2, "9/2"]).valid  # 4.5 sits 0.5 from 4 and 5... i.e. from 2*2
    assert not prob.verify([2, "15/2"]).valid


# --------------------------------------------------------------------------- #
# Exact vs floating point
# --------------------------------------------------------------------------- #
def test_float_and_exact_agree_on_the_seed_constructions() -> None:
    p = Erdos143Problem(n_max=200)
    for src in p.seed_programs():
        ns: dict = {"__name__": "seed"}
        exec(compile(src, "<seed>", "exec"), ns)
        cand = ns["build"]()
        exact = p._ns["verify"](cand)
        screen = p._ns["verify_float"](cand)
        assert exact["valid"] == screen["valid"], cand[:5]
        if exact["valid"]:
            assert exact["score"] == pytest.approx(screen["score"], rel=1e-12)


def test_float_screen_lies_where_exact_is_right() -> None:
    """THE reason validity is decided in exact rationals and never in floats.

    A = {x, 11} with x = (12 - 1e-12)/5 misses the condition by exactly 1e-12:
        |5x - 11| = 1 - 1e-12 < 1   =>   INVALID.
    A float check with any sane tolerance is *lenient* and calls it valid. If we trusted the float
    path, that set would be banked as a "win". The exact path is the authority.
    """
    p = Erdos143Problem(n_max=200)
    x = Fraction(12 * 10**12 - 1, 5 * 10**12)  # (12 - 1e-12)/5
    cand = [str(x), 11]

    assert abs(5 * x - 11) == 1 - Fraction(1, 10**12)  # exactly 1e-12 short of the requirement

    exact = p.verify(cand)
    assert not exact.valid
    assert exact.detail["k"] == 5
    assert Fraction(exact.detail["gap"]) < 1

    screen = p._ns["verify_float"](cand, tol=1e-9)
    assert screen["valid"] is True  # the float screen is fooled — documented, and never trusted


def test_floats_are_parsed_to_their_exact_binary_value() -> None:
    """Fraction(float) is exact, so we verify the number actually stored — not a rounded ideal."""
    p = Erdos143Problem(n_max=200)
    assert p._ns["parse_real"](0.1) == Fraction(0.1) != Fraction(1, 10)
    assert p._ns["parse_real"]("2.4") == Fraction(12, 5)  # strings ARE the exact decimal
    assert p._ns["parse_real"]([7, 5]) == Fraction(7, 5)


def test_fast_path_agrees_with_the_literal_scan() -> None:
    """nearest_multiple_gap (O(1)/pair) must equal naive_gap (the definition, transcribed)."""
    p = Erdos143Problem(n_max=200)
    fast, naive = p._ns["nearest_multiple_gap"], p._ns["naive_gap"]
    rng = random.Random(7)
    for _ in range(2000):
        x = Fraction(rng.randint(21, 400), 10)
        y = x + Fraction(rng.randint(1, 3000), 10)
        g1, _k = fast(x, y)
        g2, _a = naive(x, y)
        assert g1 == g2, (x, y, g1, g2)
        assert (g1 < 1) == (g2 < 1)


# --------------------------------------------------------------------------- #
# Safety on garbage — a mutated program emits anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "junk",
    [
        None, "abc", b"x", 3, 3.5, [], {}, [[]], [None], [2, None], [2, "oops"], [1, 2], [2, 2],
        [0, 5], [-3, 7], [float("nan")], [float("inf")], [2, float("nan")], {"set": [2, 4]},
        [[1, 0]], [2, [3, 0]], ["1/0"], [2] * 3, list(range(2, 40)), "2,3,5", {"a": 1},
        [Fraction(3), 3], [True, 5],
    ],
)
def test_verifier_never_raises(prob: Erdos143Problem, junk: object) -> None:
    v = prob.verify(junk)
    assert isinstance(v.valid, bool)
    if not v.valid:
        assert v.score == float("-inf")
        assert "reason" in (v.detail or {})


def test_dict_candidate_is_unwrapped(prob: Erdos143Problem) -> None:
    assert prob.verify({"set": [2, 3, 5, 7]}).valid
    assert not prob.verify({"set": [2, 4]}).valid


def test_element_above_n_max_is_rejected() -> None:
    p = Erdos143Problem(n_max=50)
    assert not p.verify([2, 3, 5, 51]).valid
    assert p.verify([2, 3, 5, 47]).valid


# --------------------------------------------------------------------------- #
# Baseline, improvement, rediscovery
# --------------------------------------------------------------------------- #
def test_best_known_is_the_primes_partial_sum() -> None:
    p = Erdos143Problem(n_max=1000)
    expected = math.fsum(1.0 / (q * math.log(q)) for q in _primes(1000))
    assert p.best_known() == pytest.approx(expected, rel=1e-12)
    assert p.best_known() < ERDOS_PRIMITIVE_CONSTANT  # Lichtman's proven supremum
    src = p.best_known_source()
    assert "Lichtman" in src["citation"]
    assert "arXiv:2202.02384" in src["citation"]


def test_rediscovering_the_primes_is_not_an_improvement(prob: Erdos143Problem) -> None:
    v = prob.verify(_primes(200))
    assert v.valid
    assert v.score == pytest.approx(prob.best_known(), rel=1e-12)
    assert prob.is_improvement(v) is False  # it IS the baseline


def test_known_constructions_are_flagged_as_rediscovery() -> None:
    p = Erdos143Problem(n_max=200)
    half = list(range(101, 201))
    v = p.verify(half)
    assert v.valid
    assert v.detail["known_construction"] == "integers in (N/2, N]"
    assert p.is_improvement(v) is False


def test_is_improvement_requires_beating_the_baseline(prob: Erdos143Problem) -> None:
    assert prob.is_improvement(prob.verify([2, 4])) is False  # invalid
    assert prob.is_improvement(prob.verify([101, 103, 107])) is False  # valid but weak

    # A genuine (hypothetical) improvement: valid, unknown construction, above the baseline.
    from propab.evolve.problem import Verdict

    fake = Verdict(valid=True, score=prob.best_known() + 0.1, detail={"known_construction": None})
    assert prob.is_improvement(fake) is True
    # ...but not if it is one of our own bugs dressed up as a discovery.
    alarmed = Verdict(
        valid=True,
        score=prob.best_known() + 0.1,
        detail={"known_construction": None, "lichtman_alarm": "impossible"},
    )
    assert prob.is_improvement(alarmed) is False


def test_no_integer_set_may_exceed_the_lichtman_supremum() -> None:
    """Lichtman PROVES sum 1/(a log a) <= 1.6366 on integer primitive sets. If one of our integer
    candidates ever scored above it, our verifier would be wrong — so we assert it never does."""
    for n in (100, 1000, 3000):
        p = Erdos143Problem(n_max=n)
        v = p.verify(_primes(n))
        assert v.valid
        assert v.score < ERDOS_PRIMITIVE_CONSTANT
        assert "lichtman_alarm" not in v.detail


# --------------------------------------------------------------------------- #
# The verifier pin: what judged == what ships
# --------------------------------------------------------------------------- #
def test_fingerprint_pins_the_source_that_actually_judged() -> None:
    from propab.evolve.ledger_impl import verifier_fingerprint

    p = Erdos143Problem(n_max=200)
    src = p.verifier_source()
    assert verifier_fingerprint(src) == p.verifier_fingerprint()
    assert f"N_MAX = {p.n_max}" in src

    # Re-exec'ing the shipped source reproduces the target's verdicts exactly.
    ns: dict = {"__name__": "standalone"}
    exec(compile(src, "verifier.py", "exec"), ns)
    for cand in (_primes(200), [2, 4], [101, 103], ["7/2", "9"]):
        assert ns["verify"](cand)["valid"] == p.verify(cand).valid


def test_different_n_is_a_different_claim_and_a_different_pin() -> None:
    assert Erdos143Problem(n_max=100).verifier_fingerprint() != (
        Erdos143Problem(n_max=200).verifier_fingerprint()
    )


# --------------------------------------------------------------------------- #
# Seeds + the growth harness (the only thing this target is entitled to report)
# --------------------------------------------------------------------------- #
def test_all_seed_programs_execute_and_are_judged() -> None:
    p = Erdos143Problem(n_max=200)
    seeds = p.seed_programs()
    assert len(seeds) >= 6
    for src in seeds:
        ns: dict = {"__name__": "seed"}
        exec(compile(src, "<seed>", "exec"), ns)
        cand = ns["build"]()
        assert isinstance(cand, list)
        p.verify(cand)  # must not raise, valid or not


def test_greedy_from_2_rediscovers_the_primes() -> None:
    """Greedy ascending from 2 IS the sieve of Eratosthenes — a real (small) structural finding."""
    p = Erdos143Problem(n_max=200)
    ns: dict = {"__name__": "seed"}
    exec(compile(p.seed_programs()[3], "<greedy>", "exec"), ns)  # greedy rational grid
    cand = ns["build"]()
    v = p.verify(cand)
    assert v.valid
    assert v.detail["known_construction"].startswith("primes")
    assert v.score == pytest.approx(p.best_known(), rel=1e-12)


def test_partial_sum_growth_harness_over_increasing_n() -> None:
    """The measurement: max achievable partial sum at N, and how it moves as N grows.

    This asserts the SHAPE we actually observe and nothing more: the best construction we can find
    at each N is the primes; the maximum creeps up very slowly and stays under Lichtman's proven
    supremum; and it grows far more slowly than log log N. That is EVIDENCE CONSISTENT WITH the
    conjecture being true. It is not, and cannot be, a proof of anything.
    """
    ns = [50, 100, 200, 500, 1000]
    rows = partial_sum_trend(ns)
    assert {r["N"] for r in rows} == set(ns)

    best = {}
    for n in ns:
        valid = [r for r in rows if r["N"] == n and r["valid"]]
        assert valid, f"no valid seed at N={n}"
        best[n] = max(valid, key=lambda r: r["score"])

    scores = [best[n]["score"] for n in ns]

    # 1. monotone in N (a larger box cannot contain less)
    assert scores == sorted(scores)

    # 2. the maximum we can find is the primes, at every N, and it never beats the baseline
    for n in ns:
        assert best[n]["known_construction"].startswith("primes")
        assert best[n]["beats_baseline"] is False
        assert best[n]["score"] == pytest.approx(best[n]["baseline_primes"], rel=1e-12)

    # 3. it plateaus: 20x more room (50 -> 1000) buys ~0.10, and it stays below the Lichtman sup
    assert scores[-1] - scores[0] < 0.2
    assert scores[-1] < ERDOS_PRIMITIVE_CONSTANT

    # 4. it is NOT tracking log log N (which would be the divergent-looking signal)
    growth = scores[-1] - scores[0]
    loglog_growth = math.log(math.log(ns[-1])) - math.log(math.log(ns[0]))
    assert growth < 0.25 * loglog_growth

    # 5. the Koukoulopoulos-Lamzouri-Lichtman quantity sum(1/x)/log N decreases (-> 0)
    ratios = [best[n]["sum_1_over_x_div_log_N"] for n in ns]
    assert ratios == sorted(ratios, reverse=True)


def test_trend_report_leads_with_the_limitation() -> None:
    report = trend_report(partial_sum_trend([50, 100]))
    assert "NOT A SOLUTION TO ERD" in report.upper()
    assert "no finite search can be one" in report.lower()
    assert "cannot settle #143" in report
