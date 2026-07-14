"""Tests for Target A — best-known binary linear codes (evolve/targets/ecc.py).

No network. The whole point of this target is a verifier that is exact, total, and offline.
"""
from __future__ import annotations

import numpy as np
import pytest

from propab.domain_modules.coding_theory.constructors import (
    MAX_EXHAUSTIVE_K,
    best_known_distance,
    extended_hamming_code,
    hamming_code,
    parity_check_code,
    reed_muller_rm1,
    repetition_code,
    simplex_code,
)
from propab.evolve.problem import INVALID, Problem, Verdict
from propab.evolve.targets import ecc
from propab.evolve.targets.ecc import BestKnown, ECCProblem, UnsourcedCellError


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_implements_problem_protocol():
    assert isinstance(ECCProblem(7, 4), Problem)


def test_describe_states_the_spec():
    text = ECCProblem(35, 10).describe()
    for needed in ("n=35", "k=10", "minimum distance", "full rank", "GF(2)"):
        assert needed.lower() in text.lower(), needed
    # it must tell the model the number to beat
    assert "12" in text and "13" in text


# --------------------------------------------------------------------------- #
# A known-optimal code verifies to exactly its tabulated distance.
#
# This is also an independent cross-check of the sourced record table: the distances below are
# COMPUTED by exhaustive enumeration from the repo's own constructors, and compared against the
# best-known values this module sourced from codetables.de.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("build", "n", "k", "expected_d"),
    [
        (lambda: hamming_code(3), 7, 4, 3),             # [7,4,3] Hamming
        (lambda: extended_hamming_code(3), 8, 4, 4),    # [8,4,4] extended Hamming
        (lambda: hamming_code(4), 15, 11, 3),           # [15,11,3] Hamming
        (lambda: simplex_code(4), 15, 4, 8),            # [15,4,8] simplex
        (lambda: reed_muller_rm1(4), 16, 5, 8),         # [16,5,8] RM(1,4)
        (lambda: repetition_code(10), 10, 1, 10),       # [10,1,10] repetition
        (lambda: parity_check_code(5), 6, 5, 2),        # [6,5,2] parity check
    ],
)
def test_known_optimal_code_verifies_to_its_true_distance(build, n, k, expected_d):
    problem = ECCProblem(n, k)
    verdict = problem.verify(build())

    assert verdict.valid
    assert verdict.score == float(expected_d)
    assert verdict.detail["min_distance"] == expected_d
    # the sourced record agrees with the computed optimum
    assert problem.best_known() == float(expected_d)

    # A result without a witness is not a result.
    witness = verdict.detail["witness_codeword"]
    message = verdict.detail["witness_message"]
    assert len(witness) == n
    assert len(message) == k
    assert sum(witness) == expected_d

    # The witness must be independently re-derivable from the generator + message.
    g = np.asarray(verdict.detail["generator_matrix"]) % 2
    assert list((np.asarray(message) @ g) % 2) == list(witness)


# --------------------------------------------------------------------------- #
# Garbage in, verdict out. A mutated program emits anything; verify() must NEVER raise.
# --------------------------------------------------------------------------- #
GARBAGE = [
    None,
    "not a matrix",
    b"\x00\x01",
    {"g": 1},
    {1, 2, 3},
    [],
    [[]],
    0,
    3.5,
    float("nan"),
    np.array([]),
    np.zeros((0, 0)),
    np.arange(7),                                  # 1-D
    np.zeros((2, 3, 4)),                           # 3-D — would crash compute_min_distance directly
    [[1, 0], [0]],                                 # ragged
    np.full((4, 7), np.nan),                       # NaN
    np.full((4, 7), np.inf),                       # inf
    np.full((4, 7), 0.5),                          # non-integer
    np.full((4, 7), 1e300),                        # out of range
    np.array([["a", "b"], ["c", "d"]]),            # strings
    np.ones((3, 7), dtype=np.uint8),               # wrong k
    np.ones((4, 9), dtype=np.uint8),               # wrong n
    np.zeros((4, 7), dtype=np.uint8),              # all-zero -> rank deficient
    np.array([[1, 1, 1, 1, 1, 1, 1]] * 4, dtype=np.uint8),   # duplicate rows -> rank 1
    [hamming_code(3), hamming_code(3)],            # a list of matrices, not a matrix
]


@pytest.mark.parametrize("junk", GARBAGE, ids=range(len(GARBAGE)))
def test_verify_is_total_on_garbage(junk):
    problem = ECCProblem(7, 4)
    verdict = problem.verify(junk)   # must not raise

    assert verdict.valid is False
    assert verdict.score == float("-inf")
    assert verdict.detail["reason"]
    assert problem.is_improvement(verdict) is False


def test_verify_rejects_rank_deficient_generator():
    # 4 rows but the 4th is the sum of the first three -> rank 3, not a [7,4] code.
    g = hamming_code(3).astype(int)
    g[3] = (g[0] + g[1] + g[2]) % 2
    verdict = ECCProblem(7, 4).verify(g)

    assert verdict.valid is False
    assert "rank" in verdict.detail["reason"].lower()


# --------------------------------------------------------------------------- #
# is_improvement — the anti-self-deception surface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("build", "n", "k"),
    [
        (lambda: hamming_code(3), 7, 4),
        (lambda: extended_hamming_code(3), 8, 4),
        (lambda: simplex_code(4), 15, 4),
        (lambda: reed_muller_rm1(4), 16, 5),
        (lambda: hamming_code(4), 15, 11),
    ],
)
def test_is_improvement_refuses_rediscovery_of_a_tabulated_code(build, n, k):
    """Reconstructing a code that is already in the table is NOT a discovery."""
    problem = ECCProblem(n, k)
    verdict = problem.verify(build())

    assert verdict.valid                       # it is a real, correctly-verified code
    assert verdict.score == problem.best_known()   # ...that exactly matches the record
    assert problem.is_improvement(verdict) is False


def test_is_improvement_rejects_a_distance_with_no_witness():
    problem = ECCProblem(35, 10)
    forged = Verdict(
        valid=True,
        score=13.0,
        detail={"target": [35, 10], "min_distance": 13},   # no witness_codeword
    )
    assert problem.is_improvement(forged) is False


def test_is_improvement_rejects_table_lookup_evidence():
    problem = ECCProblem(35, 10)
    forged = Verdict(
        valid=True,
        score=13.0,
        detail={
            "target": [35, 10],
            "min_distance": 13,
            "witness_codeword": [1] * 13 + [0] * 22,
            "witness_message": [1] + [0] * 9,
            "construction_source": "best_known_table",     # read off a table
            "generator_matrix": np.zeros((10, 35), dtype=int).tolist(),
        },
    )
    assert problem.is_improvement(forged) is False


def test_is_improvement_rejects_a_witness_that_does_not_reproduce():
    """The claimed distance must re-derive from the evidence alone, as a third party would."""
    problem = ECCProblem(35, 10)
    real = problem.verify(_valid_generator(35, 10))
    assert real.valid

    tampered = Verdict(
        valid=True,
        score=13.0,
        detail={**real.detail, "min_distance": 13, "witness_message": [1] + [0] * 9},
    )
    assert problem.is_improvement(tampered) is False


def test_is_improvement_rejects_a_verdict_for_a_different_cell():
    problem = ECCProblem(35, 10)
    other = ECCProblem(7, 4).verify(hamming_code(3))
    assert other.valid
    assert problem.is_improvement(other) is False


def test_is_improvement_rejects_invalid_verdicts():
    problem = ECCProblem(35, 10)
    assert problem.is_improvement(INVALID) is False
    assert problem.is_improvement(None) is False


# --------------------------------------------------------------------------- #
# is_improvement — the POSITIVE path.
#
# No real open cell can be beaten on demand (that is the whole point), so we lower the record on a
# real cell and check that a real, witnessed, re-derivable code is accepted — and that raising the
# record back flips it to False. This exercises the accept path without faking any evidence.
# --------------------------------------------------------------------------- #
def _valid_generator(n: int, k: int) -> np.ndarray:
    """A real [n,k] generator: G = [I_k | R], deterministic. Always full rank."""
    rng = np.random.default_rng(7)
    r = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8)
    return np.hstack([np.eye(k, dtype=np.uint8), r])


def test_is_improvement_accepts_a_real_witnessed_beat():
    problem = ECCProblem(35, 10)
    verdict = problem.verify(_valid_generator(35, 10))
    assert verdict.valid
    achieved = int(verdict.score)
    assert 0 < achieved < problem.record.upper     # a normal, unremarkable code

    # Same code, but pretend the record was one below what we achieved.
    problem.record = BestKnown(
        35, 10, achieved - 1, 13, "synthetic lower bound (test)", "synthetic upper bound (test)"
    )
    assert problem.best_known() == float(achieved - 1)
    assert problem.is_improvement(verdict) is True

    # Restore a record equal to what we achieved -> a tie is not a beat.
    problem.record = BestKnown(
        35, 10, achieved, 13, "synthetic lower bound (test)", "synthetic upper bound (test)"
    )
    assert problem.is_improvement(verdict) is False


def test_is_improvement_refuses_to_exceed_a_proven_upper_bound():
    """Beating a published NONEXISTENCE proof is our bug, not a discovery. Refuse it."""
    problem = ECCProblem(35, 10)
    verdict = problem.verify(_valid_generator(35, 10))
    achieved = int(verdict.score)

    # Record whose proven upper bound sits BELOW what we just computed.
    problem.record = BestKnown(
        35, 10, 1, achieved - 1, "synthetic (test)", "synthetic upper bound (test)"
    )
    assert verdict.score > problem.record.upper
    assert problem.is_improvement(verdict) is False


# --------------------------------------------------------------------------- #
# The record itself
# --------------------------------------------------------------------------- #
def test_unsourced_cell_is_not_ready_to_run():
    """A cell we cannot cite must be refused, not silently given a baseline of zero."""
    assert ecc.lookup(40, 10) is None
    with pytest.raises(UnsourcedCellError):
        ECCProblem(40, 10)


def test_cell_beyond_the_exhaustive_limit_is_refused():
    """Without a full 2^k enumeration there is no honest witness."""
    assert ecc.lookup(31, 21) is not None            # we have the record...
    with pytest.raises(ValueError, match="exhaustive"):
        ECCProblem(31, 21)                           # ...but k=21 > MAX_EXHAUSTIVE_K


def test_baseline_is_conservative_against_the_repo_table():
    """The repo's BEST_KNOWN_TABLE says [16,12] -> d=4. The truth is d=2 (only 15 nonzero vectors
    exist in GF(2)^4, so 16 distinct nonzero parity-check columns cannot exist => d <= 2).

    We take the MAX of every source, so a wrong table entry can never let us bank a fake record.
    """
    assert ecc.lookup(16, 12).lower == 2             # our sourced value: correct
    repo = best_known_distance(16, 12)
    problem = ECCProblem(16, 12)
    assert problem.best_known() == float(max(2, repo or 0))
    assert problem.best_known() >= 2                 # never below the truth


def test_open_cells_are_really_open_and_verifiable():
    cells = ecc.open_cells()
    assert cells, "no open cells sourced — the target has nothing to search"
    for r in cells:
        assert r.lower < r.upper                     # a real gap
        assert r.gap >= 1
        assert r.k <= MAX_EXHAUSTIVE_K               # the verifier can actually check it
        assert r.lower_source and r.upper_source     # both bounds are cited
    assert (35, 10) in {(r.n, r.k) for r in cells}


def test_sourced_records_obey_the_singleton_bound():
    for (n, k), rec in ecc.BEST_KNOWN.items():
        assert 0 < k <= n
        assert 1 <= rec.lower <= rec.upper <= n - k + 1, f"[{n},{k}] violates Singleton"


def test_recommended_cell_is_open():
    problem = ecc.recommended_problem()
    assert (problem.n, problem.k) == ecc.RECOMMENDED_CELL
    assert problem.record.is_open
    assert problem.record.gap >= 1
    assert problem.k <= MAX_EXHAUSTIVE_K


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
def test_every_seed_program_runs_and_emits_verifiable_candidates():
    problem = ECCProblem(35, 10)
    seeds = problem.seed_programs()
    assert len(seeds) >= 5

    for i, source in enumerate(seeds):
        namespace: dict = {}
        exec(compile(source, f"<seed{i}>", "exec"), namespace)   # noqa: S102 — our own source
        assert "build" in namespace, f"seed {i} defines no build()"

        out = namespace["build"]()
        candidates = out if isinstance(out, list) else [out]
        assert candidates, f"seed {i} emitted nothing"

        verdicts = [problem.verify(c) for c in candidates]
        valid = [v for v in verdicts if v.valid]
        assert valid, f"seed {i} emitted no valid [35,10] code"
        for v in valid:
            assert v.detail["witness_codeword"]
            assert 0 < v.score <= problem.record.upper


def test_seeds_never_produce_a_false_improvement():
    """The seeds are known constructions. None of them may be scored as a discovery."""
    problem = ECCProblem(35, 10)
    for i, source in enumerate(problem.seed_programs()):
        namespace: dict = {}
        exec(compile(source, f"<seed{i}>", "exec"), namespace)   # noqa: S102
        out = namespace["build"]()
        for candidate in (out if isinstance(out, list) else [out]):
            assert problem.is_improvement(problem.verify(candidate)) is False
