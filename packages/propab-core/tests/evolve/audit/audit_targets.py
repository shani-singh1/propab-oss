"""Adversarial fixtures for the Auditor (WS-6).

Every target here is a REAL verifier — exact integer / rational arithmetic, no scripted stubs — so
the audit is exercised against genuine computation. Each broken variant contains a bug we could
plausibly have WRITTEN ourselves, not a contrived one:

  * IllegalAcceptedProblem    — forgets the rank check, and scores "min weight over the *nonzero*
                                codewords". A rank-3 matrix then looks like a legal [7,4] with d=3.
  * DegenerateAcceptedProblem — the same convenience, plus the vacuous-truth hole: the all-zero code
                                has NO nonzero codewords, so the minimum is taken over the EMPTY SET.
                                It reports the best possible score for the emptiest possible object.
  * EchoScoreProblem          — trusts the candidate's self-reported `min_distance`.
  * CachedScoreProblem        — the score depends on in-memory state that no record carries.
  * ApproxSolverProblem       — scores a rational objective with a numeric routine (~3e-7 error).

These live in their own module (not inside the test file) because the auditor re-verifies claims in a
FRESH PROCESS: the Problem must be importable by module path there. A fixture that only exists inside
a pytest function is, by construction, unreproducible — which is the auditor's whole point.
"""
from __future__ import annotations

import random
import socket
from fractions import Fraction
from typing import Any

from propab.evolve.auditor import KILL, BestKnownSource, CheckResult, Control
from propab.evolve.problem import INVALID, Verdict

NOW = 1_750_000_000.0   # fixed "retrieved_at" so provenance staleness is deterministic in tests

# The [7,4] Hamming code: full rank over GF(2), true minimum distance 3, systematic form [I | P].
HAMMING_7_4 = [
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
]
# The same code with coordinates reversed: still rank 4, still d=3, but NOT in systematic form.
HAMMING_7_4_PERMUTED = [row[::-1] for row in HAMMING_7_4]

# Row 3 == row 0 XOR row 1 => rank 3. An ILLEGAL [7,4] generator: it spans 8 codewords, not 16.
RANK_DEFICIENT_7_4 = [
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 1, 1],
]
ALL_ZERO_7_4 = [[0] * 7 for _ in range(4)]

# In-memory state that no record carries. Simulates a search-state / result cache — the thing a fresh
# process will not have, and therefore the thing that must never be load-bearing for a claim.
LIVE_CACHE: dict[str, float] = {}


# --------------------------------------------------------------------------- #
# GF(2) helpers
# --------------------------------------------------------------------------- #
def _is_binary_matrix(candidate: Any, k: int, n: int) -> bool:
    if not isinstance(candidate, (list, tuple)) or len(candidate) != k:
        return False
    for row in candidate:
        if not isinstance(row, (list, tuple)) or len(row) != n:
            return False
        for x in row:
            if isinstance(x, bool) or not isinstance(x, int) or x not in (0, 1):
                return False
    return True


def _pack(rows: list[list[int]]) -> list[int]:
    return [sum(bit << i for i, bit in enumerate(row)) for row in rows]


def _gf2_rank(rows: list[list[int]], n: int) -> int:
    words = _pack(rows)
    rank = 0
    for col in range(n):
        pivot = next((i for i in range(rank, len(words)) if (words[i] >> col) & 1), None)
        if pivot is None:
            continue
        words[rank], words[pivot] = words[pivot], words[rank]
        for i in range(len(words)):
            if i != rank and (words[i] >> col) & 1:
                words[i] ^= words[rank]
        rank += 1
    return rank


def _codewords(rows: list[list[int]], k: int) -> list[int]:
    words = _pack(rows)
    out = []
    for msg in range(1 << k):
        acc = 0
        for i in range(k):
            if (msg >> i) & 1:
                acc ^= words[i]
        out.append(acc)
    return out


def _min_weight_over_nonzero_messages(rows: list[list[int]], k: int, n: int) -> tuple[int, int, int]:
    """Correct for a FULL-RANK generator: min Hamming weight over all 2^k - 1 nonzero messages."""
    words = _pack(rows)
    best_w, best_c, best_m = n + 1, 0, 0
    for msg in range(1, 1 << k):
        acc = 0
        for i in range(k):
            if (msg >> i) & 1:
                acc ^= words[i]
        w = bin(acc).count("1")
        if w < best_w:
            best_w, best_c, best_m = w, acc, msg
    return best_w, best_c, best_m


def _min_positive_weight(rows: list[list[int]], k: int) -> tuple[int | None, int, int]:
    """THE BUG, written the way we would actually write it: "the minimum weight of a NONZERO codeword".

    Skipping zero codewords silently repairs a rank-deficient matrix (it just ignores the collisions),
    and on the all-zero matrix it is a minimum over the EMPTY SET — vacuously true.
    Returns (None, 0, 0) when no nonzero codeword exists.
    """
    words = _pack(rows)
    best_w: int | None = None
    best_c = best_m = 0
    for msg in range(1, 1 << k):
        acc = 0
        for i in range(k):
            if (msg >> i) & 1:
                acc ^= words[i]
        if acc == 0:
            continue                      # "not a real codeword" — the fatal little convenience
        w = bin(acc).count("1")
        if best_w is None or w < best_w:
            best_w, best_c, best_m = w, acc, msg
    return best_w, best_c, best_m


def _witness(k: int, n: int, d: float, word: int, msg: int) -> dict[str, Any]:
    return {
        "n": n,
        "k": k,
        "min_distance": d,
        "witness_codeword": [(word >> i) & 1 for i in range(n)],
        "witness_message": [(msg >> i) & 1 for i in range(k)],
        "method": "exhaustive_enumeration",
    }


# --------------------------------------------------------------------------- #
# The honest target
# --------------------------------------------------------------------------- #
class CodeProblem:
    """Maximize the minimum distance of a binary [n,k] linear code. Exact, integral, offline.

    The fiction for these tests: the cited record for this cell is d=2 and the engine found d=3.
    `known_constructions` is the tabulated-construction list — empty by default (an open cell), and
    populated in the rediscovery test so that the same candidate becomes a rediscovery instead.
    """

    name = "code[7,4]"

    def __init__(
        self,
        n: int = 7,
        k: int = 4,
        best: float = 2.0,
        known_constructions: list[list[list[int]]] | None = None,
    ) -> None:
        self.n = n
        self.k = k
        self.best = float(best)
        self.known_constructions = known_constructions or []

    # -- Problem contract -------------------------------------------------- #
    def describe(self) -> str:
        return f"Build a binary [{self.n},{self.k}] generator matrix maximizing minimum distance."

    def verify(self, candidate: Any) -> Verdict:
        if not _is_binary_matrix(candidate, self.k, self.n):
            return INVALID
        rows = [list(r) for r in candidate]
        if _gf2_rank(rows, self.n) != self.k:
            return Verdict(False, float("-inf"), {"reason": "generator matrix is not full rank"})
        d, word, msg = _min_weight_over_nonzero_messages(rows, self.k, self.n)
        return Verdict(True, float(d), _witness(self.k, self.n, d, word, msg))

    def best_known(self) -> float:
        return self.best

    def is_improvement(self, verdict: Verdict) -> bool:
        return bool(verdict.valid) and verdict.score > self.best

    def seed_programs(self) -> list[str]:
        return ["def build():\n    return " + repr(HAMMING_7_4) + "\n"]

    # -- Auditor hooks ------------------------------------------------------ #
    def best_known_provenance(self) -> BestKnownSource:
        return BestKnownSource(
            value=self.best,
            source=f"Grassl, Bounds on the minimum distance of linear codes, [{self.n},{self.k}] cell",
            citation="https://codetables.de/",
            retrieved_at=NOW,
            kind="table",
        )

    def exact_best_known(self) -> Fraction:
        return Fraction(self.best).limit_denominator(10**6)

    def independent_check(self, candidate: Any) -> tuple[bool, str]:
        """Second, deliberately separate implementation of "is this a legal [n,k] generator?".

        verify() decides full rank by Gaussian elimination. This one never touches elimination: it
        enumerates all 2^k codewords and checks they are DISTINCT (equivalent to full rank, computed
        a completely different way). If the two ever disagree, one of them is broken — and we believe
        this one.
        """
        if not _is_binary_matrix(candidate, self.k, self.n):
            return False, "candidate is not a k x n binary matrix"
        words = _codewords([list(r) for r in candidate], self.k)
        distinct = len(set(words))
        if distinct != (1 << self.k):
            return (
                False,
                f"the {1 << self.k} messages collapse onto only {distinct} distinct codewords — the "
                f"generator is rank-deficient, so this is NOT a [{self.n},{self.k}] code at all",
            )
        return True, f"all {1 << self.k} codewords distinct; the matrix is a legal generator"

    def controls(self) -> list[Control]:
        return [
            Control("hamming_7_4_known_d3", HAMMING_7_4, expect_valid=True, expect_score=3.0),
            Control("rank_deficient", RANK_DEFICIENT_7_4, expect_valid=False),
            Control("wrong_shape", [[1, 0], [0, 1]], expect_valid=False),
            Control("non_binary", [[2, 0, 0, 0, 1, 1, 0], *HAMMING_7_4[1:]], expect_valid=False),
        ]

    def exact_score(self, candidate: Any) -> Fraction | None:
        v = self.verify(candidate)
        if not v.valid:
            return None
        return Fraction(int(v.detail["min_distance"]))   # a minimum distance is an INTEGER

    def is_degenerate(self, candidate: Any) -> str | None:
        if _is_binary_matrix(candidate, self.k, self.n) and not any(any(r) for r in candidate):
            return "the generator matrix is all-zero: it generates only the zero codeword"
        return None

    def trivial_rediscovery(self, candidate: Any, verdict: Verdict) -> str | None:
        _ = verdict
        rows = [list(r) for r in candidate] if isinstance(candidate, (list, tuple)) else None
        if rows is not None and rows in self.known_constructions:
            return (
                f"the candidate is identical to a tabulated construction for [{self.n},{self.k}] — "
                "reproducing a known code is not a discovery"
            )
        return None

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        """Re-check the RECORDED witness against the RECORDED candidate, from scratch."""
        word = witness.get("witness_codeword")
        msg = witness.get("witness_message")
        claimed_d = witness.get("min_distance")
        if not isinstance(word, list) or not isinstance(msg, list):
            return False, "witness is missing witness_codeword / witness_message"
        if not _is_binary_matrix(candidate, self.k, self.n):
            return False, "candidate is not a legal matrix, so the witness cannot be checked"
        rows = [list(r) for r in candidate]
        acc = [0] * self.n
        for i, bit in enumerate(msg):
            if bit:
                acc = [a ^ b for a, b in zip(acc, rows[i])]
        if acc != word:
            return False, f"the witness message encodes to {acc}, not to the witness codeword {word}"
        if sum(word) != claimed_d:
            return False, f"the witness codeword has weight {sum(word)}, not the claimed {claimed_d}"
        return True, f"witness codeword {word} re-verified: weight {sum(word)} == claimed d"


# --------------------------------------------------------------------------- #
# Realistic verifier bugs
# --------------------------------------------------------------------------- #
class IllegalAcceptedProblem(CodeProblem):
    """verify() drops the rank check and scores "min weight of a nonzero codeword".

    On RANK_DEFICIENT_7_4 that yields d=3 — indistinguishable from the real Hamming code — so a
    rank-3 matrix is reported as a legal [7,4] improvement. Its own controls still pass: the
    negatives it ships (wrong shape, non-binary) do not cover this bug. That is precisely why
    controls alone are not enough, and why the independent second implementation exists.
    """

    name = "code[7,4]-illegal-accepted"

    def verify(self, candidate: Any) -> Verdict:
        if not _is_binary_matrix(candidate, self.k, self.n):
            return INVALID
        rows = [list(r) for r in candidate]
        d, word, msg = _min_positive_weight(rows, self.k)   # NO rank check — the bug
        if d is None:
            return INVALID
        return Verdict(True, float(d), _witness(self.k, self.n, d, word, msg))

    def controls(self) -> list[Control]:
        return [
            Control("hamming_7_4_known_d3", HAMMING_7_4, expect_valid=True, expect_score=3.0),
            Control("wrong_shape", [[1, 0], [0, 1]], expect_valid=False),
            Control("non_binary", [[2, 0, 0, 0, 1, 1, 0], *HAMMING_7_4[1:]], expect_valid=False),
        ]


class DegenerateAcceptedProblem(IllegalAcceptedProblem):
    """The vacuous-truth hole: "the minimum weight of a nonzero codeword" over the ALL-ZERO code is a
    minimum over the empty set. Written the natural way — seed the running minimum with n and take
    the min — it reports d = n = 7: the best possible score for the emptiest possible object.

    Nothing about the arithmetic is wrong. Every constraint is "satisfied". The check passes for the
    wrong reason, and a search will find this hole long before it finds a code.
    """

    name = "code[7,4]-degenerate-accepted"

    def verify(self, candidate: Any) -> Verdict:
        if not _is_binary_matrix(candidate, self.k, self.n):
            return INVALID
        rows = [list(r) for r in candidate]
        d, word, msg = _min_positive_weight(rows, self.k)
        if d is None:
            d, word, msg = self.n, 0, 0     # vacuous: no nonzero codeword => "the distance is n"
        return Verdict(True, float(d), _witness(self.k, self.n, d, word, msg))

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"


class BrokenVerifierProblem(CodeProblem):
    """An off-by-one in the distance computation. It cannot score a KNOWN code correctly."""

    name = "code[7,4]-broken-verifier"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        return v if not v.valid else Verdict(True, v.score + 1.0, dict(v.detail))


class RaisesOnGarbageProblem(CodeProblem):
    """verify() crashes on garbage instead of returning Verdict(valid=False) — contract violation.

    Mutated programs emit None constantly. If a crash is read as "invalid", the engine's invalid
    signal actually means "we do not know", and a crash is not evidence of anything.
    """

    name = "code[7,4]-raises"

    def verify(self, candidate: Any) -> Verdict:
        if candidate is None:
            raise TypeError("cannot verify None")
        return super().verify(candidate)


class NondeterministicProblem(CodeProblem):
    """A stochastic scorer. The engine takes the max over millions of draws, so it WILL "win"."""

    name = "code[7,4]-nondeterministic"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        return v if not v.valid else Verdict(True, v.score + random.random(), dict(v.detail))

    def controls(self) -> list[Control]:
        # Deliberately loose, so the noise does not trip the CONTROL check first: the point of this
        # fixture is that the determinism check is what catches it.
        return [
            Control("hamming_noisy", HAMMING_7_4, expect_valid=True, expect_score=3.5, tol=1.0),
            Control("rank_deficient", RANK_DEFICIENT_7_4, expect_valid=False),
        ]


class CachedScoreProblem(CodeProblem):
    """A MEMOIZING verifier: if it has scored this candidate before, it returns the cached number.

    This is a bug we would absolutely write — memoizing an expensive verifier is the obvious
    optimization. But the cache is in-memory search state that no Record carries, so whatever put a
    wrong number in there (a stale scoring path, an earlier buggy build, a hand-poked value during
    debugging) becomes the published result.

    In THIS process the claim is airtight: verify() returns the same number every single time,
    deterministically, and every in-process check agrees with every other. Only a FRESH process —
    which has an empty cache — discovers that the number was never in the candidate at all.
    """

    name = "code[7,4]-cached"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        if not v.valid:
            return v
        cached = LIVE_CACHE.get(repr(candidate))
        if cached is None:
            return v
        return Verdict(True, float(cached), dict(v.detail))

    def exact_score(self, candidate: Any) -> Fraction | None:
        v = self.verify(candidate)
        return None if not v.valid else Fraction(v.score).limit_denominator(10**6)

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"


class EchoScoreProblem(CodeProblem):
    """verify() TRUSTS the candidate's self-reported `min_distance`: the program grades itself."""

    name = "code[7,4]-echo"

    def verify(self, candidate: Any) -> Verdict:
        if not isinstance(candidate, dict):
            return INVALID
        matrix = candidate.get("matrix")
        if not _is_binary_matrix(matrix, self.k, self.n):
            return INVALID
        rows = [list(r) for r in matrix]
        if _gf2_rank(rows, self.n) != self.k:
            return INVALID
        _, word, msg = _min_weight_over_nonzero_messages(rows, self.k, self.n)
        claimed = candidate.get("min_distance")          # the bug: believe the program
        score = float(claimed) if isinstance(claimed, (int, float)) else 0.0
        return Verdict(True, score, _witness(self.k, self.n, score, word, msg))

    def _matrix(self, candidate: Any) -> Any:
        return candidate.get("matrix") if isinstance(candidate, dict) else candidate

    def independent_check(self, candidate: Any) -> tuple[bool, str]:
        return super().independent_check(self._matrix(candidate))

    def is_degenerate(self, candidate: Any) -> str | None:
        return super().is_degenerate(self._matrix(candidate))

    def exact_score(self, candidate: Any) -> Fraction | None:
        return None

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"

    def controls(self) -> list[Control]:
        return [
            Control(
                "hamming_self_reported_3",
                {"matrix": HAMMING_7_4, "min_distance": 3},
                expect_valid=True,
                expect_score=3.0,
            ),
            Control(
                "rank_deficient",
                {"matrix": RANK_DEFICIENT_7_4, "min_distance": 3},
                expect_valid=False,
            ),
        ]


class NetworkVerifierProblem(CodeProblem):
    """verify() reaches for the network. The contract forbids it; the auditor ENFORCES it.

    In-process this is harmless (constructing a socket object sends nothing). In the audit's fresh
    process sockets are blocked — so an LLM-in-the-verifier, the thing that turned "win" #1 into
    judge noise, cannot hide in here.
    """

    name = "code[7,4]-network"

    def verify(self, candidate: Any) -> Verdict:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # no connect(): no traffic
        sock.close()
        return super().verify(candidate)


class IncoherentImprovementProblem(CodeProblem):
    """is_improvement() says yes to anything valid, even when it does not beat best_known."""

    name = "code[7,4]-incoherent"

    def is_improvement(self, verdict: Verdict) -> bool:
        return bool(verdict.valid)


class NoWitnessProblem(CodeProblem):
    """verify() returns a bare score with no evidence attached."""

    name = "code[7,4]-no-witness"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        return Verdict(True, v.score, {}) if v.valid else v

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"


class TableLookupProblem(CodeProblem):
    """The witness admits the number came out of a table rather than a computation."""

    name = "code[7,4]-table"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        if not v.valid:
            return v
        detail = dict(v.detail)
        detail["construction_source"] = "best_known_table"
        return Verdict(True, v.score, detail)


class ExtraKillProblem(CodeProblem):
    """A target that ships its own target-specific kill check."""

    name = "code[7,4]-extra"

    def audit_checks(self):
        identity = [[1 if i == j else 0 for j in range(self.k)] for i in range(self.k)]

        def systematic_form_required(ctx) -> CheckResult:
            leading = [list(r[: self.k]) for r in ctx.candidate]
            if leading != identity:
                return CheckResult(
                    "target[systematic_form]",
                    False,
                    KILL,
                    "this target only publishes generator matrices in systematic form [I | P]; the "
                    f"leading {self.k}x{self.k} block is {leading}, not the identity",
                )
            return CheckResult("target[systematic_form]", True, KILL)

        return [systematic_form_required]


class VacuousIndependentCheckProblem(CodeProblem):
    """The highest-severity check, stubbed out.

    Someone had to ship, the second implementation was tedious, and `return True` made the audit go
    green. Now the one thing standing between a verifier bug and a published illegal object approves
    everything — including candidates that are not even matrices. The audit would still say PASS,
    which is the auditor's own vacuous-truth failure, one level up.
    """

    name = "code[7,4]-vacuous-independent-check"

    def independent_check(self, candidate: Any) -> tuple[bool, str]:
        return True, "looks fine"


class ModelMemoryProvenanceProblem(CodeProblem):
    """The baseline is "what the model remembered". This is failure #2, verbatim."""

    name = "code[7,4]-model-memory"

    def best_known_provenance(self) -> BestKnownSource:
        return BestKnownSource(
            value=self.best,
            source="the model said this was the record",
            citation="",
            retrieved_at=NOW,
            kind="model_memory",
        )


class DriftedProvenanceProblem(CodeProblem):
    """best_known() and the cited source disagree: the code is not comparing against the record."""

    name = "code[7,4]-drift"

    def best_known_provenance(self) -> BestKnownSource:
        return BestKnownSource(
            value=self.best + 1.0,     # the table says 3; the code compares against 2
            source="Grassl, codetables.de, [7,4] cell",
            citation="https://codetables.de/",
            retrieved_at=NOW,
            kind="table",
        )


class UnserializableProblem(CodeProblem):
    """The candidate cannot be written down, so it can never be published or re-checked."""

    name = "code[7,4]-unserializable"

    def _matrix(self, candidate: Any) -> Any:
        return candidate.get("matrix") if isinstance(candidate, dict) else candidate

    def verify(self, candidate: Any) -> Verdict:
        return super().verify(self._matrix(candidate))

    def independent_check(self, candidate: Any) -> tuple[bool, str]:
        return super().independent_check(self._matrix(candidate))

    def is_degenerate(self, candidate: Any) -> str | None:
        return super().is_degenerate(self._matrix(candidate))

    def exact_score(self, candidate: Any) -> Fraction | None:
        return None

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"


# --------------------------------------------------------------------------- #
# Variants that OMIT a hook entirely. Registered in module globals so they stay picklable for the
# fresh-process re-check (you cannot `del` an inherited method, so lookup is blocked explicitly).
# --------------------------------------------------------------------------- #
def _without(class_name: str, *dropped: str) -> type:
    blocked = set(dropped)

    def __getattribute__(self, item):   # noqa: N807
        if item in blocked:
            raise AttributeError(item)
        return object.__getattribute__(self, item)

    cls = type(
        class_name,
        (CodeProblem,),
        {
            "name": f"code[7,4]-missing-{'-'.join(dropped)}",
            "__module__": __name__,
            "__qualname__": class_name,
            "__doc__": f"CodeProblem with {', '.join(dropped)} absent.",
            "__getattribute__": __getattribute__,
        },
    )
    globals()[class_name] = cls
    return cls


NoProvenanceProblem = _without("NoProvenanceProblem", "best_known_provenance")
NoIndependentCheckProblem = _without("NoIndependentCheckProblem", "independent_check")
NoControlsProblem = _without("NoControlsProblem", "controls")


# --------------------------------------------------------------------------- #
# A rational-arithmetic target, for the floating-point kills.
# --------------------------------------------------------------------------- #
class RationalSumProblem:
    """Maximize a sum of rationals given as [numerator, denominator] pairs.

    The objective is EXACTLY representable (Fraction) but verify() accumulates it in floating point —
    the real-world shape of the float-artifact failure: a rational problem run through a float
    verifier. 1/10 + 2/10 sums to 0.30000000000000004, which "beats" a baseline of 0.3.
    """

    name = "rational-sum"

    def __init__(self, best: float = 0.3, exact_best: Fraction = Fraction(3, 10)) -> None:
        self.best = float(best)
        self.exact_best = exact_best

    def describe(self) -> str:
        return "Return [[num, den], ...] maximizing the sum; each term in (0, 1]."

    def _terms(self, candidate: Any) -> list[Fraction] | None:
        if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
            return None
        terms: list[Fraction] = []
        for pair in candidate:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None
            num, den = pair
            if isinstance(num, bool) or isinstance(den, bool):
                return None
            if not isinstance(num, int) or not isinstance(den, int) or den <= 0 or num < 0:
                return None
            frac = Fraction(num, den)
            if frac > 1:
                return None
            terms.append(frac)
        return terms

    def verify(self, candidate: Any) -> Verdict:
        terms = self._terms(candidate)
        if terms is None:
            return INVALID
        total = 0.0
        for frac in terms:
            total += float(frac)          # naive float accumulation — the artifact lives here
        return Verdict(
            True,
            total,
            {
                "terms": [[t.numerator, t.denominator] for t in terms],
                "float_sum": total,
                "method": "float_accumulation",
            },
        )

    def best_known(self) -> float:
        return self.best

    def is_improvement(self, verdict: Verdict) -> bool:
        return bool(verdict.valid) and verdict.score > self.best

    def seed_programs(self) -> list[str]:
        return ["def build():\n    return [[1, 10], [2, 10]]\n"]

    def best_known_provenance(self) -> BestKnownSource:
        return BestKnownSource(
            value=self.best,
            source="synthetic fixture record",
            citation="tests/evolve/audit_targets.py::RationalSumProblem",
            retrieved_at=NOW,
            kind="table",
        )

    def exact_best_known(self) -> Fraction:
        return self.exact_best

    def exact_score(self, candidate: Any) -> Fraction | None:
        terms = self._terms(candidate)
        return None if terms is None else sum(terms, Fraction(0))

    def independent_check(self, candidate: Any) -> tuple[bool, str]:
        terms = self._terms(candidate)
        if terms is None:
            return False, "candidate is not a list of >=2 [num, den] pairs with 0 < num/den <= 1"
        return True, f"{len(terms)} legal rational terms"

    def controls(self) -> list[Control]:
        return [
            Control("two_halves", [[1, 2], [1, 2]], expect_valid=True, expect_score=1.0),
            Control("zero_denominator", [[1, 0], [1, 2]], expect_valid=False),
            Control("term_above_one", [[3, 2], [1, 2]], expect_valid=False),
        ]

    def is_degenerate(self, candidate: Any) -> str | None:
        return None

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        if witness.get("terms") != candidate:
            return False, "the witness terms do not match the candidate"
        return True, "witness terms match the candidate"


class ApproxSolverProblem(RationalSumProblem):
    """verify() scores via a numerical routine carrying ~3e-7 of error — comfortably above the
    float-noise floor, so ONLY the exact re-check can tell that the "improvement" is not real."""

    name = "rational-sum-approx"
    ERROR = 3e-7

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        if not v.valid:
            return v
        detail = dict(v.detail)
        detail["method"] = "iterative_numeric_solver"
        return Verdict(True, v.score + self.ERROR, detail)

    def controls(self) -> list[Control]:
        # A numeric verifier's controls carry a numeric tolerance, as they would in real life.
        return [
            Control("two_halves", [[1, 2], [1, 2]], True, expect_score=1.0, tol=1e-5),
            Control("zero_denominator", [[1, 0], [1, 2]], False),
        ]

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"


class ToleranceProblem(RationalSumProblem):
    """The verifier declares its own slop, and the entire "improvement" fits inside it."""

    name = "rational-sum-tolerance"

    def verify(self, candidate: Any) -> Verdict:
        v = super().verify(candidate)
        if not v.valid:
            return v
        detail = dict(v.detail)
        detail["tol"] = 1e-3                            # accurate only to 1e-3…
        return Verdict(True, v.score + 1e-4, detail)    # …and the "win" is 1e-4

    def controls(self) -> list[Control]:
        return [
            Control("two_halves", [[1, 2], [1, 2]], True, expect_score=1.0, tol=1e-2),
            Control("zero_denominator", [[1, 0], [1, 2]], False),
        ]

    def exact_score(self, candidate: Any) -> Fraction | None:
        return None      # isolate the tolerance kill from the exact-arithmetic kill

    def check_witness(self, candidate: Any, witness: dict[str, Any]) -> tuple[bool, str]:
        return True, "n/a for this fixture"
