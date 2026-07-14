"""Target A — best-known binary linear codes [n, k, d].

The base-rate target: a cheap EXACT verifier (exhaustive enumeration of the 2^k - 1 nonzero
codewords), a legible public record (Grassl/Brouwer, codetables.de), and hundreds of genuinely
open cells. A candidate is a k x n generator matrix over GF(2); the objective is to maximise the
minimum distance d.

This module is a THIN adapter. The verifier already exists and is battle-tested — see
``domain_modules/coding_theory/constructors.py`` (``compute_min_distance``, ``is_valid_generator``,
``trivial_rediscovery``, ``is_table_lookup_evidence``). Nothing here recomputes a distance.


WHY THIS FILE CARRIES ITS OWN RECORD TABLE
------------------------------------------
``coding_theory.BEST_KNOWN_TABLE`` is NOT usable as the baseline for a discovery claim:

  1. It is WRONG in at least one cell. It lists [16,12] -> d=4. The true value is d=2, and that is
     provable in one line without consulting any table: a [16,12] code has r = n - k = 4 parity
     checks, so its parity-check matrix H has 4 rows and 16 columns; d >= 3 requires all columns to
     be nonzero and pairwise distinct, but GF(2)^4 contains only 15 nonzero vectors < 16. Hence
     d <= 2. (Equivalently, sphere-packing: 2^12 * (1 + 16) = 69632 > 65536 = 2^16.) codetables.de
     independently reports lower = upper = 2 for [16,12].
  2. Every one of its cells is CLOSED (lower bound == upper bound == proven optimum). It contains no
     open cell, so a campaign run against it can only ever produce a rediscovery.

So the table below is sourced independently, records the UPPER bound as well as the lower bound (a
result above a *proven* upper bound is our bug, not a discovery), and carries a citation per bound.
An [n,k] cell with no sourced record is NOT ready to run: ``ECCProblem`` refuses to construct for it
rather than silently defaulting to a baseline of zero.

Source: M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes",
https://codetables.de — the per-cell BKLC pages, accessed 2026-07-14. Each open cell below was
looked up individually, not read off a summary grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...domain_modules.coding_theory.constructors import (
    MAX_EXHAUSTIVE_K,
    best_known_distance,
    compute_min_distance,
    is_table_lookup_evidence,
    is_valid_generator,
    recompute_distance_of_witness,
    trivial_rediscovery,
)
from ..problem import NEG_INF, Candidate, Verdict

CODETABLES = "M. Grassl, codetables.de (BKLC), accessed 2026-07-14"


# --------------------------------------------------------------------------- #
# The sourced record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BestKnown:
    """The real, citable record for one [n, k] cell.

    ``lower`` is the best minimum distance anyone has achieved (the number to beat).
    ``upper`` is the proven upper bound (no [n, k, upper + 1] code exists).
    The cell is OPEN — worth attacking — exactly when lower < upper.
    """

    n: int
    k: int
    lower: int
    upper: int
    lower_source: str
    upper_source: str
    accessed: str = CODETABLES

    @property
    def is_open(self) -> bool:
        return self.lower < self.upper

    @property
    def gap(self) -> int:
        return self.upper - self.lower


# Exactly-determined cells (lower == upper == optimum) for n <= 16, k = 1..n.
# Row n, position i  ->  k = i + 1. Cross-validated against two independent sources: the
# codetables.de bounds grid and the repo's BEST_KNOWN_TABLE. They agree on every entry EXCEPT
# [16,12], where the repo says 4 and the truth is 2 (see module docstring for the proof).
_OPTIMAL_SMALL_N: dict[int, list[int]] = {
    2: [2, 1],
    3: [3, 2, 1],
    4: [4, 2, 2, 1],
    5: [5, 3, 2, 2, 1],
    6: [6, 4, 3, 2, 2, 1],
    7: [7, 4, 4, 3, 2, 2, 1],
    8: [8, 5, 4, 4, 2, 2, 2, 1],
    9: [9, 6, 4, 4, 3, 2, 2, 2, 1],
    10: [10, 6, 5, 4, 4, 3, 2, 2, 2, 1],
    11: [11, 7, 6, 5, 4, 4, 3, 2, 2, 2, 1],
    12: [12, 8, 6, 6, 4, 4, 4, 3, 2, 2, 2, 1],
    13: [13, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    14: [14, 9, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    15: [15, 10, 8, 8, 7, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1],
    16: [16, 10, 8, 8, 8, 6, 6, 5, 4, 4, 4, 2, 2, 2, 2, 1],
}

_CLOSED_ANCHORS: dict[tuple[int, int], tuple[int, str]] = {
    (23, 12): (7, "binary Golay code [23,12,7]; optimal"),
    (24, 12): (8, "extended binary Golay code [24,12,8]; optimal"),
    (31, 16): (7, "BCH [31,16,7]; optimal"),
    (31, 21): (5, "BCH [31,21,5]; optimal"),
    (31, 26): (3, "Hamming [31,26,3]; optimal"),
}


def _build_registry() -> dict[tuple[int, int], BestKnown]:
    reg: dict[tuple[int, int], BestKnown] = {}
    for n, row in _OPTIMAL_SMALL_N.items():
        for i, d in enumerate(row):
            k = i + 1
            note = f"exactly determined: optimal binary [{n},{k},{d}] code"
            reg[(n, k)] = BestKnown(n, k, d, d, note, note)
    for (n, k), (d, note) in _CLOSED_ANCHORS.items():
        reg[(n, k)] = BestKnown(n, k, d, d, note, note)

    # ---- GENUINELY OPEN CELLS (lower < upper). Each looked up individually on codetables.de. ----
    for rec in (
        BestKnown(
            32, 14, 8, 9,
            "subcode of a [32,17,8] code — Y. Cheng & N.J.A. Sloane, 'Codes from symmetry "
            "groups', SIAM J. Discrete Math. 2 (1989) 28-37",
            "shortening to a [29,11,9] code — D.B. Jaffe, 'Binary linear codes: new results "
            "on nonexistence' (1996)",
        ),
        BestKnown(
            35, 10, 12, 13,
            "subcode of a [35,12,12] code — M. Morii, email communication, September 1993",
            "shortening to a [34,9,13] code — P.W. Heijnen, dissertation, T.U. Delft, "
            "October 1993 (no binary [33,9,13] code exists)",
        ),
        BestKnown(
            36, 10, 13, 14,
            "shortening of [38,12,13], itself a truncation of [39,12,14] — J. Bierbrauer & "
            "Y. Edel, IEEE Trans. Inf. Th. 43 (1997) 953-968",
            "shortening to a [33,7,14] code — H.C.A. van Tilborg, Discr. Math. 33 (1981) "
            "197-207",
        ),
    ):
        reg[(rec.n, rec.k)] = rec
    return reg


BEST_KNOWN: dict[tuple[int, int], BestKnown] = _build_registry()


class UnsourcedCellError(ValueError):
    """Raised for an [n, k] cell with no sourced best-known record.

    A cell whose record we cannot cite is not ready to run: without a real baseline, any
    ``is_improvement`` verdict is meaningless.
    """


def lookup(n: int, k: int) -> BestKnown | None:
    """The sourced record for [n, k], or None if we cannot cite one."""
    return BEST_KNOWN.get((int(n), int(k)))


def open_cells(max_k: int = MAX_EXHAUSTIVE_K) -> list[BestKnown]:
    """Sourced cells with a real gap (lower < upper) that this verifier can actually check.

    Sorted cheapest-to-verify first: verification cost is 2^k, independent of n.
    """
    cells = [r for r in BEST_KNOWN.values() if r.is_open and r.k <= max_k]
    return sorted(cells, key=lambda r: (r.k, r.n))


# --------------------------------------------------------------------------- #
# The Problem
# --------------------------------------------------------------------------- #
class ECCProblem:
    """One [n, k] cell of the binary-linear-code table. Implements `Problem`."""

    def __init__(self, n: int, k: int) -> None:
        n, k = int(n), int(k)
        if not 0 < k <= n:
            raise ValueError(f"require 0 < k <= n, got n={n}, k={k}")
        if k > MAX_EXHAUSTIVE_K:
            raise ValueError(
                f"k={k} exceeds the exact verifier's exhaustive limit "
                f"({MAX_EXHAUSTIVE_K}). Without a full 2^k enumeration there is no honest "
                "witness, so this cell cannot be run."
            )
        record = lookup(n, k)
        if record is None:
            raise UnsourcedCellError(
                f"no sourced best-known record for [{n},{k}] — this cell is NOT ready to run. "
                f"Look it up on codetables.de and add it to {__name__}.BEST_KNOWN with a "
                "citation for both the lower and the upper bound."
            )
        self.n = n
        self.k = k
        self.record = record
        self.name = f"ecc[{n},{k}]"

    # -- prompt surface ----------------------------------------------------- #
    def describe(self) -> str:
        r = self.record
        status = (
            f"OPEN: nobody has achieved d={r.upper}, and no code with d>{r.upper} exists "
            f"(proven). Closing this gap is a publishable result."
            if r.is_open
            else f"CLOSED: d={r.lower} is proven optimal. No improvement is possible."
        )
        return f"""\
TARGET: a binary linear code with block length n={self.n} and dimension k={self.k}.

OBJECTIVE: maximise the minimum distance d — the smallest Hamming weight among the 2^{self.k}-1
nonzero codewords. Higher d is strictly better.

CURRENT RECORD
  best known (lower bound):  d = {r.lower}   [{r.lower_source}]
  proven upper bound:        d = {r.upper}   [{r.upper_source}]
  status: {status}
  To count as a discovery your code must achieve d >= {r.lower + 1}.

CANDIDATE FORMAT
  Return a generator matrix G as a {self.k} x {self.n} numpy array of 0/1 (dtype uint8), or a list
  of such matrices. The code is the row space of G over GF(2).

HARD CONSTRAINTS
  - G must be exactly {self.k} rows by {self.n} columns.
  - G must have FULL RANK {self.k} over GF(2) (rows linearly independent mod 2). A rank-deficient G
    is not a [{self.n},{self.k}] code at all and scores nothing.
  - Deterministic: seed any RNG explicitly.

THE VIEW THAT MAKES THIS TRACTABLE
  Think of G by its COLUMNS, not its rows: G is {self.n} column vectors c_1..c_{self.n} drawn from
  GF(2)^{self.k}. For a nonzero message u, the codeword uG has weight |{{j : <u, c_j> = 1}}|, so

        d = min over the 2^{self.k}-1 nonzero u of  #{{ j : <u, c_j> = 1 }}

  i.e. you are choosing {self.n} points in GF(2)^{self.k} so that NO nonzero hyperplane through the
  origin contains too many of them. Full rank == the columns span GF(2)^{self.k}. Good codes in this
  regime come from structure, not from randomness: quasi-cyclic / circulant blocks, cyclic codes
  from a generator polynomial, shortened or punctured BCH codes, and column multisets with a
  prescribed symmetry group (that is how the current record here was set).
"""

    # -- verifier ----------------------------------------------------------- #
    def _invalid(self, reason: str) -> Verdict:
        return Verdict(
            valid=False,
            score=NEG_INF,
            detail={"reason": reason, "target": [self.n, self.k], "n": self.n, "k": self.k},
        )

    def verify(self, candidate: Candidate) -> Verdict:
        """Exact, deterministic, and total: a mutated program can emit literally anything, so
        every failure path returns Verdict(valid=False, score=-inf) instead of raising."""
        try:
            return self._verify(candidate)
        except Exception as exc:  # noqa: BLE001 — garbage in, verdict out. Never raise.
            return self._invalid(f"candidate rejected: {type(exc).__name__}: {exc}")

    def _verify(self, candidate: Candidate) -> Verdict:
        g = self._coerce(candidate)
        if isinstance(g, str):
            return self._invalid(g)

        valid, reason = is_valid_generator(g)
        if not valid:
            return self._invalid(f"invalid generator: {reason}")

        # The exact verifier: exhaustive enumeration of all 2^k - 1 nonzero codewords.
        result = compute_min_distance(g)
        d = result.get("min_distance")
        witness = result.get("witness_codeword")
        message = result.get("witness_message")
        if d is None or witness is None or message is None:
            return self._invalid(
                str(result.get("notes") or "verifier could not certify a distance")
            )

        # Independent re-check on a second code path. A distance whose achieving codeword does not
        # reproduce is a bug, not a result — and a result without a witness is not a result.
        recheck = recompute_distance_of_witness(g, message)
        if (
            not recheck.get("ok")
            or recheck.get("weight") != int(d)
            or list(recheck.get("recomputed_codeword") or []) != list(witness)
        ):
            return self._invalid(
                "witness re-check failed: the claimed distance has no valid achieving codeword"
            )

        return Verdict(
            valid=True,
            score=float(d),
            detail={
                "target": [self.n, self.k],
                "n": self.n,
                "k": self.k,
                "min_distance": int(d),
                "witness_codeword": list(witness),
                "witness_message": list(message),
                "witness_weight": int(recheck["weight"]),
                "generator_matrix": g.astype(int).tolist(),
                "method": "exhaustive_enumeration",
                "verification_method": "exhaustive_enumeration",
                "construction_source": "program_candidate",
                "codewords_enumerated": result.get("codewords_enumerated"),
                "best_known_lower": self._baseline(),
                "best_known_upper": self.record.upper,
                "best_known_source": self.record.lower_source,
                "beats_best_known": int(d) > self._baseline(),
            },
        )

    def _coerce(self, candidate: Candidate) -> np.ndarray | str:
        """Coerce a candidate to a k x n GF(2) matrix, or return a reason string."""
        if candidate is None:
            return "candidate is None"
        if isinstance(candidate, (str, bytes, dict, set)):
            return f"candidate is a {type(candidate).__name__}, not a matrix"
        arr = np.asarray(candidate)
        if arr.dtype == object or arr.dtype.kind not in "biuf":
            return f"candidate is not a numeric matrix (dtype {arr.dtype})"
        if arr.ndim != 2:
            return f"candidate must be a 2-D k x n matrix, got ndim={arr.ndim}"
        if arr.shape != (self.k, self.n):
            return f"candidate shape {tuple(arr.shape)} != target ({self.k}, {self.n})"
        if arr.dtype.kind == "f":
            if not np.all(np.isfinite(arr)):
                return "candidate contains NaN/inf"
            if not np.all(arr == np.floor(arr)):
                return "candidate has non-integer entries; a GF(2) matrix is required"
        if np.any(np.abs(arr) > 2**31):
            return "candidate has out-of-range entries"
        return np.asarray(arr, dtype=np.int64) % 2

    # -- the record --------------------------------------------------------- #
    def _baseline(self) -> int:
        """The number to beat. Conservative by construction: the HIGHEST claim from any source
        wins, so we can never bank a 'record' against a baseline that is too low."""
        claims = [self.record.lower]
        repo = best_known_distance(self.n, self.k)
        if repo is not None:
            claims.append(int(repo))
        return max(claims)

    def best_known(self) -> float:
        return float(self._baseline())

    def is_improvement(self, verdict: Verdict) -> bool:
        try:
            return self._is_improvement(verdict)
        except Exception:  # noqa: BLE001 — an unjudgeable verdict is not an improvement.
            return False

    def _is_improvement(self, verdict: Verdict) -> bool:
        if verdict is None or not getattr(verdict, "valid", False):
            return False
        score = getattr(verdict, "score", NEG_INF)
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False
        detail: dict[str, Any] = getattr(verdict, "detail", None) or {}

        # Judge only candidates verified for THIS cell.
        if list(detail.get("target") or []) != [self.n, self.k]:
            return False
        d = detail.get("min_distance")
        if not isinstance(d, int) or isinstance(d, bool):
            return False
        if int(float(score)) != d:  # score and evidence must agree
            return False

        # 1. The existing anti-self-deception guards. A distance read off a table, or asserted
        #    without an achieving codeword, is not a computation.
        if is_table_lookup_evidence(detail):
            return False
        # 2. Reproducing a tabulated code is a rediscovery, not a discovery.
        if trivial_rediscovery(detail, self.n, self.k, d):
            return False
        # 3. The witness must independently re-check against the submitted generator.
        if not self._witness_holds(detail, d):
            return False
        # 4. It must STRICTLY beat the sourced record.
        if d <= self._baseline():
            return False
        # 5. It must not exceed the PROVEN upper bound. Beating a published nonexistence proof is
        #    overwhelmingly likelier to be our bug than a discovery — refuse it and let a human look.
        if d > self.record.upper:
            return False
        return True

    def _witness_holds(self, detail: dict[str, Any], d: int) -> bool:
        """Re-derive the result from the evidence alone, exactly as a third party would."""
        g = detail.get("generator_matrix")
        message = detail.get("witness_message")
        witness = detail.get("witness_codeword")
        if g is None or message is None or witness is None:
            return False
        arr = np.asarray(g)
        if arr.ndim != 2 or arr.shape != (self.k, self.n):
            return False
        valid, _ = is_valid_generator(arr)
        if not valid:
            return False
        recheck = recompute_distance_of_witness(arr, list(message))
        if not recheck.get("ok"):
            return False
        if recheck.get("weight") != d:
            return False
        if list(recheck.get("recomputed_codeword") or []) != list(witness):
            return False
        # The witness proves d <= its weight. That the code has NO lighter codeword is what the
        # exhaustive enumeration establishes — so re-run it; it is cheap and it is the whole claim.
        recomputed = compute_min_distance(arr)
        return recomputed.get("min_distance") == d

    # -- seeds -------------------------------------------------------------- #
    def seed_programs(self) -> list[str]:
        return [
            _PRELUDE.format(n=self.n, k=self.k) + body
            for body in (
                _SEED_NAMED_CONSTRUCTIONS,
                _SEED_COLUMN_MULTISET,
                _SEED_QUASI_CYCLIC,
                _SEED_CYCLIC_POLYNOMIAL,
                _SEED_GREEDY_COLUMN_SWAP,
                _SEED_SYSTEMATIC_RANDOM,
            )
        ]


# --------------------------------------------------------------------------- #
# Seed programs — self-contained Python (numpy + stdlib only), each defines build().
# These are the LLM's mutation surface, so they are written to be READ and EDITED.
# --------------------------------------------------------------------------- #
_PRELUDE = '''\
import itertools
import time

import numpy as np

N = {n}   # block length
K = {k}   # dimension


def gf2_rank(a):
    """Rank over GF(2)."""
    a = (np.asarray(a, dtype=np.int64) % 2).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        piv = -1
        for i in range(r, rows):
            if a[i, c]:
                piv = i
                break
        if piv < 0:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] = (a[i] + a[r]) % 2
        r += 1
        if r == rows:
            break
    return r


_MSG_CACHE = {{}}


def all_messages(k):
    """All 2^k - 1 nonzero messages as a (2^k - 1) x k bit matrix. Cached."""
    m = _MSG_CACHE.get(k)
    if m is None:
        idx = np.arange(1, 2 ** k, dtype=np.int64)
        m = (idx[:, None] >> np.arange(k, dtype=np.int64)[None, :]) & 1
        _MSG_CACHE[k] = m
    return m


def min_distance(g):
    """Minimum distance = the smallest nonzero codeword weight (exact, same definition the
    verifier uses). Vectorised over all 2^k - 1 messages in one matmul — use it freely."""
    g = np.asarray(g, dtype=np.int64) % 2
    k = g.shape[0]
    weights = ((all_messages(k) @ g) % 2).sum(axis=1)
    return int(weights.min())


def take_rank_k(rows, k=K, n=N):
    """Greedily keep k independent rows (a subcode); top up with unit rows if short."""
    rows = np.asarray(rows, dtype=np.int64) % 2
    chosen = []
    for row in rows:
        if len(chosen) == k:
            break
        if gf2_rank(np.array(chosen + [row], dtype=np.int64)) == len(chosen) + 1:
            chosen.append(row)
    i = 0
    while len(chosen) < k and i < n:
        e = np.zeros(n, dtype=np.int64)
        e[i] = 1
        if gf2_rank(np.array(chosen + [e], dtype=np.int64)) == len(chosen) + 1:
            chosen.append(e)
        i += 1
    if len(chosen) != k:
        return None
    return (np.array(chosen, dtype=np.int64) % 2).astype(np.uint8)


def fit(g, n=N, k=K):
    """Adapt any generator matrix to the target [n, k] shape.

    Columns: puncture if too long, extend by repeating columns if too short.
    Rows:    take a rank-k subcode (or top up to rank k).
    Returns None if it cannot be made into a valid [n, k] generator.
    """
    g = np.asarray(g, dtype=np.int64) % 2
    if g.ndim != 2 or g.size == 0:
        return None
    cols = g.shape[1]
    if cols > n:
        g = g[:, :n]
    elif cols < n:
        extra = np.array([g[:, i % cols] for i in range(n - cols)], dtype=np.int64).T
        g = np.hstack([g, extra])
    out = take_rank_k(g, k, n)
    if out is None or out.shape != (k, n) or gf2_rank(out) != k:
        return None
    return out


def from_columns(cols, k=K, n=N):
    """Build G from an explicit list of n column vectors in GF(2)^k."""
    g = (np.array(cols, dtype=np.int64).T % 2)
    if g.shape != (k, n) or gf2_rank(g) != k:
        return None
    return g.astype(np.uint8)


def int_to_vec(c, k=K):
    """Integer 1..2^k-1 -> its binary column vector in GF(2)^k."""
    return [(c >> b) & 1 for b in range(k)]


'''

_SEED_NAMED_CONSTRUCTIONS = '''\
def hamming(r):
    """[2^r - 1, 2^r - 1 - r, 3]."""
    cols = [int_to_vec(c, r) for c in range(1, 2 ** r)]
    h = np.array(cols, dtype=np.int64).T
    ident = {1 << b for b in range(r)}
    order = [c for c in range(1, 2 ** r) if c not in ident] + [1 << b for b in range(r)]
    h = np.array([int_to_vec(c, r) for c in order], dtype=np.int64).T
    kk = 2 ** r - 1 - r
    a = h[:, :kk]
    return np.hstack([np.eye(kk, dtype=np.int64), a.T]) % 2


def extended_hamming(r):
    """[2^r, 2^r - 1 - r, 4] — Hamming plus an overall parity bit."""
    g = hamming(r)
    return np.hstack([g, (g.sum(axis=1) % 2).reshape(-1, 1)])


def simplex(r):
    """[2^r - 1, r, 2^(r-1)] — every nonzero codeword has the SAME weight."""
    return np.array([int_to_vec(c, r) for c in range(1, 2 ** r)], dtype=np.int64).T % 2


def reed_muller_rm1(m):
    """RM(1, m) = [2^m, m + 1, 2^(m-1)]."""
    pts = list(itertools.product((0, 1), repeat=m))
    rows = [np.ones(2 ** m, dtype=np.int64)]
    for i in range(m):
        rows.append(np.array([p[i] for p in pts], dtype=np.int64))
    return np.array(rows, dtype=np.int64) % 2


def repetition(n):
    """[n, 1, n]."""
    return np.ones((1, n), dtype=np.int64)


def parity_check(k):
    """[k + 1, k, 2] — G = [I_k | 1]."""
    return np.hstack([np.eye(k, dtype=np.int64), np.ones((k, 1), dtype=np.int64)])


def build():
    """The classical constructions, each ADAPTED to the target [n, k] by shortening,
    puncturing and column-extension. These are the ancestors — recombine them."""
    out = []
    for r in range(2, 8):
        for fn in (hamming, extended_hamming, simplex):
            try:
                out.append(fit(fn(r)))
            except Exception:
                pass
    for m in range(2, 8):
        try:
            out.append(fit(reed_muller_rm1(m)))
        except Exception:
            pass
    out.append(fit(repetition(N)))
    out.append(fit(parity_check(K)))
    return [g for g in out if g is not None]
'''

_SEED_COLUMN_MULTISET = '''\
def build():
    """Columns-first. G is N columns chosen from GF(2)^K \\\\ {0}; the minimum distance is

        d = min over nonzero u of  #{j : <u, c_j> = 1}

    so the game is to spread the columns so that no hyperplane through the origin swallows too
    many of them. Vary WHICH columns you take — that is the whole search."""
    out = []
    pool = list(range(1, 2 ** K))

    # (a) the first N nonzero vectors — a punctured simplex code
    out.append(from_columns([int_to_vec(c) for c in pool[:N]]))

    # (b) heaviest columns first: high-weight columns hit more hyperplanes
    by_weight = sorted(pool, key=lambda c: (-bin(c).count("1"), c))
    out.append(from_columns([int_to_vec(c) for c in by_weight[:N]]))

    # (c) evenly spaced through the pool — a crude 'spread'
    if len(pool) >= N:
        step = len(pool) // N
        picked = [pool[(i * step) % len(pool)] for i in range(N)]
        if len(set(picked)) == N:
            out.append(from_columns([int_to_vec(c) for c in picked]))

    return [g for g in out if g is not None]
'''

_SEED_QUASI_CYCLIC = '''\
def circulant(first_row, k=K):
    """k x k circulant: row i is first_row rotated right by i."""
    return np.array(
        [np.roll(np.asarray(first_row, dtype=np.int64), i) for i in range(k)],
        dtype=np.int64,
    ) % 2


def build():
    """Quasi-cyclic: G = [C_1 | C_2 | ... | C_m | R], each C a K x K circulant. QC codes hold a
    large share of the current records in this region — mutate the circulant seed rows."""
    out = []
    blocks = N // K
    rest = N - blocks * K
    seeds = (
        [1, 1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0],
    )
    for offset in range(len(seeds)):
        cols = []
        for b in range(blocks):
            row = np.zeros(K, dtype=np.int64)
            pat = seeds[(b + offset) % len(seeds)]
            for i, bit in enumerate(pat):
                if i < K:
                    row[i] = bit
            cols.append(circulant(row))
        if rest:
            tail = np.zeros((K, rest), dtype=np.int64)
            for i in range(K):
                for j in range(rest):
                    tail[i, j] = (i + j + offset) % 2
            cols.append(tail)
        if not cols:
            continue
        g = np.hstack(cols) % 2
        if g.shape == (K, N) and gf2_rank(g) == K:
            out.append(g.astype(np.uint8))
        else:
            fitted = fit(g)
            if fitted is not None:
                out.append(fitted)
    return out
'''

_SEED_CYCLIC_POLYNOMIAL = '''\
def cyclic_from_poly(poly, n=N, k=K):
    """Cyclic [n, k] code: rows are the n - deg shifts of the generator polynomial g(x),
    where deg g(x) = n - k. poly is a bit list, lowest degree first."""
    deg = len(poly) - 1
    if deg != n - k:
        return None
    rows = []
    for i in range(k):
        row = np.zeros(n, dtype=np.int64)
        for j, bit in enumerate(poly):
            row[i + j] = bit % 2
        rows.append(row)
    g = np.array(rows, dtype=np.int64) % 2
    if gf2_rank(g) != k:
        return None
    return g.astype(np.uint8)


def build():
    """Cyclic codes from a generator polynomial of degree N-K. Classic BCH-style structure:
    mutate the polynomial's coefficients (and try its reciprocal)."""
    out = []
    deg = N - K
    polys = []
    # a few dense/structured degree-(N-K) polynomials, always with constant term 1
    polys.append([1] + [(i % 2) for i in range(1, deg)] + [1])
    polys.append([1] + [(1 if i % 3 else 0) for i in range(1, deg)] + [1])
    polys.append([1] + [(1 if (i * i) % 5 < 2 else 0) for i in range(1, deg)] + [1])
    polys.append([1] * (deg + 1))
    for p in polys:
        if len(p) != deg + 1:
            continue
        g = cyclic_from_poly(p)
        if g is not None:
            out.append(g)
        g = cyclic_from_poly(list(reversed(p)))
        if g is not None:
            out.append(g)
    return out
'''

_SEED_GREEDY_COLUMN_SWAP = '''\
# Wall-clock budget. The sandbox hard-kills a slow program, and a killed program scores NOTHING,
# so a search seed must bound its own runtime rather than its iteration count — the cost of one
# iteration scales as 2^K, so a fixed iteration count that is fine at K=10 is fatal at K=16.
BUDGET_S = 5.0


def build():
    """Hill-climb the column multiset: swap one column at a time, keep the swap unless it makes
    the minimum distance worse (ties are kept, so the walk can cross plateaus).

    This is a real, deterministic baseline — the engine has to beat IT, not just beat random.
    Obvious ways to improve it: choose the column to replace from the SUPPORT of a
    minimum-weight codeword instead of uniformly at random; anneal instead of hill-climbing;
    or restart from a structured (quasi-cyclic / cyclic) code instead of a random one."""
    deadline = time.monotonic() + BUDGET_S
    rng = np.random.default_rng(20260714)

    # Lookup table: every column vector of GF(2)^K, indexed by its integer encoding.
    vecs = np.array([int_to_vec(c) for c in range(2 ** K)], dtype=np.int64)
    pool = np.arange(1, 2 ** K, dtype=np.int64)

    best_g = None
    best_d = -1
    while time.monotonic() < deadline:
        cols = rng.choice(pool, size=N, replace=True)
        g = vecs[cols].T
        while gf2_rank(g) != K and time.monotonic() < deadline:
            cols[int(rng.integers(N))] = int(rng.choice(pool))
            g = vecs[cols].T
        if gf2_rank(g) != K:
            break
        d = min_distance(g)

        while time.monotonic() < deadline:
            i = int(rng.integers(N))
            old = cols[i]
            cols[i] = int(rng.choice(pool))
            g2 = vecs[cols].T
            if gf2_rank(g2) != K:
                cols[i] = old
                continue
            d2 = min_distance(g2)
            if d2 >= d:            # accept improvements AND lateral moves
                g, d = g2, d2
            else:
                cols[i] = old

        if d > best_d:
            best_d = d
            best_g = (g % 2).astype(np.uint8)

    return [best_g] if best_g is not None else []
'''

_SEED_SYSTEMATIC_RANDOM = '''\
def build():
    """Baseline: G = [I_K | R] for pseudo-random R. Systematic form is always full rank, so this
    always produces a VALID code — it just will not produce a good one. Beat it."""
    out = []
    for seed in (1, 2, 3, 5, 8):
        rng = np.random.default_rng(seed)
        r = rng.integers(0, 2, size=(K, N - K), dtype=np.uint8)
        out.append(np.hstack([np.eye(K, dtype=np.uint8), r]) % 2)
    return out
'''


# --------------------------------------------------------------------------- #
# The recommended first cell.
#
# [32,14]: record d=8 (Cheng & Sloane 1989), proven upper bound d=9. Measured on the seed pool:
#   [32,14]  seeds already reach d=8 == the record.  One more unit closes the cell.
#   [35,10]  seeds reach d=11; record 12; a beat needs 13.  Two units to climb.
#   [36,10]  seeds reach d=11; record 13; a beat needs 14.  Three units to climb.
# So on [32,14] the search STARTS at the frontier — every candidate spent goes at the open
# question itself rather than at re-climbing to the record. It is also the only cell whose gap the
# quasi-cyclic seed already brackets.
#
# The cost: verification is 2^k, so [32,14] runs ~1 candidate/sec against the current
# coding_theory.compute_min_distance (which loops in Python over the 2^k codewords), versus ~10-20
# /sec at k=10. Vectorising that enumeration is worth 16x here and returns bit-identical values.
#
# HONESTY: an open cell means NOBODY KNOWS — it does NOT mean a better code exists. The true
# optimum may well equal the lower bound, with the upper bound merely not yet tightened. If the
# [32,14,9] code does not exist, no amount of search will find it, and the engine cannot produce
# the nonexistence proof that would close the cell the other way (that is an argument, not an
# object). Expected yield on any single cell is genuinely uncertain; the case for this target is
# the BASE RATE across many cells, not a bet on one.
# --------------------------------------------------------------------------- #
RECOMMENDED_CELL = (32, 14)


def recommended_problem() -> ECCProblem:
    """[32,14]: record d=8, proven upper bound d=9 — the seed pool already ties the record, so the
    search starts at the frontier and only needs +1 to close the cell."""
    return ECCProblem(*RECOMMENDED_CELL)
