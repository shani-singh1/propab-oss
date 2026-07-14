"""
Binary linear code constructors, GF(2) linear algebra, and real minimum-distance
computation for the coding_theory domain.

Everything here computes on an ACTUAL k x n generator matrix over GF(2). The
minimum distance is never read from a table when a witness is claimed: it is
computed by exhaustive enumeration of the 2^k - 1 nonzero codewords (small k) and
the achieving codeword is emitted as an independently re-checkable witness.

Domain-independent core never imports this module; all coding-theory specifics
live here and in the sibling modules.
"""
from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np

# Exhaustive 2^k enumeration is only honest/feasible for small k. Above this we
# refuse to certify a distance by enumeration (we would otherwise silently lie).
#
# Production cap: 2^16-1 = 65 535 nonzero codewords is the worst-case enumeration,
# which completes in ~2s (pure-Python loop + numpy matmul per message). k=18 costs
# ~7s and k=20 ~25s — an unacceptable per-verification runaway for a deployed
# service — so we cap at 16. Every named construction this module builds
# (Hamming/simplex/RM/repetition/parity) and every Golay anchor has k <= 12, well
# within the cap; larger table anchors (e.g. length-31 BCH with k=21/26) remain
# available as best-known REFERENCE values for rediscovery rejection but are never
# enumerated here.
MAX_EXHAUSTIVE_K = 16

# Best-known LOWER bounds d for binary linear [n, k] codes. Values are taken from
# the Brouwer / Grassl tables of bounds on the minimum distance of linear codes
# (M. Grassl, "Bounds on the minimum distance of linear codes and quantum codes",
# http://www.codetables.de, accessed 2026-07; and A. E. Brouwer's classic tables,
# reproduced in the codetables.de server). They are used ONLY for rediscovery
# rejection and honest computed-vs-known reporting — never reported as if they
# were computed by this module.
#
# Every entry below either (a) coincides with a provably-optimal family this
# module can construct and independently distance-check (repetition d=n; parity
# d=2; identity d=1; Hamming (2^r-1, 2^r-1-r, 3); extended Hamming
# (2^r, 2^r-1-r, 4); simplex (2^r-1, r, 2^{r-1}); first-order Reed-Muller
# RM(1,m) = (2^m, m+1, 2^{m-1})), or (b) satisfies the Griesmer bound as a
# necessary consistency check (see tests). Anchors verified against constructors:
# [7,4,3], [8,4,4], [15,4,8] simplex, [15,11,3] Hamming, [16,5,8] RM(1,4),
# [16,11,4] extended Hamming.
#   key: (n, k) -> best-known minimum distance d
BEST_KNOWN_TABLE: dict[tuple[int, int], int] = {
    # k = 1 (repetition codes): d = n (provably optimal; constructed here)
    (2, 1): 2, (3, 1): 3, (4, 1): 4, (5, 1): 5, (6, 1): 6, (7, 1): 7,
    (8, 1): 8, (9, 1): 9, (10, 1): 10, (11, 1): 11, (12, 1): 12,
    (13, 1): 13, (14, 1): 14, (15, 1): 15, (16, 1): 16,
    # k = n-1 (single-parity-check codes): d = 2 (provably optimal; constructed here)
    (3, 2): 2, (4, 3): 2, (5, 4): 2, (6, 5): 2, (7, 6): 2, (8, 7): 2,
    (9, 8): 2, (10, 9): 2, (11, 10): 2, (12, 11): 2, (13, 12): 2, (14, 13): 2,
    (15, 14): 2, (16, 15): 2,
    # k = n (trivial / identity): d = 1 (constructed here)
    (2, 2): 1, (3, 3): 1, (4, 4): 1, (5, 5): 1, (6, 6): 1, (7, 7): 1,
    # small optimal codes from the Brouwer/Grassl tables (well-established)
    (4, 2): 2,
    (5, 2): 3, (5, 3): 2,
    (6, 2): 4, (6, 3): 3, (6, 4): 2,
    (7, 2): 4, (7, 3): 4, (7, 4): 3, (7, 5): 2,   # (7,4,3) = Hamming
    (8, 2): 5, (8, 3): 4, (8, 4): 4, (8, 5): 2, (8, 6): 2,  # (8,4,4) extended Hamming
    (9, 2): 6, (9, 3): 4, (9, 4): 4, (9, 5): 3, (9, 6): 2, (9, 7): 2,
    (10, 2): 6, (10, 3): 5, (10, 4): 4, (10, 5): 4, (10, 6): 3, (10, 7): 2,
    (11, 2): 7, (11, 3): 6, (11, 4): 5, (11, 5): 4, (11, 6): 4, (11, 7): 3,
    (11, 8): 2, (11, 9): 2,
    (12, 2): 8, (12, 3): 6, (12, 4): 6, (12, 5): 4, (12, 6): 4, (12, 7): 4,
    (12, 8): 3, (12, 9): 2, (12, 10): 2,
    # n = 13 (Brouwer/Grassl codetables.de)
    (13, 2): 8, (13, 3): 7, (13, 4): 6, (13, 5): 5, (13, 6): 4, (13, 7): 4,
    (13, 8): 4, (13, 9): 3, (13, 10): 2, (13, 11): 2,
    # n = 14 (Brouwer/Grassl codetables.de)
    (14, 2): 9, (14, 3): 8, (14, 4): 7, (14, 5): 6, (14, 6): 5, (14, 7): 4,
    (14, 8): 4, (14, 9): 4, (14, 10): 3, (14, 11): 2, (14, 12): 2,
    # n = 15: [15,11,3] Hamming; [15,7,5] BCH; [15,5,7]; [15,4,8] simplex;
    # plus other Brouwer/Grassl entries at this length.
    (15, 2): 10, (15, 3): 8, (15, 4): 8, (15, 5): 7, (15, 6): 6, (15, 7): 5,
    (15, 8): 4, (15, 9): 4, (15, 10): 4, (15, 11): 3, (15, 12): 2, (15, 13): 2,
    # n = 16: [16,5,8] RM(1,4); [16,11,4] extended Hamming; other Grassl entries.
    (16, 2): 10, (16, 3): 8, (16, 4): 8, (16, 5): 8, (16, 6): 6, (16, 7): 6,
    # (16, 12) was 4 — provably wrong. r = n-k = 4 parity checks, so H has 4 rows and 16 columns;
    # d >= 3 needs all columns nonzero and pairwise distinct, but GF(2)^4 has only 15 nonzero
    # vectors < 16. So d <= 2, and codetables.de reports lower = upper = 2.
    (16, 8): 5, (16, 9): 4, (16, 10): 4, (16, 11): 4, (16, 12): 2, (16, 13): 2,
    (16, 14): 2,
    # Longer well-established anchors (BCH / Golay families).
    (23, 12): 7,               # binary Golay [23,12,7]
    (24, 12): 8,               # extended binary Golay [24,12,8]
    (31, 26): 3, (31, 21): 5, (31, 16): 7,  # Hamming/BCH length 31
}


# --------------------------------------------------------------------------- #
# GF(2) linear algebra
# --------------------------------------------------------------------------- #
def as_gf2(matrix: Any) -> np.ndarray:
    """Coerce to a 2-D uint8 numpy array reduced mod 2."""
    arr = np.asarray(matrix, dtype=np.int64) % 2
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.astype(np.uint8)


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank over GF(2) via Gaussian elimination (mod 2)."""
    a = as_gf2(matrix).copy().astype(np.int64)
    rows, cols = a.shape
    rank = 0
    pivot_row = 0
    for col in range(cols):
        pivot = -1
        for r in range(pivot_row, rows):
            if a[r, col] % 2 == 1:
                pivot = r
                break
        if pivot == -1:
            continue
        a[[pivot_row, pivot]] = a[[pivot, pivot_row]]
        for r in range(rows):
            if r != pivot_row and a[r, col] % 2 == 1:
                a[r] = (a[r] + a[pivot_row]) % 2
        pivot_row += 1
        rank += 1
        if pivot_row == rows:
            break
    return rank


def is_valid_generator(matrix: np.ndarray) -> tuple[bool, str]:
    """A k x n generator matrix is valid iff it has full row rank k over GF(2)."""
    g = as_gf2(matrix)
    if g.ndim != 2 or g.size == 0:
        return False, "generator must be a non-empty 2-D matrix"
    k, n = g.shape
    if k > n:
        return False, f"k={k} exceeds n={n}; not a valid [n,k] code"
    r = gf2_rank(g)
    if r != k:
        return False, f"generator rows are linearly dependent over GF(2) (rank {r} != k {k})"
    return True, "full row rank over GF(2)"


# --------------------------------------------------------------------------- #
# Real minimum-distance computation (the checkable witness)
# --------------------------------------------------------------------------- #
def compute_min_distance(matrix: Any) -> dict[str, Any]:
    """
    Compute the TRUE minimum distance of the binary linear code generated by
    ``matrix`` by exhaustively enumerating all 2^k - 1 nonzero codewords and
    returning the minimum Hamming weight, together with an achieving codeword
    (the witness) and the message that produces it.

    For a linear code, minimum distance == minimum nonzero codeword weight, so
    this is exact. Only runs when k <= MAX_EXHAUSTIVE_K (feasible & honest).
    """
    g = as_gf2(matrix)
    k, n = g.shape
    valid, reason = is_valid_generator(g)
    result: dict[str, Any] = {
        "n": n,
        "k": k,
        "generator_matrix": g.astype(int).tolist(),
        "generator_valid": valid,
        "validity_reason": reason,
        "method": "exhaustive_enumeration",
    }
    if not valid:
        result.update(
            min_distance=None,
            witness_codeword=None,
            witness_message=None,
            enumeration_complete=False,
            notes=f"invalid generator: {reason}",
        )
        return result
    if k > MAX_EXHAUSTIVE_K:
        result.update(
            min_distance=None,
            witness_codeword=None,
            witness_message=None,
            enumeration_complete=False,
            notes=(
                f"k={k} exceeds exhaustive limit {MAX_EXHAUSTIVE_K}; refusing to "
                "certify a distance without a full enumeration (no honest witness)."
            ),
        )
        return result

    g_int = g.astype(np.int64)
    # VECTORIZED exhaustive enumeration: build every message at once, one matmul, one weight sum.
    #
    # This is the search's throughput ceiling, so the old Python loop over 2^k messages (one tiny
    # numpy call each) *was* the wall: measured 469 ms/candidate at k=14, i.e. ~2 candidates/sec, and
    # a live campaign ran at 1.7/sec — ~80% of that ceiling. Vectorizing is therefore a direct
    # search-budget multiplier, not a micro-optimisation.
    #
    # Bit-identical by construction: row i holds the binary digits of i MSB-first, which is exactly
    # `itertools.product((0, 1), repeat=k)` order, and `argmin` returns the FIRST minimum — matching
    # the loop's strict `w < min_w` (first message to achieve the minimum wins). Same distance, same
    # witness codeword, same witness message.
    num = 1 << k
    shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
    messages = ((np.arange(num, dtype=np.int64)[:, None] >> shifts[None, :]) & 1)
    codewords = (messages @ g_int) % 2
    weights = codewords.sum(axis=1)

    # The all-zero message is not a codeword we may certify; a full-rank G admits no other
    # zero-weight codeword, but guard anyway rather than trust it.
    weights = np.where(weights > 0, weights, n + 1)

    best = int(np.argmin(weights))
    min_w = int(weights[best])
    witness_cw: list[int] | None = None
    witness_msg: list[int] | None = None
    if min_w <= n:
        witness_cw = codewords[best].astype(int).tolist()
        witness_msg = messages[best].astype(int).tolist()
    else:
        min_w = n + 1

    result.update(
        min_distance=min_w if witness_cw is not None else None,
        witness_codeword=witness_cw,
        witness_message=witness_msg,
        enumeration_complete=True,
        codewords_enumerated=(2 ** k) - 1,
        notes=(
            f"exhaustive enumeration of 2^{k}-1 nonzero codewords; "
            f"min Hamming weight = {min_w}"
        ),
    )
    return result


def recompute_distance_of_witness(
    generator_matrix: Any,
    witness_message: list[int],
) -> dict[str, Any]:
    """
    Independently RE-CHECK a claimed witness: recompute the codeword from the
    generator and the claimed message, and return its weight. Used to guard
    against a bug that reports a distance without a valid achieving codeword.
    """
    g = as_gf2(generator_matrix).astype(np.int64)
    msg = np.array(list(witness_message), dtype=np.int64) % 2
    if msg.shape[0] != g.shape[0]:
        return {"ok": False, "reason": "message length != k", "weight": None}
    cw = (msg @ g) % 2
    return {
        "ok": True,
        "recomputed_codeword": cw.astype(int).tolist(),
        "weight": int(cw.sum()),
    }


# --------------------------------------------------------------------------- #
# Explicit binary linear code constructors (all return k x n GF(2) generators)
# --------------------------------------------------------------------------- #
def repetition_code(n: int) -> np.ndarray:
    """[n, 1, n] repetition code."""
    n = max(1, int(n))
    return as_gf2(np.ones((1, n), dtype=np.uint8))


def parity_check_code(k: int) -> np.ndarray:
    """[k+1, k, 2] single-parity-check code: G = [I_k | 1]."""
    k = max(1, int(k))
    g = np.hstack([np.eye(k, dtype=np.uint8), np.ones((k, 1), dtype=np.uint8)])
    return as_gf2(g)


def identity_code(k: int) -> np.ndarray:
    """[k, k, 1] trivial code (no redundancy)."""
    k = max(1, int(k))
    return as_gf2(np.eye(k, dtype=np.uint8))


def _hamming_parity_columns(r: int) -> np.ndarray:
    """All 2^r - 1 nonzero binary columns of length r (parity-check of Hamming code)."""
    cols = [
        [(c >> b) & 1 for b in range(r)]
        for c in range(1, 2 ** r)
    ]
    return np.array(cols, dtype=np.uint8).T  # shape (r, 2^r - 1)


def hamming_code(r: int) -> np.ndarray:
    """
    Binary Hamming code [2^r - 1, 2^r - 1 - r, 3] for r >= 2.

    Build H (r x n) from all nonzero length-r columns arranged so the last r
    columns form I_r, then G = [I_k | P^T] with P read off H = [P^T | I_r].
    r=3 gives the classic [7,4,3] Hamming code.
    """
    r = max(2, int(r))
    n = 2 ** r - 1
    k = n - r
    # Columns: put the r identity columns (weight-1) at the END so H = [A | I_r].
    identity_cols = {1 << b for b in range(r)}
    non_identity = [c for c in range(1, 2 ** r) if c not in identity_cols]
    ordered = non_identity + [1 << b for b in range(r)]

    def col_vec(c: int) -> list[int]:
        return [(c >> b) & 1 for b in range(r)]

    h = np.array([col_vec(c) for c in ordered], dtype=np.uint8).T  # (r, n)
    a = h[:, :k]                      # r x k  (the non-identity part)
    # H = [A | I_r]  ->  G = [I_k | A^T]
    g = np.hstack([np.eye(k, dtype=np.uint8), a.T.astype(np.uint8)])
    return as_gf2(g)


def extended_hamming_code(r: int) -> np.ndarray:
    """
    Extended Hamming code [2^r, 2^r - 1 - r, 4]: append an overall parity bit to
    each row of the Hamming generator. r=3 gives the [8,4,4] code.
    """
    g = hamming_code(r).astype(np.int64)
    parity = (g.sum(axis=1) % 2).reshape(-1, 1)
    return as_gf2(np.hstack([g, parity]))


def simplex_code(r: int) -> np.ndarray:
    """
    Binary simplex code [2^r - 1, r, 2^(r-1)] — dual of the Hamming code.
    Generator rows = the parity-check matrix of the Hamming code (all nonzero
    length-r columns). Every nonzero codeword has weight exactly 2^(r-1).
    """
    r = max(2, int(r))
    g = _hamming_parity_columns(r)  # (r, 2^r - 1) — rows are a basis
    return as_gf2(g)


def reed_muller_rm1(m: int) -> np.ndarray:
    """
    First-order Reed-Muller code RM(1, m): [2^m, m+1, 2^(m-1)].
    Rows: all-ones plus the m coordinate functions over F_2^m.
    """
    m = max(1, int(m))
    n = 2 ** m
    points = list(itertools.product((0, 1), repeat=m))
    rows = [np.ones(n, dtype=np.uint8)]
    for i in range(m):
        rows.append(np.array([p[i] for p in points], dtype=np.uint8))
    return as_gf2(np.array(rows, dtype=np.uint8))


def random_generator(n: int, k: int, seed: int = 0) -> np.ndarray:
    """
    A pseudo-random full-rank systematic generator G = [I_k | R] for [n, k].
    Systematic form guarantees full row rank (so it is always a valid code).
    """
    n = int(n)
    k = int(k)
    if k <= 0 or k > n:
        raise ValueError(f"require 0 < k <= n, got n={n}, k={k}")
    rng = np.random.default_rng(seed)
    r = rng.integers(0, 2, size=(k, n - k), dtype=np.uint8) if n > k else np.zeros((k, 0), np.uint8)
    return as_gf2(np.hstack([np.eye(k, dtype=np.uint8), r]))


# Registry of named, parameterised constructors for the verifier/router.
NAMED_CONSTRUCTORS = {
    "hamming": hamming_code,
    "extended_hamming": extended_hamming_code,
    "simplex": simplex_code,
    "repetition": repetition_code,
    "parity_check": parity_check_code,
    "identity": identity_code,
    "reed_muller": reed_muller_rm1,
}


def build_named_code(name: str, param: int) -> np.ndarray | None:
    ctor = NAMED_CONSTRUCTORS.get(name)
    if ctor is None:
        return None
    try:
        return ctor(param)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Table / rediscovery helpers
# --------------------------------------------------------------------------- #
def best_known_distance(n: int, k: int) -> int | None:
    """Best-known LOWER bound d for [n, k] from the tabulated Brouwer/Grassl range."""
    return BEST_KNOWN_TABLE.get((int(n), int(k)))


def is_table_lookup_evidence(evidence: dict[str, Any]) -> bool:
    """
    True when the evidence's distance came from a table lookup rather than a real
    computed witness — mirrors math_combinatorics' _is_table_lookup_evidence.
    """
    if evidence.get("construction_source") == "best_known_table":
        return True
    method = str(evidence.get("verification_method") or evidence.get("method") or "")
    if method in {"table_lookup", "tabulation", "best_known_table"}:
        return True
    # A distance without an achieving witness is not a real computation.
    if evidence.get("min_distance") is not None and not evidence.get("witness_codeword"):
        return True
    return False


def trivial_rediscovery(evidence: dict[str, Any], n: int, k: int, computed_d: int | None) -> bool:
    """
    A confirmed result is a trivial rediscovery when it merely reproduces (or falls
    below) the best-known tabulated distance for [n, k], or when its distance came
    from a table lookup instead of a computed witness.
    """
    if is_table_lookup_evidence(evidence):
        return True
    known = best_known_distance(n, k)
    if known is None or computed_d is None:
        # No table entry: cannot call it a rediscovery from the table alone.
        return False
    # Meets-or-below best-known => rediscovery (not a novel improvement).
    return computed_d <= known


# --------------------------------------------------------------------------- #
# Hypothesis parsing
# --------------------------------------------------------------------------- #
_NK_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\]")
_N_RE = re.compile(r"\bn\s*=\s*(\d+)", re.I)
_K_RE = re.compile(r"\bk\s*=\s*(\d+)", re.I)
_D_RE = re.compile(r"\b(?:d|distance|minimum\s+distance)\s*(?:=|>=|≥|of|is|at least)\s*(\d+)", re.I)


def parse_code_params(statement: str) -> dict[str, int | None]:
    """Parse [n, k, d] (or n=, k=, d=) from hypothesis text."""
    s = statement or ""
    n = k = d = None
    m = _NK_RE.search(s)
    if m:
        n = int(m.group(1))
        k = int(m.group(2))
        if m.group(3):
            d = int(m.group(3))
    if n is None:
        mn = _N_RE.search(s)
        if mn:
            n = int(mn.group(1))
    if k is None:
        mk = _K_RE.search(s)
        if mk:
            k = int(mk.group(1))
    if d is None:
        md = _D_RE.search(s)
        if md:
            d = int(md.group(1))
    return {"n": n, "k": k, "d": d}


def parse_construction_name(statement: str, methodology: str = "") -> str | None:
    """Identify a named-construction request from the text/methodology."""
    s = f"{statement} {methodology}".lower()
    if "extended hamming" in s or "extended-hamming" in s:
        return "extended_hamming"
    if "hamming" in s:
        return "hamming"
    if "simplex" in s:
        return "simplex"
    if "reed" in s and "muller" in s:
        return "reed_muller"
    if "repetition" in s or "repeat code" in s:
        return "repetition"
    if "parity" in s and "check" in s:
        return "parity_check"
    return None
