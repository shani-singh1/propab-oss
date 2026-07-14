r"""Erdős problem #143 — a BOUNDED SIDE-RUN. Read the honesty section before reading anything else.

The problem ($500, Erdős). Let A ⊂ (1, ∞) be a countably infinite set of REALS such that

    for all x ≠ y in A and all integers k ≥ 1:      |k·x − y| ≥ 1                        (SEP)

Does (SEP) force sparsity — in particular, must  Σ_{x∈A} 1/(x log x)  converge?

===============================================================================================
WHAT THIS MODULE CAN AND CANNOT DO  (this text is copied verbatim into every exported bundle)
===============================================================================================
A true counterexample to #143 is an INFINITE set with a DIVERGENT sum. **A finite search can never
settle this problem.** No output of this module is a solution, a refutation, or a proof, and none of
it should ever be reported as one.

The most this target can produce is:

  (i)  EVIDENCE about the shape of the objective: the maximum partial sum achievable at a given N,
       and how that maximum grows in N. A plateau is consistent with the conjecture being TRUE. A
       maximum that keeps climbing like log log N would be a hint worth chasing — and nothing more.
  (ii) A candidate construction PATTERN which a human (or a prover) would still have to prove works
       for an infinite set. The pattern is the deliverable, not the number.

And the prior is strongly against us. Every known partial result points TOWARD the conjecture:
Erdős (1935) proved the integer case (Σ 1/(a log a) is bounded on primitive sets); Behrend and
Erdős–Sárközy–Szemerédi sharpened it; Lichtman (2023) proved the sharp form; and
Koukoulopoulos–Lamzouri–Lichtman recently proved Σ_{x ∈ A, x < n} 1/x = o(log n) for the real case.
**The refutation branch is probably empty.** This is a bounded side-bet with a small expected value,
run because the verifier is cheap — not a headline.

Reporting rule for anything this module produces: say exactly what was measured — *the maximum
partial sum found at N, and its growth trend in N* — and nothing else.

===============================================================================================
THE MATH THAT MAKES THE VERIFIER CHEAP AND EXACT
===============================================================================================
(SEP) quantifies over infinitely many k, but on a bounded set only finitely many k can bind. Three
elementary reductions collapse it to O(1) exact comparisons per pair:

  L1 (only the smaller element multiplies). Let 1 < x < y. For k ≥ 2, k·y − x ≥ 2y − x > x > 1, so
     |k·y − x| > 1 automatically. The k = 1 case, |y − x| ≥ 1, is the same constraint as k = 1 in
     the other direction. Hence (SEP) for the pair {x, y} reduces to |k·x − y| ≥ 1 for all k ≥ 1.

  L2 (only the nearest multiple binds). k·x is increasing in k, so min_{k≥1} |k·x − y| is attained
     at k = ⌊y/x⌋ or k = ⌊y/x⌋ + 1, and both are ≥ 1 because y/x > 1. So the pair is checked with
     exactly TWO comparisons — no scan over k at all.
     (The brief's bound k ≤ ⌈y/x⌉ is correct but scans O(y/x) values of k; L2 is O(1).)

  L3 (elements below 2 are self-destructive — a THEOREM, not an assumption). If x ∈ A with
     1 < x < 2, then the multiples {k·x} have spacing x < 2, so every real y > x lies within x/2 < 1
     of some k·x, violating (SEP). Hence **|A| ≥ 2 ⟹ A ⊂ [2, ∞)**.
     Consequence, and the reason `min_size = 2` is enforced: for a SINGLETON there is no pair, so
     (SEP) is vacuous and 1/(x log x) → ∞ as x → 1⁺. The finite objective is therefore UNBOUNDED via
     the degenerate set {1+ε} — a boundary artifact that says nothing about #143, which is a
     property of the TAIL of an infinite set. Singletons are excluded by construction; without that
     exclusion the search "wins" instantly with junk.

  L4 (the integer case is exactly primitivity). For integers x, y, k, the quantity |k·x − y| is a
     non-negative integer, so |k·x − y| ≥ 1 ⟺ y ≠ k·x. Quantified over k ≥ 1 this says: no element
     of A is a multiple of another. **On integers, (SEP) ⟺ A is a primitive set.** This is what
     gives us a real, sourced baseline (see `best_known`), and it is why the primes are a valid seed.

Complexity: O(m²) exact rational comparisons for a set of size m (2 per pair, by L2) — independent
of N. Cross-checking against the literal O(m² · N/x_min) scan is available and is enabled for small
sets; the two must agree.

Floating point: **validity is decided in exact rational arithmetic and never in floats.** Elements
are parsed to `fractions.Fraction` (ints, "7/5", "2.4", [num, den] are exact; a float is converted
to its exact binary value, which is the number actually stored). A float check with any tolerance is
*lenient* — it accepts a set that misses (SEP) by less than the tolerance — so it can screen but can
never certify. `verify_float()` exists only as that screen, with FLOAT_TOL = 1e-9 documented, and
`tests/evolve/test_erdos143.py` exhibits a set that the float screen calls valid and the exact
verifier correctly calls INVALID (it violates (SEP) by exactly 1e-12). The exact path is authority.

The score, by contrast, is a float: Σ 1/(x log x) is transcendental and cannot be exact. It is
summed with `math.fsum` (exactly-rounded). So: **exact validity, floating objective** — the claim
("this set satisfies (SEP)") is exact; only its measurement is numeric.
"""
from __future__ import annotations

import math
from typing import Any

from ..ledger import Record
from ..problem import Candidate, Verdict
from ..program import Program

__all__ = [
    "Erdos143Problem",
    "ERDOS_PRIMITIVE_CONSTANT",
    "HONESTY_STATEMENT",
    "partial_sum_trend",
    "trend_report",
]

# Σ_p 1/(p log p) ≈ 1.6366. Lichtman, "A proof of the Erdős primitive set conjecture",
# Annals of Mathematics 197 (2023) 1013-1041 (arXiv:2202.02384): this is the SUPREMUM of
# Σ_{a∈A} 1/(a log a) over all primitive sets A of integers > 1 — i.e. over all INTEGER sets
# satisfying (SEP), by L4. It is *not* known to bound real sets: that is precisely #143.
ERDOS_PRIMITIVE_CONSTANT = 1.6366

# A claimed improvement must beat the baseline by more than float noise.
IMPROVEMENT_MARGIN = 1e-9

HONESTY_STATEMENT = """\
THIS IS NOT A SOLUTION TO ERDŐS #143, AND NO FINITE SEARCH CAN BE ONE.

A counterexample to #143 is an INFINITE set of reals with a DIVERGENT sum Σ 1/(x log x). This target
searches FINITE sets. It therefore measures exactly one thing, and claims exactly one thing:

    the largest partial sum Σ_{x∈A} 1/(x log x) we could achieve with a finite set A ⊂ [2, N]
    satisfying |k·x − y| ≥ 1 for all x ≠ y in A and all integers k ≥ 1,

together with how that maximum grows as N grows. Nothing in this bundle settles, refutes, or makes
progress on the conjecture itself. A construction found here would still have to be proved to work
for an infinite set — by a human — before it meant anything.

The prior is against the refutation branch. Erdős (1935) proved the integer analogue is bounded;
Lichtman (2023) proved the sharp constant Σ_p 1/(p log p) ≈ 1.6366; Koukoulopoulos–Lamzouri–Lichtman
proved Σ_{x∈A, x<n} 1/x = o(log n) for the real case. All of it points toward the conjecture being
TRUE. A plateau in the numbers below is the EXPECTED outcome and is evidence FOR the conjecture, not
a failure of the search.
"""


# --------------------------------------------------------------------------- #
# The verifier. This exact source is what judges every candidate (it is exec'd below), what the
# fingerprint hashes, and what ships in the publishable bundle. One source of truth: if the code
# that judged differed by one byte from the code we publish, the fingerprint would be a lie.
# --------------------------------------------------------------------------- #
_VERIFIER_BASE = r'''"""Standalone verifier — Erdős #143, finite sets satisfying the separation condition.

Stdlib only. Imports nothing from propab. Reads as a literal transcription of the condition:

    A is valid  <=>  for all x != y in A and all integers k >= 1:  |k*x - y| >= 1

Validity is decided in EXACT rational arithmetic (fractions.Fraction) — never in floating point.
The score, sum of 1/(x*log(x)), is necessarily floating (log is transcendental) and is summed with
math.fsum.

Read `naive_gap()` first: it is the definition, transcribed. `nearest_multiple_gap()` is the fast
path used for large sets; `verify()` cross-checks the two against each other whenever the set is
small enough, and reports a disagreement as INVALID (a verifier that contradicts itself certifies
nothing).
"""
from __future__ import annotations

import math
from fractions import Fraction

VERIFIER_ID = "erdos143-separation/1"

# --- configuration; overridden by the block appended at the end of this file --- #
N_MAX = None              # elements must lie in (1, N_MAX]; None = unbounded
MIN_SIZE = 2              # singletons excluded: (SEP) is vacuous on them and 1/(x log x) -> inf
FLOAT_TOL = 1e-9          # used ONLY by verify_float(), a screen — never an authority
CROSS_CHECK_MAX_PAIRS = 4000
CROSS_CHECK_MAX_K = 200000
# ------------------------------------------------------------------------------ #

# Lichtman (Annals 2023): over INTEGER sets satisfying the condition (= primitive sets),
# sum 1/(a log a) <= sum_p 1/(p log p) ~ 1.6366. An integer candidate scoring above this is
# mathematically impossible => it means THIS CODE IS WRONG. Flagged, not celebrated.
LICHTMAN_SUP = 1.6366


def parse_real(v):
    """Exact parse of one element.

    int / "7/5" / "2.4" / [num, den]  -> the exact rational they denote.
    float                             -> the exact binary value of that float (Fraction(float) is
                                         exact, not rounded): we verify the number actually stored,
                                         so a "violation" can never be a float artifact.
    """
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool):
        raise TypeError("bool is not a real")
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("non-finite element")
        return Fraction(v)
    if isinstance(v, str):
        return Fraction(v.strip())
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return Fraction(int(v[0]), int(v[1]))
    raise TypeError("cannot parse element: {!r}".format(v))


def gap_num_den(x, y):
    """min over integers k >= 1 of |k*x - y|, EXACTLY, as an integer fraction. Requires 1 < x < y.

    Everything here is integer cross-multiplication — no floats, and no Fraction normalisation in
    the hot path. Writing x = a/b and y = c/d with b, d > 0:

        |k*x - y|  =  |k*a*d - c*b| / (b*d)

    so the condition "gap >= 1" is EXACTLY the integer test  |k*a*d - c*b| >= b*d.

    k*x increases with k, so the multiple of x nearest to y is at k = floor(y/x) or that plus one,
    and both are >= 1 because y/x > 1. Two candidates, and we are done.

    Returns (numerator, denominator, k) — all ints, denominator > 0.
    """
    a, b = x.numerator, x.denominator
    c, d = y.numerator, y.denominator
    ad, cb = a * d, c * b
    k0 = cb // ad  # = floor(y/x)
    if k0 < 1:
        k0 = 1
    g0 = abs(k0 * ad - cb)
    g1 = abs((k0 + 1) * ad - cb)
    if g1 < g0:
        return g1, b * d, k0 + 1
    return g0, b * d, k0


def nearest_multiple_gap(x, y):
    """min over integers k >= 1 of |k*x - y| as an exact Fraction, with the k attaining it."""
    num, den, k = gap_num_den(x, y)
    return Fraction(num, den), k


def naive_gap(x, y, kmax_cap=CROSS_CHECK_MAX_K):
    """THE DEFINITION, transcribed. min over BOTH orderings and ALL binding k of |k*a - b|.

    For the pair {x, y} the condition ranges over both ordered pairs and every integer k >= 1.
    Once k*a > b + 1 the term can never violate again (k*a - b > 1 and grows with k), so no k
    beyond floor((b+1)/a) can ever violate and the scan stops there. We scan ONE k past that bound
    so that the value returned is the true min over all k >= 1 (the minimiser is at floor(b/a) or
    floor(b/a)+1, and the latter can sit just outside the violation bound) — that makes this a
    cross-check on `nearest_multiple_gap`'s VALUE, not merely on its verdict.

    Returns (gap, (base, k)), or (None, None) if the scan exceeded its cap (garbage input).
    """
    best = None
    arg = None
    for a, b in ((x, y), (y, x)):
        kmax = int((b + 1) // a) + 1
        if kmax < 2:
            kmax = 2
        if kmax > kmax_cap:
            return None, None
        for k in range(1, kmax + 1):
            d = abs(k * a - b)
            if best is None or d < best:
                best = d
                arg = (a, k)
    return best, arg


def primes_upto(n):
    """Primes <= n by sieve. The canonical primitive set (Lichtman: the extremal one)."""
    n = int(n)
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = bytearray(len(sieve[p * p :: p]))
    return [i for i in range(2, n + 1) if sieve[i]]


def known_construction(xs, n_max=None):
    """Name the construction if it is one we already knew — a trivial rediscovery, not a discovery.

    `xs` is the sorted list of exact elements. Returns a name, or None if the set is new to us.
    """
    if any(x.denominator != 1 for x in xs):
        return None
    vals = [int(x) for x in xs]
    top = int(n_max) if n_max is not None else vals[-1]
    if vals == primes_upto(vals[-1]):
        return "primes<=N (Erdos/Lichtman extremal primitive set)"
    if vals == list(range(top // 2 + 1, top + 1)):
        return "integers in (N/2, N]"
    if vals == [i for i in range(3, top + 1, 2)]:
        return "odd integers >= 3"
    return None


def _prepare(candidate):
    """Normalise a candidate to a sorted list of exact Fractions, or return (None, reason)."""
    if isinstance(candidate, dict):
        for key in ("set", "elements", "A", "candidate"):
            if key in candidate:
                candidate = candidate[key]
                break
    if isinstance(candidate, (str, bytes)) or not isinstance(candidate, (list, tuple)):
        return None, "candidate is not a list of reals"
    if len(candidate) == 0:
        return None, "empty set"
    if len(candidate) > 200000:
        return None, "candidate too large"
    xs = [parse_real(v) for v in candidate]
    if len(set(xs)) != len(xs):
        return None, "elements are not distinct"
    xs.sort()
    if xs[0] <= 1:
        return None, "every element must be > 1 (got min = {})".format(xs[0])
    if N_MAX is not None and xs[-1] > Fraction(N_MAX):
        return None, "element {} exceeds N_MAX = {}".format(xs[-1], N_MAX)
    if len(xs) < MIN_SIZE:
        return None, (
            "degenerate: |A| = {} < {}. A singleton satisfies the separation condition vacuously "
            "and 1/(x log x) -> infinity as x -> 1+, so the finite objective is unbounded for "
            "|A| = 1. That is a boundary artifact, not evidence about Erdos #143 (which is a "
            "property of the TAIL of an INFINITE set). Excluded by construction."
        ).format(len(xs), MIN_SIZE)
    return xs, None


def separation(xs, cross_check=True):
    """Decide (SEP) exactly. Returns (ok, witness).

    witness on failure: the violating pair, the k, and the exact gap.
    witness on success: the TIGHTEST constraint in the whole set (min over pairs of the nearest-
    multiple gap) — a quantified certificate of margin, re-derivable by anyone from candidate.json.
    """
    m = len(xs)
    pairs = m * (m - 1) // 2
    do_cross = cross_check and pairs <= CROSS_CHECK_MAX_PAIRS

    # Running minimum, held as an unnormalised integer fraction (mn / md) to keep the loop in ints.
    mn = None
    md = 1
    argmin = None
    for i in range(m):
        x = xs[i]
        for j in range(i + 1, m):
            y = xs[j]
            num, den, k = gap_num_den(x, y)  # exact, O(1) — lemma L2

            if do_cross:
                # The fast path must agree with the literal definition, or we certify nothing.
                ngap, _narg = naive_gap(x, y)
                if ngap is not None and ngap != Fraction(num, den):
                    return False, {
                        "reason": "verifier self-contradiction: fast path disagrees with the "
                                  "literal scan — this is a BUG in the verifier, not a result",
                        "pair": [str(x), str(y)],
                        "fast_gap": str(Fraction(num, den)),
                        "naive_gap": str(ngap),
                    }

            # "gap >= 1"  IS  "num >= den". Exact integer test; no float ever touches this.
            if num < den:
                gap = Fraction(num, den)
                return False, {
                    "reason": "separation violated",
                    "violating_pair": [str(x), str(y)],
                    "k": k,
                    "gap": str(gap),
                    "gap_float": float(gap),
                    "required": ">= 1",
                }
            # num/den < mn/md  <=>  num*md < mn*den   (all denominators positive)
            if mn is None or num * md < mn * den:
                mn, md = num, den
                argmin = [str(x), str(y), k]

    min_gap = Fraction(mn, md) if mn is not None else None
    return True, {
        "certificate": "min over all pairs (x<y) of min_{k>=1} |k*x - y|",
        "min_gap": str(min_gap) if min_gap is not None else None,
        "min_gap_float": float(min_gap) if min_gap is not None else None,
        "min_slack": str(min_gap - 1) if min_gap is not None else None,
        "tightest_pair_and_k": argmin,
        "cross_checked_against_naive_scan": bool(do_cross),
    }


def score_of(xs):
    """Sum of 1/(x*log(x)) — the #143 objective. Float (log is transcendental); fsum-accurate."""
    terms = []
    for x in xs:
        fx = float(x)
        if fx <= 1.0:
            raise ValueError("element rounds to <= 1 in double precision; cannot score")
        terms.append(1.0 / (fx * math.log(fx)))
    return math.fsum(terms)


def verify(candidate):
    """The contract. Returns {"valid": bool, "score": float, "detail": {...}}. NEVER raises."""
    try:
        xs, err = _prepare(candidate)
        if xs is None:
            return {"valid": False, "score": float("-inf"), "detail": {"reason": err}}

        ok, witness = separation(xs)
        if not ok:
            return {"valid": False, "score": float("-inf"), "detail": witness}

        score = score_of(xs)
        sum_recip = math.fsum(1.0 / float(x) for x in xs)
        top = float(N_MAX) if N_MAX is not None else float(xs[-1])
        ratio = sum_recip / math.log(top) if top > 1.0 else float("nan")
        integral = all(x.denominator == 1 for x in xs)

        detail = {
            "verifier_id": VERIFIER_ID,
            "mode": "exact-rational",
            "n_max": N_MAX,
            "n_elements": len(xs),
            "min_element": str(xs[0]),
            "max_element": str(xs[-1]),
            "all_integers": integral,
            "objective": "sum 1/(x log x)",
            "score": score,
            "sum_1_over_x": sum_recip,
            "sum_1_over_x_div_log_N": ratio,  # the Koukoulopoulos-Lamzouri-Lichtman quantity: -> 0
            "known_construction": known_construction(xs, N_MAX),
            "witness": witness,
        }

        # Self-consistency alarm, not a discovery: Lichtman PROVED no integer set can exceed this.
        if integral and score > LICHTMAN_SUP + 1e-9:
            detail["lichtman_alarm"] = (
                "An all-integer set scored {:.6f} > sum_p 1/(p log p) ~ {}. Lichtman (Annals 2023) "
                "PROVES that is impossible. Conclusion: this verifier is buggy. Do not report this "
                "as a result.".format(score, LICHTMAN_SUP)
            )

        return {"valid": True, "score": score, "detail": detail}
    except Exception as exc:  # a mutated program can emit anything; never propagate
        return {
            "valid": False,
            "score": float("-inf"),
            "detail": {"reason": "{}: {}".format(type(exc).__name__, exc)},
        }


def verify_float(candidate, tol=FLOAT_TOL):
    """A FLOATING-POINT SCREEN. NOT AN AUTHORITY. Provided only so the difference is demonstrable.

    Accepts |k*x - y| >= 1 - tol, so it accepts sets that MISS the condition by up to `tol`. It can
    cheaply reject an obvious violation; it can never certify one that holds. `verify()` is the
    authority. See test_erdos143.py::test_float_screen_lies_where_exact_is_right.
    """
    try:
        xs, err = _prepare(candidate)
        if xs is None:
            return {"valid": False, "score": float("-inf"), "detail": {"reason": err}}
        fs = sorted(float(x) for x in xs)
        for i, x in enumerate(fs):
            for y in fs[i + 1 :]:
                k0 = max(1, int(y // x))
                for k in (k0, k0 + 1):
                    if abs(k * x - y) < 1.0 - tol:
                        return {
                            "valid": False,
                            "score": float("-inf"),
                            "detail": {"reason": "separation violated (float screen)",
                                       "violating_pair": [x, y], "k": k, "tol": tol},
                        }
        return {"valid": True, "score": score_of(xs), "detail": {"mode": "float-screen", "tol": tol}}
    except Exception as exc:
        return {"valid": False, "score": float("-inf"),
                "detail": {"reason": "{}: {}".format(type(exc).__name__, exc)}}
'''


def _verifier_source(n_max: int | None, min_size: int) -> str:
    """The base verifier plus its configuration. Different N ⇒ different claim ⇒ different pin."""
    return (
        _VERIFIER_BASE
        + "\n\n# --- configuration for this target instance (overrides the defaults above) --- #\n"
        + f"N_MAX = {n_max!r}\n"
        + f"MIN_SIZE = {min_size!r}\n"
    )


def _load(source: str) -> dict[str, Any]:
    ns: dict[str, Any] = {"__name__": "erdos143_verifier"}
    exec(compile(source, "<erdos143-verifier>", "exec"), ns)  # noqa: S102 — our own pinned source
    return ns


# --------------------------------------------------------------------------- #
# The Problem
# --------------------------------------------------------------------------- #
class Erdos143Problem:
    """Erdős #143 as a finite-search `Problem`. A BOUNDED SIDE-BET — see the module docstring.

    candidate: a list of reals in (1, n_max]. Elements may be ints, decimal strings ("2.4"),
               rational strings ("7/5"), [num, den] pairs, or floats. All are parsed EXACTLY.
    valid:     the separation condition (SEP) holds for every pair and every integer k ≥ 1.
    score:     Σ 1/(x log x) — the partial sum whose finiteness is the open question.
    """

    def __init__(self, n_max: int = 1000, min_size: int = 2) -> None:
        if n_max <= 2:
            raise ValueError("n_max must exceed 2 (L3: |A| >= 2 forces A ⊂ [2, ∞))")
        self.n_max = int(n_max)
        self.min_size = int(min_size)
        self.name = f"erdos143-N{self.n_max}"
        self._source = _verifier_source(self.n_max, self.min_size)
        self._ns = _load(self._source)
        self._best_known: float | None = None

    # ---- the exact code that judges, and its pin ---- #
    def verifier_source(self) -> str:
        """The standalone verifier source — byte-for-byte what `verify()` runs and what we ship."""
        return self._source

    def verifier_fingerprint(self) -> str:
        from ..ledger_impl import verifier_fingerprint

        return verifier_fingerprint(self._source)

    # ---- Problem protocol ---- #
    def describe(self) -> str:
        return f"""\
ERDŐS PROBLEM #143 (finite-search proxy, N = {self.n_max}).

Find a finite set A of REAL numbers in (1, {self.n_max}] maximising

    score(A) = Σ_{{x ∈ A}} 1 / (x · log x)

subject to the SEPARATION CONDITION, which must hold exactly:

    for all x ≠ y in A and ALL integers k ≥ 1:      |k·x − y| ≥ 1

Candidate format: `build()` returns a list of elements (or a list of such lists). Elements may be
ints, floats, decimal strings "2.4", rational strings "7/5", or [numerator, denominator] pairs.
Rationals/strings are parsed EXACTLY — prefer them: a construction that only works in floating point
does not work. Return the list SORTED ascending.

Facts you should exploit (all proved, all checkable):
- |A| ≥ 2 forces every element ≥ 2. (If x < 2 then the multiples k·x have spacing < 2, so every
  y > x sits within x/2 < 1 of one of them.) Sets of size 1 are REJECTED as degenerate.
- On INTEGERS the condition is exactly "no element divides another" (a primitive set), because
  |k·x − y| is a non-negative integer. The primes are therefore valid, and Lichtman (2023) proved
  they are the best possible integer set: Σ_p 1/(p log p) ≈ {ERDOS_PRIMITIVE_CONSTANT}.
- So an integer construction CANNOT beat {ERDOS_PRIMITIVE_CONSTANT}. To improve on the baseline you
  must use genuinely NON-INTEGER reals, whose extra freedom is exactly what makes #143 open.
- Including 2 forces every other element to be an odd integer (a real at distance ≥ 1 from every even
  integer must be the midpoint of a gap). Small elements are worth the most (1/(x log x) is
  decreasing) but constrain everything above them the most. That tension IS the problem.

Baseline to beat: Σ_{{p ≤ {self.n_max}}} 1/(p log p) = {self.best_known():.6f} (the primes).
Reproducing the primes, the odd integers, or the integers in (N/2, N] is a REDISCOVERY and scores as
no improvement. Something genuinely new must be non-integral.

Honesty: no finite set can settle #143 (a counterexample is INFINITE and divergent). We are
measuring the achievable partial sum at N and its growth in N. Nothing more is being claimed."""

    def verify(self, candidate: Candidate) -> Verdict:
        """Cheap, exact, deterministic, never raises. O(m²) exact-rational comparisons."""
        try:
            out = self._ns["verify"](candidate)
        except Exception as exc:  # the standalone verifier already traps; belt and braces
            return Verdict(valid=False, score=float("-inf"),
                           detail={"reason": f"{type(exc).__name__}: {exc}"})
        if not out.get("valid"):
            return Verdict(valid=False, score=float("-inf"), detail=out.get("detail", {}))
        return Verdict(valid=True, score=float(out["score"]), detail=out.get("detail", {}))

    def best_known(self) -> float:
        """Σ_{p ≤ N} 1/(p log p): the primes' partial sum at N.

        Why this baseline (there is no canonical table for this problem — being explicit as required):
          * By L4 the condition restricted to integers is exactly primitivity, so every integer
            primitive set is a legal candidate and vice versa.
          * Lichtman, "A proof of the Erdős primitive set conjecture", Annals of Math. 197 (2023)
            1013-1041 (arXiv:2202.02384), PROVES that the primes maximise Σ 1/(a log a) over all
            primitive sets: the supremum is Σ_p 1/(p log p) ≈ 1.6366.
          * So the primes are the strongest construction anyone has for the integer half of this
            problem, they are sourced to a theorem rather than to a model's memory, and they are
            reproducible here in one sieve.

        Two caveats we do not paper over:
          1. Lichtman's theorem bounds the FULL sum over a primitive set. It does not prove that the
             primes maximise the PARTIAL sum at every finite N, so beating this number with some
             other integer set at some N would be a (minor) finding, not a contradiction.
          2. For genuinely REAL sets nothing at all is proven — that is the open problem. A real set
             beating this baseline is the only outcome here that would be interesting, and it would
             still not settle #143.
        """
        if self._best_known is None:
            primes = self._ns["primes_upto"](self.n_max)
            self._best_known = math.fsum(1.0 / (p * math.log(p)) for p in primes)
        return self._best_known

    def best_known_source(self) -> dict[str, str]:
        """Citation for `best_known()`, carried into every exported bundle."""
        return {
            "baseline": f"sum_{{p <= {self.n_max}}} 1/(p log p) = {self.best_known():.9f}",
            "construction": "the primes up to N, recomputed by sieve inside verifier.py",
            "why_it_is_the_right_baseline": (
                "On integers the separation condition is exactly primitivity (|k*x - y| is a "
                "non-negative integer, so it is >= 1 iff y is not a multiple of x). Lichtman proved "
                "the primes are the extremal primitive set."
            ),
            "citation": (
                "J. D. Lichtman, 'A proof of the Erdos primitive set conjecture', "
                "Annals of Mathematics 197 (2023) 1013-1041, arXiv:2202.02384"
            ),
            "supremum_over_integer_sets": str(ERDOS_PRIMITIVE_CONSTANT),
            "status": (
                "PROVEN for integer sets; NOTHING is proven for real sets — that is Erdos #143 "
                "itself, and it is why this baseline is a reference point and not an upper bound."
            ),
        }

    def is_improvement(self, verdict: Verdict) -> bool:
        """A genuine beat of the primes baseline. Rejects rediscovery and rejects our own bugs."""
        if not verdict.valid:
            return False
        detail = verdict.detail or {}
        if detail.get("known_construction"):
            return False  # reproducing the primes / odds / (N/2, N] is not a discovery
        if detail.get("lichtman_alarm"):
            return False  # an "impossible" score means our verifier is wrong, not that we won
        return verdict.score > self.best_known() + IMPROVEMENT_MARGIN

    def seed_programs(self) -> list[str]:
        """Known-ish constructions. Evolution works by recombining these — they are not the answer."""
        n = self.n_max
        return [
            _SEED_PRIMES.format(n=n),
            _SEED_HALF_INTERVAL.format(n=n),
            _SEED_ODD_INTEGERS.format(n=n),
            _SEED_GREEDY_RATIONAL.format(n=n, q=4),
            _SEED_PRIMES_PLUS_REALS.format(n=n, q=6),
            _SEED_HALF_SHIFT.format(n=n),
            _SEED_GEOMETRIC.format(n=n),
        ]

    # ---- ledger glue ---- #
    def make_record(
        self,
        candidate: Candidate,
        verdict: Verdict,
        program: Program | None = None,
        *,
        best_known: float | None = None,
        notes: str = "",
    ) -> Record:
        """Build a publishable `Record`: witness + program + the pin on the verifier that judged it.

        `witness["limitations"]` carries HONESTY_STATEMENT, which `export_publishable()` prints at
        the very top of the bundle README. The caveat travels with the claim, by construction.
        """
        code = program.code if program is not None else ""
        pid = program.id if program is not None else "unknown"
        return Record(
            problem=self.name,
            score=float(verdict.score),
            best_known_at_time=float(self.best_known() if best_known is None else best_known),
            candidate=candidate,
            witness={
                **(verdict.detail or {}),
                "verifier_source": self._source,  # the ledger re-checks and then blob-stores this
                "best_known_source": self.best_known_source(),
                "limitations": HONESTY_STATEMENT,
                "measured": (
                    f"the maximum partial sum sum 1/(x log x) achieved by a finite separated set "
                    f"in (1, {self.n_max}] — NOT anything about the infinite conjecture"
                ),
            },
            program_code=code,
            program_id=pid,
            generation=program.generation if program is not None else 0,
            verifier_fingerprint=self.verifier_fingerprint(),
            notes=notes,
        )


# --------------------------------------------------------------------------- #
# Seed programs — plain source, `def build()`, stdlib only, deterministic.
# --------------------------------------------------------------------------- #
_SEED_PRIMES = '''\
"""The primes <= N. Valid because |k*p - q| is a positive integer for distinct primes (no prime is a
multiple of another). By Lichtman (2023) this is the OPTIMAL integer construction — the baseline."""


def build():
    n = {n}
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\\x00\\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = bytearray(len(sieve[p * p :: p]))
    return [i for i in range(2, n + 1) if sieve[i]]
'''

_SEED_HALF_INTERVAL = '''\
"""Integers in (N/2, N]. Valid: no element divides another (2x > N >= y for integers). Classic, but
WEAK — the sum is ~ log(2)/log(N) -> 0, so this seed exists to be beaten and recombined."""


def build():
    n = {n}
    return list(range(n // 2 + 1, n + 1))
'''

_SEED_ODD_INTEGERS = '''\
"""Odd integers >= 3. INVALID as-is (3 divides 9), included deliberately: the mutation operator gets
a near-miss to repair — e.g. keep the odd numbers that are not multiples of a smaller kept one."""


def build():
    n = {n}
    return list(range(3, n + 1, 2))
'''

_SEED_GREEDY_RATIONAL = '''\
"""Greedy over a rational grid with step 1/q, scanning upward from 2.

Small elements are worth the most (1/(x log x) is decreasing), so take them first. Exact arithmetic
throughout — no floats — so what is emitted is exactly what the verifier sees.

(Empirically this reproduces the PRIMES: greedy-from-2 is the sieve of Eratosthenes. It is here as a
mutation base — the interesting variants start somewhere other than 2, or skip an element.)"""

from fractions import Fraction


def separated(chosen, x):
    """Exact: |k*lo - hi| >= 1 for all k >= 1, by integer cross-multiplication."""
    xa, xb = x.numerator, x.denominator
    for c in chosen:
        ca, cb = c.numerator, c.denominator
        if c < x:
            a, b, e, d = ca, cb, xa, xb   # lo = c, hi = x
        else:
            a, b, e, d = xa, xb, ca, cb   # lo = x, hi = c
        ad, eb = a * d, e * b
        k0 = eb // ad
        if k0 < 1:
            k0 = 1
        den = b * d
        if abs(k0 * ad - eb) < den or abs((k0 + 1) * ad - eb) < den:
            return False
    return True


def build():
    n, q = {n}, {q}
    chosen = []
    x = Fraction(2)
    step = Fraction(1, q)
    while x <= n:
        if separated(chosen, x):
            chosen.append(x)
        x += step
    return [str(c) for c in chosen]
'''

_SEED_PRIMES_PLUS_REALS = '''\
"""Start from the primes (the proven-best INTEGER set) and greedily bolt on NON-INTEGER reals.

This is the ONLY direction that can possibly beat the baseline: Lichtman's theorem forbids any
integer set from exceeding sum_p 1/(p log p), so any extra score must come from genuine reals.

It adds NOTHING, and the reason is a theorem, not a search failure: 2 is in the set, and a real at
distance >= 1 from every multiple of 2 must be the midpoint of a gap between consecutive even
integers — i.e. an odd integer. So ANY valid set containing 2 is integer-only. To use reals at all
you must first drop 2 — which costs 1/(2 log 2) = 0.72, about half the total. That trade is the
whole problem in miniature, and it is the mutation this seed exists to provoke."""

from fractions import Fraction


def separated(chosen, x):
    """Exact: |k*lo - hi| >= 1 for all k >= 1, by integer cross-multiplication."""
    xa, xb = x.numerator, x.denominator
    for c in chosen:
        ca, cb = c.numerator, c.denominator
        if c < x:
            a, b, e, d = ca, cb, xa, xb
        else:
            a, b, e, d = xa, xb, ca, cb
        ad, eb = a * d, e * b
        k0 = eb // ad
        if k0 < 1:
            k0 = 1
        den = b * d
        if abs(k0 * ad - eb) < den or abs((k0 + 1) * ad - eb) < den:
            return False
    return True


def build():
    n, q = {n}, {q}
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\\x00\\x00"
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = bytearray(len(sieve[p * p :: p]))
    chosen = [Fraction(i) for i in range(2, n + 1) if sieve[i]]

    x = Fraction(2)
    step = Fraction(1, q)
    while x <= n:
        if x.denominator != 1 and separated(chosen, x):
            chosen.append(x)
        x += step
    chosen.sort()
    return [str(c) for c in chosen]
'''

_SEED_HALF_SHIFT = '''\
"""Half-integers n + 1/2 for N/2 < n <= N. A genuinely NON-INTEGER valid family — the only kind that
can ever beat the primes (Lichtman forbids integer sets from doing so).

Valid: for x = a+1/2 < y = b+1/2 in the range, y/x < 2 so only k = 1, 2 can bind. k=1 gives
y - x >= 1. k=2 gives 2x - y = 2a - b + 1/2 >= 3/2, because 2a > N >= b forces 2a - b >= 1."""

from fractions import Fraction


def build():
    n = {n}
    return [str(Fraction(2 * a + 1, 2)) for a in range(n // 2 + 1, n)]
'''

_SEED_GEOMETRIC = '''\
"""Geometric family x_i = (N/8) * (5/2)^i. Q-structured: the ratios are never near-integers, so the
multiples of a smaller element never land near a larger one — the failure mode the condition punishes.

The base must scale with N. The multiples of a base b are spaced b apart, so a SMALL fixed base
sprays a dense grid across (1, N] and eventually lands within 1 of every large element: with base 4
this family is already invalid by N = 200 (4*39 = 156 sits 0.25 from 156.25). That obstruction — a
small element's multiple-grid poisoning everything above it — is the whole difficulty of #143, and
it is why the sum cannot simply be made large. Sparse family, small sum: a structural probe."""

from fractions import Fraction


def build():
    n = {n}
    r = Fraction(5, 2)
    out = []
    x = Fraction(n, 8)
    while x <= n:
        out.append(str(x))
        x *= r
    return out
'''


# --------------------------------------------------------------------------- #
# The growth harness — the ONLY thing this target is entitled to report.
# --------------------------------------------------------------------------- #
def partial_sum_trend(
    ns: list[int],
    *,
    problem_factory: Any = None,
) -> list[dict[str, Any]]:
    """Max achievable partial sum at each N, over the seed constructions. One row per (N, seed).

    This is the measurement — "how large can Σ 1/(x log x) get on a separated set in (1, N], and does
    that maximum plateau or climb as N grows?" — and it is ALL this target measures.
    """
    factory = problem_factory or Erdos143Problem
    rows: list[dict[str, Any]] = []
    for n in ns:
        prob = factory(n_max=n)
        for src in prob.seed_programs():
            ns_: dict[str, Any] = {"__name__": "erdos143_seed"}
            try:
                exec(compile(src, "<seed>", "exec"), ns_)  # noqa: S102 — our own seed sources
                cand = ns_["build"]()
            except Exception as exc:
                rows.append({"N": n, "seed": _seed_name(src), "valid": False,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            v = prob.verify(cand)
            rows.append({
                "N": n,
                "seed": _seed_name(src),
                "valid": v.valid,
                "score": v.score if v.valid else None,
                "size": (v.detail or {}).get("n_elements"),
                "sum_1_over_x": (v.detail or {}).get("sum_1_over_x"),
                "sum_1_over_x_div_log_N": (v.detail or {}).get("sum_1_over_x_div_log_N"),
                "all_integers": (v.detail or {}).get("all_integers"),
                "known_construction": (v.detail or {}).get("known_construction"),
                "baseline_primes": prob.best_known(),
                "beats_baseline": bool(v.valid and prob.is_improvement(v)),
                "reason": None if v.valid else (v.detail or {}).get("reason"),
            })
    return rows


def _seed_name(src: str) -> str:
    first = src.strip().splitlines()[0].strip().strip('"')
    return first[:60] if first else "seed"


def trend_report(rows: list[dict[str, Any]]) -> str:
    """Plain-text report. States what was measured and refuses to imply anything else."""
    out = [HONESTY_STATEMENT, "", "MAX ACHIEVABLE PARTIAL SUM AT N (over the seed constructions)", ""]
    out.append(f"{'N':>8} {'best score':>12} {'primes':>10} {'best seed':>52} {'beats?':>7}")
    by_n: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_n.setdefault(r["N"], []).append(r)
    for n in sorted(by_n):
        valid = [r for r in by_n[n] if r.get("valid")]
        if not valid:
            out.append(f"{n:>8} {'(none valid)':>12}")
            continue
        best = max(valid, key=lambda r: r["score"])
        out.append(
            f"{n:>8} {best['score']:>12.6f} {best['baseline_primes']:>10.6f} "
            f"{best['seed'][:52]:>52} {'YES' if best['beats_baseline'] else 'no':>7}"
        )
    out += [
        "",
        "Reading this table: a maximum that flattens as N grows is EVIDENCE CONSISTENT WITH the",
        "conjecture being TRUE (the sum converges). It is not a proof of anything, and a maximum that",
        "kept climbing would be a hint to chase, not a refutation. A finite search cannot settle #143.",
    ]
    return "\n".join(out)
