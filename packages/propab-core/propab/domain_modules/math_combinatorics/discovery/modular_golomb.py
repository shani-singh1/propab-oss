"""
Exact CP-SAT finder for modular / cyclic Golomb rulers in Z_k (OEIS A004135 & A004136).

Where the B_3 binary-cube target (``cp_sat_finder.py``) is squeezed by the size of the
cube, the modular Golomb rulers are a **more CP-SAT-friendly** frontier: a witness is a
small subset of Z_k, the constraint is a flat "all pairwise sums distinct mod k", and the
next terms are genuinely OPEN and uncontested (frozen since Cariboni 2017-2018).

Two sequences, one model, selected by ``include_repeats``
--------------------------------------------------------
A subset ``S`` of the cyclic group ``Z_k`` is admissible iff a chosen family of pairwise
sums ``a + b (mod k)`` are pairwise DISTINCT:

  * **A004135** (``include_repeats=False``) -- sums over *distinct* pairs ``a < b`` are
    distinct mod k.  ``a(n)`` = least k admitting such an n-subset.  Known through
    n=17 (a(17)=255); a(18) open.
  * **A004136** (``include_repeats=True``) -- sums over *all* pairs ``a <= b`` (so the
    doubles ``2a`` count too) are distinct mod k -- a *perfect / modular Golomb ruler*.
    ``a(n)`` = least k.  Known through n=18 (a(18)=307); a(19) open.

Encoding (exact, by the definition)
-----------------------------------
  * one boolean ``x_r`` per residue ``r in Z_k``: ``x_r = 1`` iff r in S;
  * ``sum(x) == n`` (fixed size); ``x_0 == 1`` (translation symmetry -- S and S+t share
    the property, so WLOG ``0 in S``: a sound reduction dividing the space by k);
  * bucket every candidate pair by its sum ``(a + b) mod k``; for each sum value holding
    >= 2 pairs, half-reify a "pair present" literal (both endpoints selected) and post
    ``AddAtMostOne`` over the bucket.  A double ``(a, a)`` is "present" exactly when
    ``x_a`` is selected, so ``x_a`` itself is used as its literal.  This is exactly
    "no two admissible pairs share a sum".

Honesty
-------
Every SAT witness is re-checked by the INDEPENDENT ``is_modular_sidon`` before it is
returned (paranoid verifier, no finder-side bookkeeping trusted).  ``decide_*`` reports
sat / unsat / unknown; ``min_modular_ruler`` scans k upward and only reports
``proven_minimal`` when *every* smaller k was decisively UNSAT within budget.  No result
here is asserted as a "record": an open-term witness is an UPPER BOUND, and a claimed
minimum k needs the full "no smaller k" sweep.

Requires ``ortools`` (shared with the B_3 backend).
"""
from __future__ import annotations

import time
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Known terms (sourced from OEIS; used by the sanity gate and the registry).
# Offset 1: index -> a(n).  ``None`` marks the open next term.
# ---------------------------------------------------------------------------
A004135_TERMS: dict[int, int | None] = {
    1: 1, 2: 2, 3: 3, 4: 6, 5: 11, 6: 19, 7: 28, 8: 40, 9: 56,
    10: 72, 11: 96, 12: 114, 13: 147, 14: 178, 15: 183, 16: 252, 17: 255,
    18: None,  # OPEN
}
A004136_TERMS: dict[int, int | None] = {
    1: 1, 2: 3, 3: 7, 4: 13, 5: 21, 6: 31, 7: 48, 8: 57, 9: 73,
    10: 91, 11: 120, 12: 133, 13: 168, 14: 183, 15: 255, 16: 255,
    17: 273, 18: 307,
    19: None,  # OPEN
}


def variant_terms(include_repeats: bool) -> dict[int, int | None]:
    """Known-term table for the selected sequence (A004136 if repeats, else A004135)."""
    return A004136_TERMS if include_repeats else A004135_TERMS


def oeis_id(include_repeats: bool) -> str:
    return "A004136" if include_repeats else "A004135"


def ortools_available() -> bool:
    """True iff the CP-SAT backend can be imported in this environment."""
    try:
        import ortools.sat.python.cp_model  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Independent, deterministic witness verifier.
# ---------------------------------------------------------------------------
def is_modular_sidon(
    S: Iterable[int], k: int, *, include_repeats: bool
) -> bool:
    """
    Exact check that ``S`` is a valid modular Golomb ruler in ``Z_k``.

    Re-derives every admissible pairwise sum from scratch: no incremental state, no
    trust in any solver output.  Returns False on out-of-range residues, duplicates, or
    any two admissible pairs sharing a sum mod k.  ``include_repeats`` toggles whether
    the doubles ``2a`` participate (A004136) or only distinct pairs ``a < b`` (A004135).
    """
    pts = [int(r) for r in S]
    if k <= 0:
        return False
    if any(r < 0 or r >= k for r in pts):
        return False
    if len(set(pts)) != len(pts):
        return False
    seen: set[int] = set()
    m = len(pts)
    for i in range(m):
        start = i if include_repeats else i + 1
        ai = pts[i]
        for j in range(start, m):
            s = (ai + pts[j]) % k
            if s in seen:
                return False
            seen.add(s)
    return True


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------
def build_ruler_model(n: int, k: int, *, include_repeats: bool, fix_zero: bool = True):
    """
    Build the exact CP-SAT model for an n-subset of Z_k with distinct pairwise sums.

    Returns ``(model, x)`` where ``x[r]`` is the boolean for residue ``r``.  ``fix_zero``
    pins ``0 in S`` (sound translation reduction).  Only sum-buckets with >= 2 pairs get
    an ``AddAtMostOne`` constraint.
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{r}") for r in range(k)]
    model.Add(sum(x) == int(n))
    if fix_zero:
        model.Add(x[0] == 1)

    buckets: dict[int, list[tuple[int, int]]] = {}
    for i in range(k):
        jstart = i if include_repeats else i + 1
        for j in range(jstart, k):
            s = (i + j) % k
            buckets.setdefault(s, []).append((i, j))

    for s, pairs in buckets.items():
        if len(pairs) < 2:
            continue
        lits = []
        for (i, j) in pairs:
            if i == j:
                # double 2i: "present" exactly iff residue i selected -> use x[i].
                lits.append(x[i])
            else:
                lit = model.NewBoolVar(f"p_{s}_{i}_{j}")
                # half-reification: (x_i and x_j) => lit.
                model.AddBoolOr([x[i].Not(), x[j].Not(), lit])
                lits.append(lit)
        model.AddAtMostOne(lits)

    return model, x


def _extract(solver, x) -> list[int]:
    return [r for r in range(len(x)) if solver.Value(x[r]) == 1]


# ---------------------------------------------------------------------------
# Decision: does Z_k contain an admissible n-subset?
# ---------------------------------------------------------------------------
def decide_modular_ruler(
    n: int,
    k: int,
    *,
    include_repeats: bool,
    time_budget: float = 30.0,
    workers: int = 8,
    log: bool = False,
) -> dict[str, Any]:
    """
    Decide whether ``Z_k`` contains an admissible n-subset (exact).

    Outcomes in ``result["outcome"]``: ``"sat"`` (witness, re-verified independently),
    ``"unsat"`` (PROVEN no such subset exists in Z_k), or ``"unknown"`` (honest timeout).
    """
    from ortools.sat.python import cp_model

    seq = oeis_id(include_repeats)
    if n > k:  # cannot pick n distinct residues from k
        return {
            "sequence": seq, "n": n, "k": k, "include_repeats": include_repeats,
            "backend": "ortools_cp_sat", "status": "TRIVIAL_INFEASIBLE",
            "outcome": "unsat", "proven": True, "set": None, "elapsed": 0.0,
        }

    start = time.time()
    model, x = build_ruler_model(n, k, include_repeats=include_repeats)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_budget)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(model)
    elapsed = time.time() - start

    base = {
        "sequence": seq, "n": n, "k": k, "include_repeats": include_repeats,
        "backend": "ortools_cp_sat", "status": solver.StatusName(status),
        "elapsed": elapsed,
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        witness = _extract(solver, x)
        if not is_modular_sidon(witness, k, include_repeats=include_repeats):
            raise AssertionError("CP-SAT returned a non-admissible ruler (encoding bug)")
        if len(witness) != n:
            raise AssertionError("CP-SAT witness has the wrong size")
        return {**base, "outcome": "sat", "proven": True, "size": len(witness),
                "set": witness, "verified": True}
    if status == cp_model.INFEASIBLE:
        return {**base, "outcome": "unsat", "proven": True, "set": None}
    return {**base, "outcome": "unknown", "proven": False, "set": None}


# ---------------------------------------------------------------------------
# Minimization: least k admitting an admissible n-subset  =>  a(n).
# ---------------------------------------------------------------------------
def min_modular_ruler(
    n: int,
    *,
    include_repeats: bool,
    k_start: int | None = None,
    k_max: int,
    per_k_budget: float = 20.0,
    workers: int = 8,
) -> dict[str, Any]:
    """
    Scan k upward for the least ``k`` admitting an admissible n-subset of ``Z_k``.

    Returns the first feasible k with its (re-verified) witness.  ``proven_minimal`` is
    True only if *every* smaller k in ``[k_start, found-1]`` was decisively **UNSAT**
    within its budget -- i.e. the returned value is a genuine ``a(n)`` recomputation, not
    merely an upper bound.  If a smaller k timed out (unknown), the found k is reported as
    an upper bound with ``proven_minimal=False``.  If nothing is feasible up to ``k_max``,
    ``outcome="none"``.
    """
    lo = k_start if k_start is not None else n
    all_smaller_unsat = True
    trail: list[dict[str, Any]] = []
    for k in range(lo, k_max + 1):
        d = decide_modular_ruler(
            n, k, include_repeats=include_repeats,
            time_budget=per_k_budget, workers=workers,
        )
        trail.append({"k": k, "outcome": d["outcome"], "elapsed": round(d["elapsed"], 3)})
        if d["outcome"] == "sat":
            return {
                "sequence": oeis_id(include_repeats), "n": n,
                "include_repeats": include_repeats, "outcome": "found",
                "k": k, "set": d["set"], "size": len(d["set"]),
                "proven_minimal": all_smaller_unsat, "verified": True,
                "scan": trail,
            }
        if d["outcome"] != "unsat":  # timeout -> can no longer prove minimality
            all_smaller_unsat = False
    return {
        "sequence": oeis_id(include_repeats), "n": n,
        "include_repeats": include_repeats, "outcome": "none",
        "k": None, "set": None, "proven_minimal": False,
        "searched_up_to": k_max, "scan": trail,
    }


# ---------------------------------------------------------------------------
# Independent certification of a witness (cheap, deterministic).
# ---------------------------------------------------------------------------
def certify_modular_ruler(
    witness: Any,
    k: int,
    *,
    include_repeats: bool,
    published_best_k: int | None = None,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """
    Independently certify that ``witness`` is a valid modular Golomb ruler in ``Z_k``.

    ``witness`` may be a dict (``{"k": .., "set": [...]}``) or a bare list of residues.
    The checks are computed independently: residues in range, distinct, expected size (if
    given), and ``is_modular_sidon`` re-run from scratch.  This is a MINIMIZE problem, so
    a witness at k certifies the UPPER BOUND ``a(n) <= k``; ``beats_published`` is True
    only when ``k < published_best_k`` (a strict improvement of a published bound).  Never
    claims optimality (that needs the exhaustive "no smaller k" proof).
    """
    if isinstance(witness, dict):
        raw = witness.get("set")
        if raw is None:
            raw = witness.get("witness", [])
        declared_k = witness.get("k")
    else:
        raw = witness
        declared_k = None

    kk = int(k if k is not None else (declared_k if declared_k is not None else 0))
    S = [int(r) for r in (raw or [])]
    size = len(S)
    distinct = len(set(S)) == size
    in_range = bool(S) and kk > 0 and all(0 <= r < kk for r in S)
    size_ok = (expected_n is None) or (size == int(expected_n))
    valid = is_modular_sidon(S, kk, include_repeats=include_repeats) if (in_range and distinct) else False
    beats = (published_best_k is not None) and valid and size_ok and (kk < int(published_best_k))

    certified = bool(in_range and distinct and size_ok and valid)
    return {
        "certified": certified,
        "sequence": oeis_id(include_repeats),
        "k": kk,
        "size": size,
        "include_repeats": include_repeats,
        "published_best_k": published_best_k,
        "beats_published": beats,
        "checks": {
            "residues_in_range": in_range,
            "distinct_residues": distinct,
            "expected_size": size_ok,
            "is_modular_sidon": valid,
        },
        "note": (
            "CERTIFIED: valid modular Golomb ruler -> upper bound a(n) <= k. NOT a proof "
            "of minimality (that needs the exhaustive 'no smaller k' sweep)."
            if certified
            else "NOT certified -- see checks for the failing condition."
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: honest attempts at the open next terms.
# ---------------------------------------------------------------------------
def attempt_open_term(
    include_repeats: bool,
    *,
    k_start: int | None = None,
    k_max: int | None = None,
    per_k_budget: float = 30.0,
    workers: int = 8,
) -> dict[str, Any]:
    """
    Honest bounded attempt at the open next term (A004135 a(18) or A004136 a(19)).

    Scans k from the previous known term upward, looking for ANY admissible n-subset (an
    UPPER BOUND for the open value).  Interpret honestly: ``found`` gives an upper bound
    (with ``proven_minimal`` only if the whole prefix was UNSAT); ``none`` within budget
    is no claim at all.
    """
    terms = variant_terms(include_repeats)
    open_n = max(terms)  # the None entry -- the open index
    prev = terms[open_n - 1]  # last known term
    lo = k_start if k_start is not None else (prev + 1 if prev else open_n)
    hi = k_max if k_max is not None else lo + 40
    return min_modular_ruler(
        open_n, include_repeats=include_repeats,
        k_start=lo, k_max=hi, per_k_budget=per_k_budget, workers=workers,
    )
