"""
Exact CP-SAT backend for the maximum B_3 set in the binary cube {0,1}^n (A396704).

Where ``finder.py`` is a metaheuristic (greedy + Dynamic Local Search) that *finds*
large B_3 sets but can only *prove* optimality for tiny n via its origin-fixed DFS,
this module is an **exact decision engine** built on Google OR-tools CP-SAT. It can
actually decide the two honest questions the target poses:

  * ``decide_b3_cpsat(n, k)`` -- does a B_3 set of size >= k exist in {0,1}^n?
    Returns SAT (with a witness) or, crucially, **UNSAT** (a proof no such set
    exists). Proving UNSAT for k = published_lower_bound + 1 is what turns a
    provisional lower bound into a *proven optimum*.
  * ``max_b3_cpsat(n)`` -- maximize |S| directly; ``proven_optimal`` is True only
    when CP-SAT closes the gap (status OPTIMAL) within the budget.

Encoding (exact, by the DEFINITION of B_3)
------------------------------------------
A set S subset {0,1}^n is B_3 iff all threefold multiset sums a+b+c are distinct as
integer vectors -- equivalently, **for each target sum vector t, at most one 3-element
multiset of points sums to t and is fully contained in S**. So:

  * one boolean ``x_p`` per cube point p (128 for n=7): x_p = 1 iff p in S;
  * group every 3-multiset {i,j,k} of points by its integer sum vector; for each sum
    bucket that holds >= 2 multisets, introduce a "triple present" literal per
    multiset with the half-reification (all its points selected) => literal, then
    an ``AddAtMostOne`` over the bucket. A bucket with one multiset needs no
    constraint. This is exactly "no two distinct triples share a sum".
  * cardinality: ``sum(x) >= k`` (decision) or ``Maximize(sum(x))``.

Symmetry
--------
Every nonempty B_3 set can be translated by coordinate complementations (XOR by a
fixed mask -- a hyperoctahedral symmetry) so that the origin is a member. We fix
``x_origin = 1``. This is *sound* (removes no optimum) and divides the space by 2^n,
which both accelerates search and keeps the UNSAT proof valid.

Honesty
-------
Every SAT witness is re-checked with the INDEPENDENT ``is_B3`` before it is returned;
a size is never reported without a witness that passes the paranoid verifier. This
module makes **no record claim**: a candidate size-17 witness must still be routed
through ``certify_b3_record`` and re-verified by a human/independent certifier, and an
UNSAT result is reported as "proven optimal for this n" only because CP-SAT returned
INFEASIBLE for size k+1 with a sound (origin-fixing) symmetry reduction.

Requires ``ortools`` (added to requirements/base.txt + pyproject.toml).
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Sequence

from propab.domain_modules.math_combinatorics.discovery.verifier import is_B3

Vector = tuple[int, ...]


def ortools_available() -> bool:
    """True iff the CP-SAT backend can be imported in this environment."""
    try:
        import ortools.sat.python.cp_model  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------
def _sum_buckets(pts: Sequence[Vector], n: int) -> dict[Vector, list[tuple[int, ...]]]:
    """Map each threefold-sum vector -> list of point-index 3-multisets summing to it."""
    buckets: dict[Vector, list[tuple[int, ...]]] = {}
    P = len(pts)
    for i in range(P):
        a = pts[i]
        for j in range(i, P):
            b = pts[j]
            ab = tuple(a[t] + b[t] for t in range(n))
            for k in range(j, P):
                c = pts[k]
                s = tuple(ab[t] + c[t] for t in range(n))
                buckets.setdefault(s, []).append((i, j, k))
    return buckets


def build_b3_model(
    n: int,
    *,
    min_size: int | None = None,
    fix_origin: bool = True,
    maximize: bool = False,
):
    """
    Build the exact CP-SAT model for B_3 sets in {0,1}^n.

    Returns ``(model, x, pts)`` where ``x[i]`` is the boolean for ``pts[i]``.

    ``min_size`` adds ``sum(x) >= min_size`` (the decision constraint); ``maximize``
    adds ``Maximize(sum(x))``. ``fix_origin`` pins the origin into S (sound
    hyperoctahedral reduction). Only sum-buckets with >= 2 triples get constraints.
    """
    from ortools.sat.python import cp_model

    pts = list(itertools.product((0, 1), repeat=n))
    idx = {p: i for i, p in enumerate(pts)}
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(pts))]

    if fix_origin:
        model.Add(x[idx[tuple([0] * n)]] == 1)

    buckets = _sum_buckets(pts, n)
    for s, tris in buckets.items():
        if len(tris) < 2:
            continue
        literals = []
        for tri in tris:
            distinct = sorted(set(tri))
            lit = model.NewBoolVar(f"t_{'_'.join(map(str, s))}__{'_'.join(map(str, tri))}")
            # Half-reification: (all points of the triple selected) => lit.
            # Clause: (~p1 | ~p2 | ... | lit).
            model.AddBoolOr([x[d].Not() for d in distinct] + [lit])
            literals.append(lit)
        # At most one triple per sum may be fully present  <=>  sums are distinct.
        model.AddAtMostOne(literals)

    if min_size is not None:
        model.Add(sum(x) >= int(min_size))
    if maximize:
        model.Maximize(sum(x))

    return model, x, pts


def _extract_witness(solver, x, pts) -> list[Vector]:
    return [pts[i] for i in range(len(pts)) if solver.Value(x[i]) == 1]


# ---------------------------------------------------------------------------
# Decision:  does a B_3 set of size >= k exist?
# ---------------------------------------------------------------------------
def decide_b3_cpsat(
    n: int,
    k: int,
    *,
    time_budget: float = 60.0,
    workers: int = 8,
    fix_origin: bool = True,
    log: bool = False,
) -> dict[str, Any]:
    """
    Decide whether {0,1}^n contains a B_3 set of size >= k (exact).

    Outcomes in ``result["outcome"]``:
      * ``"sat"``     -- a witness exists; ``result["set"]`` is a verified B_3 set
                         of size >= k (re-checked with the independent ``is_B3``).
      * ``"unsat"``   -- PROVEN: no B_3 set of size k exists in {0,1}^n. If k =
                         published_best + 1 this proves the published value optimal.
      * ``"unknown"`` -- the budget expired before CP-SAT decided (honest timeout).

    ``proven`` is True only for the ``sat``/``unsat`` cases (CP-SAT is a complete
    solver; UNSAT is a real infeasibility proof under the sound origin-fixing).
    """
    from ortools.sat.python import cp_model

    start = time.time()
    model, x, pts = build_b3_model(n, min_size=k, fix_origin=fix_origin)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_budget)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(model)
    elapsed = time.time() - start

    base = {
        "n": n,
        "k": k,
        "backend": "ortools_cp_sat",
        "status": solver.StatusName(status),
        "elapsed": elapsed,
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        witness = _extract_witness(solver, x, pts)
        if not is_B3(witness):  # paranoia: never report an unverified witness
            raise AssertionError("CP-SAT returned a non-B_3 set (encoding bug)")
        if len(witness) < k:
            raise AssertionError("CP-SAT witness smaller than requested size")
        return {
            **base,
            "outcome": "sat",
            "proven": True,
            "size": len(witness),
            "set": [list(v) for v in witness],
            "verified": True,
        }
    if status == cp_model.INFEASIBLE:
        return {**base, "outcome": "unsat", "proven": True, "set": None}
    return {**base, "outcome": "unknown", "proven": False, "set": None}


# ---------------------------------------------------------------------------
# Optimization:  maximize |S|.
# ---------------------------------------------------------------------------
def max_b3_cpsat(
    n: int,
    *,
    time_budget: float = 60.0,
    workers: int = 8,
    fix_origin: bool = True,
    log: bool = False,
) -> dict[str, Any]:
    """
    Maximize |S| for a B_3 set in {0,1}^n with CP-SAT.

    ``proven_optimal`` is True only if CP-SAT closes the optimality gap (status
    OPTIMAL) inside the budget; otherwise the returned ``size`` is the best B_3 set
    found so far (still independently verified) and ``proven_optimal`` is False.
    """
    from ortools.sat.python import cp_model

    start = time.time()
    model, x, pts = build_b3_model(n, maximize=True, fix_origin=fix_origin)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_budget)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(model)
    elapsed = time.time() - start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "n": n,
            "backend": "ortools_cp_sat",
            "status": solver.StatusName(status),
            "size": 0,
            "set": [],
            "proven_optimal": False,
            "verified": True,
            "elapsed": elapsed,
        }

    witness = _extract_witness(solver, x, pts)
    if not is_B3(witness):
        raise AssertionError("CP-SAT returned a non-B_3 set (encoding bug)")
    return {
        "n": n,
        "backend": "ortools_cp_sat",
        "status": solver.StatusName(status),
        "size": len(witness),
        "set": [list(v) for v in witness],
        "proven_optimal": status == cp_model.OPTIMAL,
        "bound": solver.BestObjectiveBound(),
        "verified": True,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Convenience: the A396704 n=7 attempt (size-17 decision).
# ---------------------------------------------------------------------------
def attempt_a7_size17(
    *, time_budget: float = 1800.0, workers: int = 8, log: bool = False
) -> dict[str, Any]:
    """
    Attempt the target: decide whether a size-17 B_3 set exists in {0,1}^7.

    Thin wrapper over ``decide_b3_cpsat(7, 17, ...)``. Interpret the outcome:
      * ``sat``     -> a candidate 17-witness (run ``certify_b3_record`` on it and
                       report the JSON; do NOT assert "record" here);
      * ``unsat``   -> a(7) = 16 is PROVEN optimal (a genuine first computation);
      * ``unknown`` -> honest timeout, no claim.
    """
    return decide_b3_cpsat(7, 17, time_budget=time_budget, workers=workers, log=log)
