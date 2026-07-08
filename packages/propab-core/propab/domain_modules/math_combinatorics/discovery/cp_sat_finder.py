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
The B_3 property is invariant under the full hyperoctahedral group B_n (bit
permutations x coordinate complementations, |B_n| = n! * 2^n). Two reductions are
applied, both *sound* (they retain at least one maximum B_3 set per symmetry orbit,
so they never remove an optimum and keep every UNSAT proof valid):

  * **origin fixing** -- every nonempty B_3 set can be complemented (XOR by a fixed
    mask) so the origin is a member; we fix ``x_origin = 1``. Divides by 2^n.
  * **lex-leader symmetry breaking** (``lex_leader=True``, default) -- for each
    *generator* g of B_n (adjacent bit transpositions + single-coordinate
    complementations) we constrain the selected set's indicator vector, read in
    point-index order, to be lexicographically **>=** its image under g. The global
    lex-largest set in each orbit satisfies *all* such constraints simultaneously
    (hence any subset of them), so keeping only the generator constraints is sound;
    it forces CP-SAT to explore essentially one canonical representative per orbit.
    Lex-max also *implies* the origin is selected (position 0 is the origin and 1 > 0
    lexicographically), so the two reductions compose cleanly. Posting only the
    ``2n-1`` generator constraints (partial lex-leader) keeps the model small while
    cutting the hyperoctahedral symmetry that dominates the n=6/7 search.

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


def _hyperoctahedral_generators(pts: Sequence[Vector], n: int) -> list[list[int]]:
    """
    Index-permutations of the cube points for a lex-leader breaking set of B_n.

    Returns one list ``perm`` per group element with ``perm[i]`` = index of
    ``g(pts[i])``. The set is deliberately *richer* than a minimal generating set --
    posting a lex-leader constraint for more elements directly forbids more symmetric
    images, which is what actually shrinks the CP-SAT search (a minimal 2n-1 generator
    set proved too weak to close n=6). Any subset of B_n is sound for lex-leader (the
    orbit's global lex-max satisfies them all), so we include:

      * every coordinate transposition  ``tau_{a,b}``  (C(n,2) of them),
      * every single-coordinate complementation  ``c_a``  (n of them), and
      * every product  ``c_a . tau_{a,b}``  (complement one swapped coordinate).

    ``perm[i] = idx[g(pts[i])]`` makes ``x[perm[i]]`` the indicator of ``g^{-1}(S)``,
    so ``x >=_lex [x[perm[i]]]`` is the lex-leader constraint for the element
    ``g^{-1}`` -- still a member of B_n, so the whole set is a sound lex-leader subset
    (transpositions and complementations are involutions, the products supply their
    remaining inverses; either way each posted constraint is genuine lex-leader).
    """
    import itertools as _it

    idx = {p: i for i, p in enumerate(pts)}

    def _perm(fn) -> list[int]:
        return [idx[fn(p)] for p in pts]

    gens: list[list[int]] = []

    def _swap(p, a, b):
        q = list(p)
        q[a], q[b] = q[b], q[a]
        return tuple(q)

    def _flip(p, a):
        q = list(p)
        q[a] ^= 1
        return tuple(q)

    for a, b in _it.combinations(range(n), 2):  # all coordinate transpositions
        gens.append(_perm(lambda p, a=a, b=b: _swap(p, a, b)))
    for a in range(n):  # all single-coordinate complementations
        gens.append(_perm(lambda p, a=a: _flip(p, a)))
    for a, b in _it.combinations(range(n), 2):  # complement-one-then-swap products
        gens.append(_perm(lambda p, a=a, b=b: _flip(_swap(p, a, b), a)))
    return gens


def _add_lex_ge(model, a: Sequence, b: Sequence) -> None:
    """
    Post the lexicographic constraint ``a >=_lex b`` on two equal-length boolean
    vectors (``a[0]`` most significant).

    Standard prefix-equality encoding: ``eq`` tracks "all earlier positions equal";
    while the prefix is equal we require ``a_t >= b_t``. Positions where ``a_t`` and
    ``b_t`` are the *same* variable (fixed points of the generator) are skipped -- they
    are trivially equal and cannot break the tie. Sound and complete for lex order.
    """
    eq_prev = None  # None == constant True (prefix all-equal so far)
    for t in range(len(a)):
        at, bt = a[t], b[t]
        if at is bt:  # identical variable: equal, ordering trivially holds, eq unchanged
            continue
        if eq_prev is None:
            model.Add(at >= bt)
        else:
            model.Add(at >= bt).OnlyEnforceIf(eq_prev)
        same = model.NewBoolVar("")  # same <=> (a_t == b_t)
        model.Add(at == bt).OnlyEnforceIf(same)
        model.Add(at + bt == 1).OnlyEnforceIf(same.Not())
        if eq_prev is None:
            eq_prev = same
        else:
            eq_next = model.NewBoolVar("")  # eq_next <=> eq_prev AND same
            model.AddBoolAnd([eq_prev, same]).OnlyEnforceIf(eq_next)
            model.AddBoolOr([eq_prev.Not(), same.Not()]).OnlyEnforceIf(eq_next.Not())
            eq_prev = eq_next


def build_b3_model(
    n: int,
    *,
    min_size: int | None = None,
    fix_origin: bool = True,
    lex_leader: bool = True,
    maximize: bool = False,
):
    """
    Build the exact CP-SAT model for B_3 sets in {0,1}^n.

    Returns ``(model, x, pts)`` where ``x[i]`` is the boolean for ``pts[i]``.

    ``min_size`` adds ``sum(x) >= min_size`` (the decision constraint); ``maximize``
    adds ``Maximize(sum(x))``. ``fix_origin`` pins the origin into S and ``lex_leader``
    adds the partial lex-leader constraints for the hyperoctahedral generators (both
    sound reductions; see the module docstring). Only sum-buckets with >= 2 triples get
    constraints.
    """
    from ortools.sat.python import cp_model

    pts = list(itertools.product((0, 1), repeat=n))
    idx = {p: i for i, p in enumerate(pts)}
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(len(pts))]

    if fix_origin:
        model.Add(x[idx[tuple([0] * n)]] == 1)

    if lex_leader:
        # Require x >=_lex g(x) for each generator g of B_n. g(S) has indicator
        # g(x)[i] = x[perm[i]] (generators are involutions), so we compare x against
        # the relabeled vector [x[perm[i]] for i in ...]. Retains the orbit's global
        # lex-maximum (a maximum B_3 set) -> sound.
        for perm in _hyperoctahedral_generators(pts, n):
            _add_lex_ge(model, x, [x[perm[i]] for i in range(len(pts))])

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
    lex_leader: bool = True,
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
    model, x, pts = build_b3_model(n, min_size=k, fix_origin=fix_origin, lex_leader=lex_leader)
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
    lex_leader: bool = True,
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
    model, x, pts = build_b3_model(n, maximize=True, fix_origin=fix_origin, lex_leader=lex_leader)
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
