"""
Smart finder for B_3 sets in the binary cube {0,1}^n (OEIS A396704).

Three cooperating engines, all sharing the cheap incremental threefold-sum index:

1. ``max_b3_branch_and_bound`` -- exact, origin-fixed DFS. For tiny n it proves the
   optimum (a(0..4) = 1,2,3,4,6). It exploits the complementation subgroup of the
   hyperoctahedral group (WLOG 0 in S) to divide the tree by 2^n.

2. ``_grow_maximal`` -- randomized greedy growth to a maximal B_3 set, using the
   ``B3Index`` incremental unique-sum index (add/remove in O(|S|^2), never a full
   rescan). Good first incumbents; reaches a(5)=8, a(6)=11.

3. ``_dls_repair`` -- fixed-size Dynamic Local Search. It holds exactly k vectors
   and drives the weighted collision penalty to zero by out/in swaps, escaping
   plateaus by bumping the weight of persistently-violated sum-keys (the standard
   robust metaheuristic for hard constraint problems). This is what crosses the
   deep local optima that plain greedy/ILS get stuck on (a(6)=11, a(7)>=16).

``find_max_b3`` orchestrates them and re-verifies every returned witness with the
INDEPENDENT ``is_B3`` before reporting a size -- a size is never reported without a
witness that passes the paranoid verifier. Optional ``seed_witnesses`` (found by
this same finder, see ``known_witnesses.py``) warm-start the n=7 search so the hard
values are reproduced fast and deterministically in CI while remaining fully
witness-backed.
"""
from __future__ import annotations

import itertools
import random
import time
from typing import Any, Iterable, Sequence

from propab.domain_modules.math_combinatorics.discovery.verifier import is_B3
from propab.domain_modules.math_combinatorics.discovery.symmetry import canonical_form

Vector = tuple[int, ...]


# ---------------------------------------------------------------------------
# Incremental unique-sum index (invariant: current members form a B_3 set).
# ---------------------------------------------------------------------------
class B3Index:
    """
    Incremental threefold-sum index over a set that stays B_3.

    ``add``/``remove`` touch only the O(|S|^2) triples that involve the changed
    vector, not the whole set. ``can_add(v)`` reports whether v keeps S a B_3 set,
    and ``blockers(v)`` returns the members whose removal would let v in (used by
    the directed perturbation). Invariant while B_3: every stored sum has count 1.
    """

    def __init__(self) -> None:
        self.members: list[Vector] = []
        self.sums: dict[Vector, tuple[Vector, Vector, Vector]] = {}

    def _pairs_with(self, v: Vector):
        ext = self.members + [v]
        L = len(ext)
        for i in range(L):
            a = ext[i]
            for j in range(i, L):
                yield a, ext[j]

    @staticmethod
    def _key(v: Vector, a: Vector, b: Vector) -> Vector:
        return tuple(v[t] + a[t] + b[t] for t in range(len(v)))

    def can_add(self, v: Vector) -> bool:
        seen: set[Vector] = set()
        for a, b in self._pairs_with(v):
            k = self._key(v, a, b)
            if k in self.sums or k in seen:
                return False
            seen.add(k)
        return True

    def add(self, v: Vector) -> None:
        for a, b in self._pairs_with(v):
            self.sums[self._key(v, a, b)] = tuple(sorted((v, a, b)))
        self.members.append(v)

    def remove(self, v: Vector) -> None:
        self.members.remove(v)
        for a, b in self._pairs_with(v):  # members now excludes v; ext == old set
            self.sums.pop(self._key(v, a, b), None)

    def blockers(self, v: Vector) -> set[Vector]:
        """Members that must leave for v to be addable (may over-approximate)."""
        blk: set[Vector] = set()
        new: dict[Vector, tuple[Vector, Vector]] = {}
        for a, b in self._pairs_with(v):
            k = self._key(v, a, b)
            if k in self.sums:
                for m in self.sums[k]:
                    if m != v:
                        blk.add(m)
            elif k in new:
                pa, pb = new[k]
                for m in (a, b, pa, pb):
                    if m != v:
                        blk.add(m)
            else:
                new[k] = (a, b)
        return blk


# ---------------------------------------------------------------------------
# Count/penalty index for fixed-size Dynamic Local Search.
# ---------------------------------------------------------------------------
class CollisionCountIndex:
    """
    Threefold-sum multiplicity index with a WEIGHTED collision penalty.

    Unlike ``B3Index`` this permits collisions (used at a fixed size k that may not
    yet be B_3). ``collisions`` is the raw count sum_key max(count-1, 0); the set is
    B_3 iff it is 0. ``penalty`` is the weighted version sum_key w[key]*max(count-1,0)
    that DLS minimizes; ``bump_weights`` raises w on currently-violated keys to
    reshape the landscape and escape local optima. add/remove are O(|S|^2).
    """

    def __init__(self) -> None:
        self.members: list[Vector] = []
        self.count: dict[Vector, int] = {}
        self.weight: dict[Vector, int] = {}
        self.collisions = 0
        self.penalty = 0

    def _pairs_with(self, v: Vector):
        ext = self.members + [v]
        L = len(ext)
        for i in range(L):
            a = ext[i]
            for j in range(i, L):
                yield a, ext[j]

    @staticmethod
    def _key(v: Vector, a: Vector, b: Vector) -> Vector:
        return tuple(v[t] + a[t] + b[t] for t in range(len(v)))

    def _w(self, k: Vector) -> int:
        return self.weight.get(k, 1)

    def add(self, v: Vector) -> None:
        for a, b in self._pairs_with(v):
            k = self._key(v, a, b)
            c = self.count.get(k, 0)
            w = self._w(k)
            self.collisions += (c) - max(c - 1, 0)          # count c -> c+1
            self.penalty += w * ((c) - max(c - 1, 0))
            self.count[k] = c + 1
        self.members.append(v)

    def remove(self, v: Vector) -> None:
        self.members.remove(v)
        for a, b in self._pairs_with(v):
            k = self._key(v, a, b)
            c = self.count.get(k, 0)
            w = self._w(k)
            self.collisions += max(c - 2, 0) - max(c - 1, 0)  # count c -> c-1
            self.penalty += w * (max(c - 2, 0) - max(c - 1, 0))
            if c - 1 <= 0:
                self.count.pop(k, None)
            else:
                self.count[k] = c - 1

    def bump_weights(self) -> None:
        """Increase weight on every currently-violated sum-key (DLS smoothing)."""
        for k, c in self.count.items():
            if c >= 2:
                self.weight[k] = self._w(k) + 1
                self.penalty += (c - 1)


# ---------------------------------------------------------------------------
# Exact branch-and-bound (proves optimum for tiny n).
# ---------------------------------------------------------------------------
def max_b3_branch_and_bound(n: int, time_limit: float = 5.0) -> tuple[list[Vector], bool]:
    """
    Origin-fixed DFS for the maximum B_3 set in {0,1}^n.

    Returns (best_set, complete). ``complete`` is True only if the whole tree was
    searched within ``time_limit`` (then the size is provably optimal). WLOG the
    origin is a member (complementation symmetry), which divides the search by 2^n.
    """
    pts = list(itertools.product((0, 1), repeat=n))
    origin = tuple([0] * n)
    cands = [p for p in pts if p != origin]
    idx = B3Index()
    idx.add(origin)
    best: list[list[Vector]] = [[origin]]
    best_size = [1]
    complete = [True]
    start = time.time()

    def rec(pos: int) -> None:
        if time.time() - start > time_limit:
            complete[0] = False
            return
        cur = len(idx.members)
        if cur + (len(cands) - pos) <= best_size[0]:
            return
        if pos == len(cands):
            if cur > best_size[0]:
                best_size[0] = cur
                best[0] = list(idx.members)
            return
        v = cands[pos]
        if idx.can_add(v):
            idx.add(v)
            if len(idx.members) > best_size[0]:
                best_size[0] = len(idx.members)
                best[0] = list(idx.members)
            rec(pos + 1)
            idx.remove(v)
            if not complete[0]:
                return
        rec(pos + 1)

    rec(0)
    return best[0], complete[0]


# ---------------------------------------------------------------------------
# Randomized greedy growth to a maximal B_3 set.
# ---------------------------------------------------------------------------
def _grow_maximal(idx: B3Index, cands: list[Vector], rng: random.Random) -> None:
    order = cands[:]
    rng.shuffle(order)
    inset = set(idx.members)
    for v in order:
        if v not in inset and idx.can_add(v):
            idx.add(v)
            inset.add(v)


def _greedy_max_set(n: int, rng: random.Random, restarts: int = 30) -> list[Vector]:
    pts = list(itertools.product((0, 1), repeat=n))
    origin = tuple([0] * n)
    best: list[Vector] = []
    for _ in range(restarts):
        idx = B3Index()
        idx.add(origin)
        _grow_maximal(idx, [p for p in pts if p != origin], rng)
        if len(idx.members) > len(best):
            best = list(idx.members)
    return best


# ---------------------------------------------------------------------------
# Fixed-size Dynamic Local Search (the workhorse for hard values).
# ---------------------------------------------------------------------------
def _dls_repair(
    n: int,
    k: int,
    time_budget: float,
    rng: random.Random,
    warm: Sequence[Vector] | None = None,
    sample_in: int = 32,
    bump_stall: int = 30,
) -> tuple[list[Vector] | None, int]:
    """
    Drive a fixed-size-k set to 0 collisions via weighted min-conflicts.

    Each step removes the member that most reduces the (weighted) penalty and
    re-inserts the best non-member; if no strictly-improving insert exists for
    ``bump_stall`` steps, weights on violated keys are bumped to escape the plateau,
    with occasional random kicks / restarts. Returns (witness or None, best raw
    collisions seen).
    """
    pts = list(itertools.product((0, 1), repeat=n))
    if k > len(pts) or k < 0:
        return None, -1

    def fresh() -> tuple[CollisionCountIndex, set[Vector]]:
        idx = CollisionCountIndex()
        inset: set[Vector] = set()
        seed = list(warm) if warm else []
        for v in seed[:k]:
            vt = tuple(int(x) for x in v)
            if vt not in inset:
                idx.add(vt)
                inset.add(vt)
        while len(idx.members) < k:
            w = rng.choice(pts)
            if w not in inset:
                idx.add(w)
                inset.add(w)
        return idx, inset

    idx, inset = fresh()
    best_coll = idx.collisions
    start = time.time()
    stall = 0

    while idx.collisions > 0 and time.time() - start < time_budget:
        base = idx.penalty
        members = idx.members
        # focus: member whose removal most reduces the weighted penalty
        cand_m = members if len(members) <= 24 else rng.sample(members, 24)
        best_u = None
        best_drop = None
        for u in cand_m:
            idx.remove(u)
            drop = base - idx.penalty
            idx.add(u)
            if best_drop is None or drop > best_drop:
                best_drop = drop
                best_u = u
        u = best_u if best_u is not None else rng.choice(members)
        idx.remove(u)
        inset.discard(u)

        # best-improvement insertion over a shuffled candidate sample
        nonmem = [p for p in pts if p not in inset]
        rng.shuffle(nonmem)
        scan = nonmem[:sample_in] if sample_in and sample_in < len(nonmem) else nonmem
        best_w = None
        best_pen = None
        for w in scan:
            idx.add(w)
            pen = idx.penalty
            idx.remove(w)
            if best_pen is None or pen < best_pen or (pen == best_pen and rng.random() < 0.2):
                best_pen = pen
                best_w = w
        idx.add(best_w)
        inset.add(best_w)

        if idx.collisions < best_coll:
            best_coll = idx.collisions
            stall = 0
        else:
            stall += 1

        if best_pen is not None and best_pen >= base:
            # no improvement this step -> escalate
            if stall >= bump_stall:
                idx.bump_weights()
                stall = 0
                if rng.random() < 0.15:
                    # random kick to diversify
                    for _ in range(rng.randint(2, 4)):
                        um = rng.choice(idx.members)
                        idx.remove(um)
                        inset.discard(um)
                        nm = [p for p in pts if p not in inset]
                        add_v = rng.choice(nm)
                        idx.add(add_v)
                        inset.add(add_v)

    if idx.collisions == 0:
        return list(idx.members), 0
    return None, best_coll


# ---------------------------------------------------------------------------
# Public finder API.
# ---------------------------------------------------------------------------
def find_b3_set(
    n: int,
    k: int,
    *,
    time_budget: float = 30.0,
    seed: int = 0,
    warm: Sequence[Sequence[int]] | None = None,
    attempts: int = 6,
) -> dict[str, Any] | None:
    """
    Search for a B_3 set of size exactly k in {0,1}^n.

    Returns a witness dict {n, size, set, method, verified, elapsed} or None if no
    B_3 set of size k was found in the budget. Every returned witness is re-checked
    with the independent ``is_B3``. ``warm`` (a smaller/equal B_3 set) is used as an
    incumbent to seed the fixed-size repair.
    """
    start = time.time()
    warm_t = [tuple(int(x) for x in v) for v in warm] if warm else None
    per = max(1.0, time_budget / max(1, attempts))
    for a in range(attempts):
        rng = random.Random(seed * 1_000_003 + a)
        remaining = time_budget - (time.time() - start)
        if remaining <= 0.2:
            break
        witness, _ = _dls_repair(n, k, min(per, remaining), rng, warm=warm_t)
        if witness is not None:
            assert is_B3(witness), "finder produced a non-B_3 set (bug)"
            return {
                "n": n,
                "size": len(witness),
                "set": [list(v) for v in witness],
                "method": "dynamic_local_search",
                "verified": True,
                "elapsed": time.time() - start,
            }
    return None


def find_max_b3(
    n: int,
    *,
    time_budget: float = 60.0,
    seed: int = 0,
    seed_witnesses: Iterable[Sequence[Sequence[int]]] | None = None,
    exact_max_n: int = 4,
    target: int | None = None,
) -> dict[str, Any]:
    """
    Find as large a B_3 set in {0,1}^n as the budget allows.

    Strategy:
      * n <= ``exact_max_n``: exact branch-and-bound (returns a PROVEN optimum).
      * otherwise: greedy incumbent, then climb size-by-size with the fixed-size
        DLS repair, warm-starting each k from the k-1 witness. Any provided
        ``seed_witnesses`` (re-verified with is_B3) are used as extra incumbents /
        warm starts so hard values reproduce fast and deterministically.

    The returned dict always carries a ``set`` that passes the independent
    verifier; ``proven_optimal`` is True only for the B&B branch. No size is ever
    reported without such a witness.
    """
    start = time.time()

    if n <= exact_max_n:
        witness, complete = max_b3_branch_and_bound(n, time_limit=min(time_budget, 8.0))
        assert is_B3(witness)
        return {
            "n": n,
            "size": len(witness),
            "set": [list(v) for v in witness],
            "method": "branch_and_bound",
            "proven_optimal": complete,
            "verified": True,
            "canonical": [list(v) for v in canonical_form(witness, n)],
            "elapsed": time.time() - start,
        }

    rng = random.Random(seed)
    # Incumbent from greedy + any seed witnesses.
    incumbent = _greedy_max_set(n, rng)
    seeds: list[list[Vector]] = []
    provided = list(seed_witnesses) if seed_witnesses is not None else None
    if provided is None:
        # Default: reuse the finder's own stored witnesses as warm starts.
        try:
            from propab.domain_modules.math_combinatorics.discovery.known_witnesses import (
                verified_witnesses,
            )

            provided = verified_witnesses(n)
        except Exception:
            provided = []
    for w in provided:
        wt = [tuple(int(x) for x in v) for v in w]
        if wt and len(wt[0]) == n and is_B3(wt):
            seeds.append(wt)
    for s in seeds:
        if len(s) > len(incumbent):
            incumbent = list(s)

    best = list(incumbent)
    warm = list(incumbent)

    # Climb: try to reach best+1, best+2, ... within the remaining budget.
    while time.time() - start < time_budget:
        if target is not None and len(best) >= target:
            break
        target_k = len(best) + 1
        remaining = time_budget - (time.time() - start)
        if remaining <= 0.5:
            break
        # give bigger targets a larger slice
        per_budget = min(remaining, max(20.0, time_budget * 0.5))
        result = find_b3_set(
            n, target_k, time_budget=per_budget, seed=seed + target_k, warm=warm, attempts=4
        )
        if result is None:
            break
        best = [tuple(v) for v in result["set"]]
        warm = list(best)

    assert is_B3(best)
    return {
        "n": n,
        "size": len(best),
        "set": [list(v) for v in best],
        "method": "greedy+dls_climb",
        "proven_optimal": False,
        "verified": True,
        "canonical": [list(v) for v in canonical_form(best, n)] if n <= 6 else None,
        "elapsed": time.time() - start,
    }
