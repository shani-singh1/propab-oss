"""
Deterministic, paranoid B_3 witness verification (headline-defensibility layer).

A set S subset of {0,1}^n is a **B_3 set** iff all threefold multiset sums
a + b + c (a, b, c in S, repeats allowed), taken componentwise over the integers
so each sum lands in {0,1,2,3}^n, are DISTINCT as integer vectors. Equivalently,
a + b + c = d + e + f with a..f in S forces the multisets {a,b,c} = {d,e,f}.

``is_B3`` is intentionally independent of the finder: it re-derives every
threefold sum from scratch by direct enumeration of unordered triples with
repetition. It never trusts an incremental index, a claimed size, or any
finder-side bookkeeping. Cost for |S|=m is C(m+2, 3) vector sums -- sub-millisecond
for the sizes of interest (m ~ 17).

``certify_b3_record`` is the record-defensibility gate. It is deliberately
paranoid: it re-checks that every vector lives in {0,1}^n, that the claimed size
strictly beats the published best, and that the set is genuinely B_3 -- each an
independent boolean, so a failure localizes cleanly. No false positives possible.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

Vector = tuple[int, ...]


def _as_tuples(S: Iterable[Sequence[int]]) -> list[Vector]:
    return [tuple(int(x) for x in v) for v in S]


def threefold_sums(S: Sequence[Sequence[int]]) -> list[Vector]:
    """
    Return every threefold multiset sum a+b+c (i<=j<=k) as an integer vector.

    Length is C(m+2, 3) for m = |S|. Provided for tests / diagnostics; ``is_B3``
    does not build the full list (it short-circuits on the first collision).
    """
    pts = _as_tuples(S)
    if not pts:
        return []
    n = len(pts[0])
    out: list[Vector] = []
    m = len(pts)
    for i in range(m):
        a = pts[i]
        for j in range(i, m):
            b = pts[j]
            for k in range(j, m):
                c = pts[k]
                out.append(tuple(a[t] + b[t] + c[t] for t in range(n)))
    return out


def is_B3(S: Sequence[Sequence[int]]) -> bool:
    """
    Exact, deterministic B_3 test on the ACTUAL set S (list of int tuples).

    Returns False if S has duplicate vectors, ragged dimensions, or any two
    distinct unordered triples share a threefold sum. Cheap and paranoid: no
    incremental state, no false positives. The empty set and singletons are B_3.
    """
    pts = _as_tuples(S)
    if not pts:
        return True
    n = len(pts[0])
    if any(len(v) != n for v in pts):
        return False
    if len({v for v in pts}) != len(pts):
        return False
    seen: set[Vector] = set()
    m = len(pts)
    for i in range(m):
        a = pts[i]
        for j in range(i, m):
            b = pts[j]
            ab = tuple(a[t] + b[t] for t in range(n))
            for k in range(j, m):
                c = pts[k]
                key = tuple(ab[t] + c[t] for t in range(n))
                if key in seen:
                    return False
                seen.add(key)
    return True


def certify_b3_record(
    witness: Any,
    published_best: int,
    *,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """
    Independent, paranoid certification that ``witness`` beats ``published_best``.

    ``witness`` may be a dict ({"n": .., "set": [...]}) or a bare list of vectors.
    The three headline checks are computed independently:

      * ``in_binary_cube`` -- every vector has the same length n and entries in {0,1}
      * ``strictly_beats``  -- size > published_best  (a strict record improvement)
      * ``is_b3``           -- ``is_B3`` re-run from scratch on the actual set

    ``certified`` is the AND of all three (plus distinctness). This function makes
    no claim about optimality and never mutates the witness. It is the only thing
    a downstream "we set a record" assertion should be allowed to trust.
    """
    if isinstance(witness, dict):
        raw = witness.get("set")
        if raw is None:
            raw = witness.get("witness", [])
        declared_n = witness.get("n")
    else:
        raw = witness
        declared_n = None

    S = _as_tuples(raw or [])
    size = len(S)

    n = expected_n if expected_n is not None else declared_n
    if n is None:
        n = len(S[0]) if S else 0

    distinct = len({v for v in S}) == size
    in_cube = bool(S) and all(len(v) == n and all(x in (0, 1) for x in v) for v in S)
    strictly_beats = size > int(published_best)
    # Only run the (already cheap) B_3 check when the basic shape is sane.
    b3_ok = is_B3(S) if (in_cube and distinct) else False

    certified = bool(in_cube and distinct and strictly_beats and b3_ok)

    return {
        "certified": certified,
        "n": n,
        "size": size,
        "published_best": int(published_best),
        "checks": {
            "in_binary_cube": in_cube,
            "distinct_vectors": distinct,
            "strictly_beats_published": strictly_beats,
            "is_b3": b3_ok,
        },
        "margin": size - int(published_best),
        "note": (
            "CERTIFIED: witness is a valid B_3 set strictly larger than the "
            "published best. This certifies a lower-bound improvement only, NOT "
            "optimality."
            if certified
            else "NOT certified -- see checks for the failing condition."
        ),
    }
