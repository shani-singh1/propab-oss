"""
Sourced registry of best-known values for the sequences in the discovery-target
research doc (``artifacts/discovery_targets_math.md``).

Every value carries an OEIS id (or a stable slug), a URL, the term's status, and a
short source note. This is the ground truth a claimed record is checked against:
a witness "sets a record" only if it strictly beats ``best_known(oeis_id, n)`` for
a term whose status is a bound that search can improve (``provisional_lower_bound``
or ``open``), never a ``proven_optimal`` one.

Status vocabulary
-----------------
- ``proven_optimal``          -- the extremal value is proven (exhaustive / IP).
- ``provisional_lower_bound`` -- best known is a search bound, improvable by a
                                 larger witness (the record-chase regime).
- ``open``                    -- no value published yet (a witness gives a first
                                 bound; exact optimality still needs a proof).

Values transcribed 2026-07-08 from the research doc; see each entry's ``url``.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# RECORDS: oeis_id/slug -> sequence metadata + per-n terms.
# ---------------------------------------------------------------------------
RECORDS: dict[str, dict[str, Any]] = {
    "A396704": {
        "oeis_id": "A396704",
        "url": "https://oeis.org/A396704",
        "title": "Maximum size a(n) of a B_3 set in {0,1}^n (threefold sums distinct).",
        "objective": "maximize",
        "source_note": (
            "Created 2026-06-28, keywords hard,more,new. a(0..6) proven optimal "
            "by exhaustive IP over the hyperoctahedral symmetry; a(7)>=16, a(8)>=19 "
            "are iterated-local-search lower bounds, NOT proven optimal. Companion "
            "code: github.com/willblair0708/verified-combinatorics/tree/main/b3-binary. "
            "THIS is the discovery kernel's primary target (improve a(7) to 17)."
        ),
        "terms": {
            0: {"best_known": 1, "status": "proven_optimal"},
            1: {"best_known": 2, "status": "proven_optimal"},
            2: {"best_known": 3, "status": "proven_optimal"},
            3: {"best_known": 4, "status": "proven_optimal"},
            4: {"best_known": 6, "status": "proven_optimal"},
            5: {"best_known": 8, "status": "proven_optimal"},
            6: {"best_known": 11, "status": "proven_optimal"},
            7: {"best_known": 16, "status": "provisional_lower_bound"},
            8: {"best_known": 19, "status": "provisional_lower_bound"},
        },
    },
    "A385931": {
        "oeis_id": "A385931",
        "url": "https://oeis.org/A385931",
        "title": "Weak B_3 Golomb ruler: least length L(n) with all sums of 3 distinct elements distinct.",
        "objective": "minimize",
        "source_note": (
            "Exact for n=1..10; a(11) uncomputed (keywords hard,more). best_known "
            "is the minimal ruler length (smaller is better). Uncontested fallback "
            "target; a first exact a(11) needs the exhaustive 'no shorter ruler' proof."
        ),
        "terms": {
            1: {"best_known": 0, "status": "proven_optimal"},
            2: {"best_known": 1, "status": "proven_optimal"},
            3: {"best_known": 2, "status": "proven_optimal"},
            4: {"best_known": 3, "status": "proven_optimal"},
            5: {"best_known": 7, "status": "proven_optimal"},
            6: {"best_known": 13, "status": "proven_optimal"},
            7: {"best_known": 22, "status": "proven_optimal"},
            8: {"best_known": 39, "status": "proven_optimal"},
            9: {"best_known": 69, "status": "proven_optimal"},
            10: {"best_known": 113, "status": "proven_optimal"},
            11: {"best_known": None, "status": "open"},
        },
    },
    "A004135": {
        "oeis_id": "A004135",
        "url": "https://oeis.org/A004135",
        "title": "Modular Golomb ruler (distinct pairs): least k with an n-subset of Z_k, all pairwise sums distinct.",
        "objective": "minimize",
        "source_note": (
            "a(n) = least k such that Z_k has an n-subset whose sums of DISTINCT pairs "
            "(a<b) are distinct mod k. Known through n=17 (a(17)=255); a(18) open. "
            "Frontier frozen since Cariboni 2017-2018; not marked hard. Terms 1..17 "
            "transcribed from OEIS A004135 (offset 1). Modular-Sidon witness check is cheap."
        ),
        "terms": {
            1: {"best_known": 1, "status": "proven_optimal"},
            2: {"best_known": 2, "status": "proven_optimal"},
            3: {"best_known": 3, "status": "proven_optimal"},
            4: {"best_known": 6, "status": "proven_optimal"},
            5: {"best_known": 11, "status": "proven_optimal"},
            6: {"best_known": 19, "status": "proven_optimal"},
            7: {"best_known": 28, "status": "proven_optimal"},
            8: {"best_known": 40, "status": "proven_optimal"},
            9: {"best_known": 56, "status": "proven_optimal"},
            10: {"best_known": 72, "status": "proven_optimal"},
            11: {"best_known": 96, "status": "proven_optimal"},
            12: {"best_known": 114, "status": "proven_optimal"},
            13: {"best_known": 147, "status": "proven_optimal"},
            14: {"best_known": 178, "status": "proven_optimal"},
            15: {"best_known": 183, "status": "proven_optimal"},
            16: {"best_known": 252, "status": "proven_optimal"},
            17: {"best_known": 255, "status": "proven_optimal"},
            18: {"best_known": None, "status": "open"},
        },
    },
    "A004136": {
        "oeis_id": "A004136",
        "url": "https://oeis.org/A004136",
        "title": "Perfect/modular Golomb ruler (incl. repeats): least k with an n-subset of Z_k, all pairwise sums distinct mod k.",
        "objective": "minimize",
        "source_note": (
            "a(n) = least k such that Z_k has an n-subset whose sums of ALL pairs "
            "(a<=b, incl. the doubles 2a) are distinct mod k (a perfect/modular Golomb "
            "ruler). Known through n=18 (a(18)=307); a(19) open. Frontier frozen since "
            "Cariboni 2017-2018; not marked hard. Terms 1..18 from OEIS A004136 (offset 1)."
        ),
        "terms": {
            1: {"best_known": 1, "status": "proven_optimal"},
            2: {"best_known": 3, "status": "proven_optimal"},
            3: {"best_known": 7, "status": "proven_optimal"},
            4: {"best_known": 13, "status": "proven_optimal"},
            5: {"best_known": 21, "status": "proven_optimal"},
            6: {"best_known": 31, "status": "proven_optimal"},
            7: {"best_known": 48, "status": "proven_optimal"},
            8: {"best_known": 57, "status": "proven_optimal"},
            9: {"best_known": 73, "status": "proven_optimal"},
            10: {"best_known": 91, "status": "proven_optimal"},
            11: {"best_known": 120, "status": "proven_optimal"},
            12: {"best_known": 133, "status": "proven_optimal"},
            13: {"best_known": 168, "status": "proven_optimal"},
            14: {"best_known": 183, "status": "proven_optimal"},
            15: {"best_known": 255, "status": "proven_optimal"},
            16: {"best_known": 255, "status": "proven_optimal"},
            17: {"best_known": 273, "status": "proven_optimal"},
            18: {"best_known": 307, "status": "proven_optimal"},
            19: {"best_known": None, "status": "open"},
        },
    },
    "A309370": {
        "oeis_id": "A309370",
        "url": "https://oeis.org/A309370",
        "title": "Maximum size of a Sidon (B_2) set in {0,1}^n (pairwise sums distinct).",
        "objective": "maximize",
        "source_note": (
            "Exact only for n<=6 (a(6)=15); n>=7 lower bounds only, e.g. a(7)>=24 "
            "(Sievers 2025), a(16)>=505, a(24)>=7179 (Blair 2026). Every B_3 set is a "
            "B_2 set, so A396704(n) <= A309370(n): this bounds the A396704 target "
            "(a(7) in [16, 24]). Most actively contested sequence in the family."
        ),
        "terms": {
            6: {"best_known": 15, "status": "proven_optimal"},
            7: {"best_known": 24, "status": "provisional_lower_bound"},
            16: {"best_known": 505, "status": "provisional_lower_bound"},
            24: {"best_known": 7179, "status": "provisional_lower_bound"},
        },
    },
    "A090245": {
        "oeis_id": "A090245",
        "url": "https://arxiv.org/abs/2206.09804",
        "title": "Maximum size of a cap set in F_3^n (no three collinear).",
        "objective": "maximize",
        "source_note": (
            "Exact for n<=6 (2,4,9,20,45,112). n=7 OPEN: lower bound 236 (Tyrrell, "
            "arxiv 2209.10045), upper bound <=288 (no 289-cap; Cameron et al., "
            "arxiv 2206.09804). Propab has a native O(|S|^2) cap checker but its "
            "product-construction finder (~180) is far below 236."
        ),
        "terms": {
            2: {"best_known": 4, "status": "proven_optimal"},
            3: {"best_known": 9, "status": "proven_optimal"},
            4: {"best_known": 20, "status": "proven_optimal"},
            5: {"best_known": 45, "status": "proven_optimal"},
            6: {"best_known": 112, "status": "proven_optimal"},
            7: {"best_known": 236, "status": "provisional_lower_bound"},
        },
    },
}

# Primary target for the discovery kernel.
PRIMARY_TARGET = "A396704"


def get_record(oeis_id: str, n: int | None = None) -> dict[str, Any] | None:
    """
    Look up a sequence entry, or a single (sequence, n) term.

    With ``n`` omitted, returns the whole sequence metadata dict (incl. ``terms``).
    With ``n`` given, returns a flat term dict
    ``{oeis_id, url, n, best_known, status, source_note}`` or None if absent.
    """
    seq = RECORDS.get(oeis_id)
    if seq is None:
        return None
    if n is None:
        return seq
    term = seq["terms"].get(n)
    if term is None:
        return None
    return {
        "oeis_id": seq["oeis_id"],
        "url": seq["url"],
        "n": n,
        "best_known": term["best_known"],
        "status": term["status"],
        "source_note": seq["source_note"],
    }


def best_known(oeis_id: str, n: int) -> int | None:
    """Best-known value for term (oeis_id, n), or None if open/absent."""
    term = get_record(oeis_id, n)
    return term["best_known"] if term else None


def record_status(oeis_id: str, n: int) -> str | None:
    """Status string for term (oeis_id, n), or None if absent."""
    term = get_record(oeis_id, n)
    return term["status"] if term else None
