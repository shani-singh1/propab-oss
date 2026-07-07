"""
Explicit B_3 witnesses FOUND BY THIS PACKAGE'S FINDER (provenance record).

These are not looked-up literature objects: each was produced by
``finder.find_max_b3`` / ``_dls_repair`` during development and is re-verified with
the independent ``is_B3`` on import (see ``verified_witnesses``). They serve two
honest purposes:

  * provenance -- concrete evidence the finder actually reaches these sizes;
  * warm starts -- so ``find_max_b3`` reproduces the hard values (a(7)>=16) quickly
    and deterministically in CI instead of re-running a multi-minute search.

IMPORTANT: a witness here for n=7 of size 16 reproduces the PUBLISHED lower bound
a(7)>=16. It is NOT a record. Any size-17 witness would be, and must be routed
through ``certify_b3_record`` and reported for independent re-verification -- it is
deliberately NOT asserted as a record anywhere in this package.
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.discovery.verifier import is_B3

# Map n -> list of witnesses (each a list of 0/1 vectors). Filled with sets the
# finder produced; kept small and canonical.
WITNESSES: dict[int, list[list[list[int]]]] = {
    # n=7, size 16 -> reproduces the published lower bound a(7) >= 16.
    # Found by finder._dls_repair during development, then canonicalized under the
    # hyperoctahedral group. Re-verified by is_B3 on import. NOT a record (== 16).
    7: [
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0],
        ],
    ],
}


def verified_witnesses(n: int) -> list[list[tuple[int, ...]]]:
    """Return the stored witnesses for n, each re-checked with is_B3 (paranoid)."""
    out: list[list[tuple[int, ...]]] = []
    for w in WITNESSES.get(n, []):
        if not w:
            continue
        tw = [tuple(int(x) for x in v) for v in w]
        if is_B3(tw):
            out.append(tw)
    return out


def best_seed(n: int) -> list[tuple[int, ...]] | None:
    """Largest verified stored witness for n (a warm-start incumbent), or None."""
    ws = verified_witnesses(n)
    if not ws:
        return None
    return max(ws, key=len)
