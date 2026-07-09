"""S1 tool: extremal_set_search wraps the trusted B_3 finder + independent certifier.

Honesty checks:
  * n=7 returns a CERTIFIED size-16 B_3 set (the published lower bound a(7)=16),
    and certification comes from the REAL certify_b3_record (cross-checked with the
    independent is_B3), NOT self-report.
  * size==best-known is certified as a valid set but is NOT flagged a record.
  * n=4 proves the optimum via exact branch-and-bound.
  * an unsupported object is rejected.
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.discovery.verifier import is_B3
from propab.tools.mathematics.extremal_set_search import extremal_set_search


def test_search_n7_certified_size16_not_record() -> None:
    r = extremal_set_search(object="b3_binary_cube", n=7, time_budget_sec=30)
    assert r.success, r.error
    o = r.output
    assert o["size"] == 16, o["size"]
    assert o["certified"] is True
    # 16 == best-known 16 -> certified valid set, but NOT a record.
    assert o["best_known"] == 16
    assert o["beats_best_known"] is False
    assert o["is_record"] is False
    # The returned witness really is a B_3 set (independent re-derivation), proving
    # `certified` was not self-reported.
    assert is_B3([tuple(v) for v in o["set"]]) is True
    # Certification came from the real certifier (its audit dict is attached).
    assert o["certification"]["checks"]["is_b3"] is True
    assert o["certification"]["checks"]["strictly_beats_published"] is False


def test_search_n4_proven_optimal() -> None:
    r = extremal_set_search(object="b3_binary_cube", n=4, time_budget_sec=10)
    assert r.success, r.error
    o = r.output
    assert o["size"] == 6  # a(4)=6, proven optimal
    assert o["proven_optimal"] is True
    assert o["certified"] is True
    assert is_B3([tuple(v) for v in o["set"]]) is True


def test_search_rejects_unknown_object() -> None:
    r = extremal_set_search(object="cap_set_in_f5", n=5, time_budget_sec=5)
    assert not r.success
    assert r.error.type == "validation_error"


def test_search_requires_n() -> None:
    r = extremal_set_search(object="b3_binary_cube", n=None, time_budget_sec=5)
    assert not r.success
    assert r.error.type == "validation_error"
