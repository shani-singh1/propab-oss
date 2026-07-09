"""S1 tool: certify_b3_record exposes the trusted independent certifier standalone.

Honesty checks:
  * a genuine size-16 B_3 set for n=7 certifies as a valid set (is_b3, certified)
    but is NOT a record (16 does not beat best-known 16);
  * a non-B_3 set is REJECTED (is_b3 False, certified False) even if its size would
    numerically beat a table value — validity is required;
  * the tool routes through the REAL certify_b3_record (its audit dict is returned).
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.discovery.known_witnesses import WITNESSES
from propab.tools.mathematics.certify_b3_record import certify_b3_record


def test_certify_valid_size16_n7_not_record() -> None:
    witness = WITNESSES[7][0]  # a real, finder-produced size-16 B_3 set
    r = certify_b3_record(set=witness, n=7)
    assert r.success, r.error
    o = r.output
    assert o["is_b3"] is True
    assert o["size"] == 16
    assert o["certified"] is True
    assert o["beats_known"] is False  # 16 is NOT > best-known 16
    assert o["is_record"] is False
    # Real certifier audit trail is present.
    assert o["certification"]["checks"]["is_b3"] is True


def test_certify_rejects_non_b3_set() -> None:
    # 000+000+110 == 000+100+010 -> a threefold-sum collision, so NOT B_3.
    non_b3 = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
    r = certify_b3_record(set=non_b3, n=3)
    assert r.success, r.error
    o = r.output
    assert o["is_b3"] is False
    assert o["certified"] is False
    assert o["is_record"] is False


def test_certify_singleton_is_b3_but_no_record() -> None:
    r = certify_b3_record(set=[[0, 0, 0]], n=3)
    assert r.success, r.error
    o = r.output
    assert o["is_b3"] is True
    assert o["certified"] is True
    assert o["beats_known"] is False


def test_certify_requires_set() -> None:
    r = certify_b3_record(set=None, n=3)
    assert not r.success
    assert r.error.type == "validation_error"
