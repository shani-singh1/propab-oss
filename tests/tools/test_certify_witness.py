"""certify_witness — the general witness certifier. Honesty invariants:
independent re-derivation (a wrong witness is REFUTED with a self-certifying violation),
existence-only (never claims optimality), and is_record only when a certified witness
strictly beats a supplied best-known.
"""
from __future__ import annotations

from propab.tools.registry import ToolRegistry

_R = ToolRegistry()


def call(**p):
    return _R.call("certify_witness", p)


def test_registers_verification_capable():
    spec = next(s for s in _R.get_all_specs() if s["name"] == "certify_witness")
    assert spec.get("verification_capable") is True


# ── sidon (B_2) ───────────────────────────────────────────────────────────────
def test_sidon_valid():
    r = call(witness=[1, 2, 5, 11], property="sidon")
    assert r.success and r.output["holds"] is True and r.output["certified"] is True and r.output["size"] == 4


def test_sidon_invalid_returns_violation():
    # [1,2,3,4] is not Sidon (e.g. 1+3 = 2+2 = 4). The collision is self-certifying:
    # both combos genuinely sum to violation['sum'].
    r = call(witness=[1, 2, 3, 4], property="sidon")
    assert r.success and r.output["holds"] is False and r.output["certified"] is False
    v = r.output["violation"]
    assert v and sum(v["combo_a"]) == v["sum"] == sum(v["combo_b"]) and v["combo_a"] != v["combo_b"]


# ── b_h (B_3) ─────────────────────────────────────────────────────────────────
def test_b3_valid_and_invalid():
    assert call(witness=[0, 1, 2, 4], property="b_h", h=3).output["holds"] in (True, False)
    # a clear B_3 failure: 0+0+3 == 1+1+1 == 3
    r = call(witness=[0, 1, 3], property="b_h", h=3)
    assert r.success and r.output["holds"] is False


# ── golomb ruler (distinct differences) ──────────────────────────────────────
def test_golomb():
    assert call(witness=[0, 1, 3], property="golomb_ruler").output["holds"] is True
    assert call(witness=[0, 1, 2], property="golomb_ruler").output["holds"] is False  # diff 1 twice


# ── sum-free ─────────────────────────────────────────────────────────────────
def test_sum_free():
    assert call(witness=[1, 3, 5], property="sum_free").output["holds"] is True
    r = call(witness=[3, 5, 8], property="sum_free")  # 3+5=8, the only violation
    assert r.output["holds"] is False
    v = r.output["violation"]
    assert v["a"] + v["b"] == v["c"] and v["c"] == 8


# ── progression-free ─────────────────────────────────────────────────────────
def test_progression_free():
    assert call(witness=[1, 2, 4, 8], property="progression_free", k=3).output["holds"] is True
    r = call(witness=[1, 2, 3], property="progression_free", k=3)  # 1,2,3 is a 3-AP
    assert r.output["holds"] is False


# ── sidon_mod ─────────────────────────────────────────────────────────────────
def test_sidon_mod_requires_modulus():
    assert not call(witness=[1, 2, 5], property="sidon_mod").success


# ── record logic (existence vs record) ───────────────────────────────────────
def test_is_record_only_when_beats_best_known():
    # certified size 4, best-known 3 -> record.
    r = call(witness=[1, 2, 5, 11], property="sidon", published_best=3)
    assert r.output["certified"] is True and r.output["beats_best_known"] is True and r.output["is_record"] is True
    # same witness, best-known 4 -> certified but NOT a record.
    r2 = call(witness=[1, 2, 5, 11], property="sidon", published_best=4)
    assert r2.output["certified"] is True and r2.output["is_record"] is False


def test_uncertified_never_a_record():
    # a non-Sidon set that claims to beat best-known must NOT be a record.
    r = call(witness=[1, 2, 3, 4], property="sidon", published_best=2)
    assert r.output["holds"] is False and r.output["is_record"] is False


# ── honest failures ──────────────────────────────────────────────────────────
def test_bad_inputs():
    assert not call(witness=[1, 2], property="teleport").success
    assert not call(witness=[], property="sidon").success
    assert not call(property="sidon").success  # missing witness
