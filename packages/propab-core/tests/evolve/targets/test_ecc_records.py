"""The sourced-record registry itself (evolve/targets/ecc.py BEST_KNOWN).

This is the file that keeps the record table HONEST. The engine's whole discovery claim is
"we beat the best known d for this [n,k]", so a wrong baseline is not a cosmetic bug:

  * a baseline that is too LOW  -> we bank a code that is already known as a novel discovery.
    This is what makes it worth testing. The repo has shipped exactly this: constructors.py
    BEST_KNOWN_TABLE lists [31,16] -> 7 ("BCH [31,16,7]; optimal"), but the optimum is 8.
  * a baseline that is too HIGH -> the cell is unbeatable and the whole campaign is wasted.
  * a CLOSED cell (lower == upper) -> search there can only ever produce a rediscovery.

So every bound in the registry is re-checked here against elementary bounds that consult NO table:
Singleton, Griesmer, sphere-packing (Hamming), and the GF(2)^r distinct-column argument. These are
theorems, not citations - they hold whatever codetables.de says, and they are what would have caught
the [16,12] -> d=4 entry that the repo shipped (4 is impossible: only 15 nonzero vectors exist in
GF(2)^4, so 16 distinct nonzero parity-check columns cannot).

No network: the registry is a static, citable table and these tests are pure arithmetic over it.
"""
from __future__ import annotations

from math import comb

import pytest

from propab.domain_modules.coding_theory.constructors import MAX_EXHAUSTIVE_K
from propab.evolve.targets import ecc
from propab.evolve.targets.ecc import ECCProblem, UnsourcedCellError

ALL_CELLS = sorted(ecc.BEST_KNOWN.values(), key=lambda r: (r.k, r.n))
OPEN_CELLS = [r for r in ALL_CELLS if r.is_open]
IDS_ALL = [f"[{r.n},{r.k}]" for r in ALL_CELLS]
IDS_OPEN = [f"[{r.n},{r.k}]" for r in OPEN_CELLS]


# --------------------------------------------------------------------------- #
# Elementary bounds. Independent reimplementations - they must not import anything
# from the module under test, or they would just echo its assumptions back.
# --------------------------------------------------------------------------- #
def griesmer_min_length(k: int, d: int) -> int:
    """Least n for which a binary [n,k,d] code can exist: sum_{i<k} ceil(d / 2^i)."""
    return sum(-(-d // (2 ** i)) for i in range(k))


def fits_sphere_packing(n: int, k: int, d: int) -> bool:
    """Hamming bound: 2^k * |ball of radius t| <= 2^n, t = floor((d-1)/2)."""
    t = (d - 1) // 2
    return (2 ** k) * sum(comb(n, i) for i in range(t + 1)) <= 2 ** n


def test_griesmer_helper_is_itself_right():
    """Guard the guard: a broken bound-checker would wave every bad cell through."""
    assert griesmer_min_length(1, 9) == 9          # [9,1,9] repetition, tight
    assert griesmer_min_length(4, 3) == 7          # [7,4,3] Hamming, tight

    # Griesmer is only NECESSARY, never sufficient: it permits n=23 at k=12,d=8, yet no [23,12,8]
    # code exists (the Golay optimum sits at n=24). It may only ever be read as "no code exists
    # BELOW here" - never as "a code exists here".
    assert griesmer_min_length(12, 8) == 23

    # The two bounds are complementary, which is why both are run. Griesmer permits [14,11,3]...
    assert griesmer_min_length(11, 3) == 14
    # ...but sphere-packing kills it (2^11 * (1+14) = 30720 > 2^14), and the Hamming code is [15,11,3].
    assert not fits_sphere_packing(14, 11, 3)
    assert fits_sphere_packing(15, 11, 3)

    # Sphere-packing must also reject the repo's impossible [16,12,4]  (2^12 * 17 = 69632 > 2^16)...
    assert not fits_sphere_packing(16, 12, 4)
    assert fits_sphere_packing(16, 12, 2)          # ...but permit the true value d=2


# --------------------------------------------------------------------------- #
# The three properties the task turns on: open, cheap enough to verify, sane.
# --------------------------------------------------------------------------- #
def test_there_are_open_cells_to_search():
    """The failure that motivated this table: a registry of 130 CLOSED cells is not a target list."""
    assert len(OPEN_CELLS) >= 100, "the engine is target-poor again"


@pytest.mark.parametrize("rec", OPEN_CELLS, ids=IDS_OPEN)
def test_open_cell_has_a_real_gap(rec):
    """lower == upper means CLOSED: searching it can only produce a rediscovery."""
    assert rec.lower < rec.upper
    assert rec.gap >= 1
    assert rec.is_open


@pytest.mark.parametrize("rec", OPEN_CELLS, ids=IDS_OPEN)
def test_open_cell_is_within_the_exhaustive_verifier_limit(rec):
    """k > 16 cannot be certified: no full 2^k enumeration means no honest witness."""
    assert rec.k <= MAX_EXHAUSTIVE_K


@pytest.mark.parametrize("rec", ALL_CELLS, ids=IDS_ALL)
def test_every_record_obeys_the_elementary_bounds(rec):
    """The battery, run against EVERY cell - closed anchors included."""
    n, k, lo, up = rec.n, rec.k, rec.lower, rec.upper

    assert 0 < k <= n
    assert 1 <= lo <= up

    # Singleton: d <= n - k + 1, for both bounds.
    assert lo <= n - k + 1, f"[{n},{k}] lower={lo} violates Singleton"
    assert up <= n - k + 1, f"[{n},{k}] upper={up} violates Singleton"

    # An [n,k,lower] code is CLAIMED TO EXIST, so every nonexistence bound must permit it.
    assert griesmer_min_length(k, lo) <= n, (
        f"[{n},{k},{lo}] violates Griesmer: needs n >= {griesmer_min_length(k, lo)}"
    )
    assert fits_sphere_packing(n, k, lo), f"[{n},{k},{lo}] violates the sphere-packing bound"

    # The GF(2)^r distinct-column argument. With r = n - k parity checks, d >= 3 forces the n
    # columns of H to be distinct and nonzero, and GF(2)^r holds only 2^r - 1 nonzero vectors.
    # This alone refutes the repo's [16,12] -> d=4.
    r = n - k
    if lo >= 3:
        assert n <= 2 ** r - 1, (
            f"[{n},{k},{lo}] impossible: d>=3 needs n <= 2^{r}-1 = {2 ** r - 1}, but n={n}"
        )


def test_the_16_12_class_of_bug_would_be_caught():
    """A regression test for the CHECK, not the table: the battery must reject [16,12,4]."""
    n, k, d = 16, 12, 4
    r = n - k
    assert d >= 3 and n > 2 ** r - 1          # the distinct-column argument fires
    assert not fits_sphere_packing(n, k, d)   # ...and so does sphere-packing
    # and the registry records the TRUE value
    assert ecc.lookup(16, 12).lower == 2
    assert ecc.lookup(16, 12).upper == 2


# --------------------------------------------------------------------------- #
# Provenance: a bound we cannot cite is not a bound we may search against.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rec", OPEN_CELLS, ids=IDS_OPEN)
def test_both_bounds_of_an_open_cell_are_sourced(rec):
    for label, source in (("lower", rec.lower_source), ("upper", rec.upper_source)):
        assert source and source.strip(), f"[{rec.n},{rec.k}] {label} bound has no source"
        # the derivation must actually be about THIS cell
        tag = "Lb" if label == "lower" else "Ub"
        assert f"{tag}({rec.n},{rec.k})" in source, (
            f"[{rec.n},{rec.k}] {label} source does not derive this cell: {source[:80]}"
        )
    assert rec.accessed


def test_every_citation_key_resolves_and_none_is_dead():
    """Every key named in _OPEN_CELLS must exist in _REFERENCES, and vice versa."""
    used = {
        key
        for _n, _k, _lo, _up, _ld, lk, _ud, uk in ecc._OPEN_CELLS
        for key in (*lk, *uk)
    }
    known = set(ecc._REFERENCES)
    assert used <= known, f"citation keys with no reference text: {sorted(used - known)}"
    assert known <= used, f"unused reference entries: {sorted(known - used)}"
    for key in used:
        assert len(ecc._REFERENCES[key]) > 10, f"reference {key} is not a real citation"


def test_unexpanded_upstream_references_are_declared_not_hidden():
    """codetables.de cites two keys it never expands (it renders "Sh1:" / "cy:" with an empty body).

    We did not invent citations for them. This test pins WHICH cells are affected so the gap stays
    visible: if a future re-scrape expands them, or a new unexpanded key appears, this fails loudly
    rather than letting an un-cited bound slip in unnoticed.
    """
    assert ecc._UNEXPANDED_UPSTREAM_KEYS == frozenset({"Sh1", "cy"})
    for key in ecc._UNEXPANDED_UPSTREAM_KEYS:
        assert ecc._REFERENCES[key].startswith("UNEXPANDED:")

    affected = {(r.n, r.k) for r in ecc.cells_with_unexpanded_reference()}
    assert affected == {
        (33, 14), (33, 15), (34, 15), (34, 16), (35, 16),                  # Sh1
        (41, 12), (42, 14), (43, 15), (50, 14), (50, 15), (50, 16),        # cy
        (51, 15), (51, 16), (52, 16), (64, 11), (64, 12),                  # cy
    }

    # The bound VALUES are still fully sourced - that is why these cells are kept. Every one of them
    # still carries a derivation chain for both bounds and still passes the elementary battery.
    for rec in ecc.cells_with_unexpanded_reference():
        assert f"Lb({rec.n},{rec.k})" in rec.lower_source
        assert f"Ub({rec.n},{rec.k})" in rec.upper_source
        assert rec.lower < rec.upper
        assert griesmer_min_length(rec.k, rec.lower) <= rec.n
        assert fits_sphere_packing(rec.n, rec.k, rec.lower)

    # ...and no OTHER cell quietly depends on an unexpanded key.
    clean = [r for r in OPEN_CELLS if (r.n, r.k) not in affected]
    assert len(clean) == len(OPEN_CELLS) - 16
    for rec in clean:
        assert "UNEXPANDED:" not in rec.lower_source
        assert "UNEXPANDED:" not in rec.upper_source


# --------------------------------------------------------------------------- #
# The [31,16] disagreement with constructors.BEST_KNOWN_TABLE.
# --------------------------------------------------------------------------- #
def test_31_16_records_the_true_optimum_not_the_bch_distance():
    """constructors.BEST_KNOWN_TABLE says 7 ("BCH [31,16,7]; optimal"). It is not optimal: the
    [32,17,8] Cheng-Sloane code shortens to [31,16,8]. A baseline of 7 would let the engine bank
    the known optimum as a discovery."""
    rec = ecc.lookup(31, 16)
    assert (rec.lower, rec.upper) == (8, 8)
    assert not rec.is_open                     # closed at 8 - nothing to find here

    # ECCProblem takes the MAX over every source, so the corrected value governs the baseline
    # even while constructors.py still carries the wrong 7.
    assert ECCProblem(31, 16).best_known() == 8.0


# --------------------------------------------------------------------------- #
# Every open cell must actually be runnable, and unsourced cells must still refuse.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rec", OPEN_CELLS, ids=IDS_OPEN)
def test_every_open_cell_constructs_a_runnable_problem(rec):
    problem = ECCProblem(rec.n, rec.k)
    assert problem.record.is_open
    # The number to beat is never below the sourced record...
    assert problem.best_known() >= float(rec.lower)
    # ...and a beat is still possible in principle (the baseline has not swallowed the gap).
    assert problem.best_known() < float(rec.upper), (
        f"[{rec.n},{rec.k}] baseline {problem.best_known()} leaves no room below upper={rec.upper}"
    )


def test_open_cells_helper_returns_only_open_verifiable_cells():
    cells = ecc.open_cells()
    assert {(r.n, r.k) for r in cells} == {(r.n, r.k) for r in OPEN_CELLS}
    for r in cells:
        assert r.is_open and r.k <= MAX_EXHAUSTIVE_K
    # sorted cheapest-to-verify first (cost is 2^k, independent of n)
    assert [r.k for r in cells] == sorted(r.k for r in cells)


def test_a_cell_we_cannot_cite_is_still_refused():
    """UnsourcedCellError must survive: an unsourced cell may never default to a guessed baseline."""
    assert ecc.lookup(41, 10) is None            # closed on codetables.de, so never sourced here
    with pytest.raises(UnsourcedCellError):
        ECCProblem(41, 10)


def test_registry_has_no_duplicate_or_mismatched_keys():
    for (n, k), rec in ecc.BEST_KNOWN.items():
        assert (rec.n, rec.k) == (n, k)
    keys = [(n, k) for n, k, *_ in ecc._OPEN_CELLS]
    assert len(keys) == len(set(keys)), "duplicate cell in _OPEN_CELLS"
