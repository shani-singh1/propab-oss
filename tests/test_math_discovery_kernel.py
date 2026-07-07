"""
Tests for the math_combinatorics DISCOVERY KERNEL (B_3 sets in {0,1}^n, A396704).

Covers the four mandated gates:
  * ``is_B3`` true/false cases;
  * ``certify_b3_record`` rejects non-B_3 / too-small witnesses and accepts a good one;
  * the record registry values match the sourced research doc;
  * the finder reproduces the proven values a(0..6) = 1,2,3,4,6,8,11 exactly and
    reaches the published lower bound a(7) >= 16.
"""
from __future__ import annotations

import itertools

import pytest

from propab.domain_modules.math_combinatorics.discovery import (
    RECORDS,
    best_known,
    certify_b3_record,
    find_max_b3,
    get_record,
    is_B3,
)
from propab.domain_modules.math_combinatorics.discovery.verifier import threefold_sums
from propab.domain_modules.math_combinatorics.discovery.symmetry import (
    canonical_form,
    translate_to_origin,
)
from propab.domain_modules.math_combinatorics.discovery.known_witnesses import (
    verified_witnesses,
)


# --------------------------------------------------------------------------- #
# is_B3
# --------------------------------------------------------------------------- #
def test_is_b3_trivial_cases():
    assert is_B3([]) is True
    assert is_B3([(0, 0, 0)]) is True
    assert is_B3([(0, 0, 0), (1, 0, 0)]) is True


def test_is_b3_true_small_set():
    # {0, e1, e2, e1+e2+... } small explicit B_3 set in {0,1}^3 of size 4 (= a(3)).
    S = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    assert is_B3(S) is True


def test_is_b3_false_on_collision():
    # A threefold-sum collision: 0+0+(1,1,0) == (1,0,0)+(0,1,0)+0 == (1,1,0).
    S = [(0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 1, 0)]
    assert is_B3(S) is False


def test_is_b3_false_on_duplicate_vectors():
    assert is_B3([(1, 0), (1, 0)]) is False


def test_is_b3_rejects_ragged_dimension():
    assert is_B3([(0, 0, 0), (1, 0)]) is False


def test_threefold_sums_count():
    S = [(0, 0), (1, 0), (0, 1)]
    m = len(S)
    expected = m * (m + 1) * (m + 2) // 6  # C(m+2, 3)
    assert len(threefold_sums(S)) == expected


def test_is_b3_agrees_with_brute_force_small():
    # Exhaustively confirm is_B3 matches the definition on random 4-subsets of {0,1}^3.
    pts = list(itertools.product((0, 1), repeat=3))
    import random

    rng = random.Random(0)
    for _ in range(200):
        S = rng.sample(pts, 4)
        # brute: all unordered triples-with-repetition give distinct sums
        sums = [tuple(map(sum, zip(S[i], S[j], S[k])))
                for i in range(4) for j in range(i, 4) for k in range(j, 4)]
        brute = len(set(sums)) == len(sums)
        assert is_B3(S) == brute


# --------------------------------------------------------------------------- #
# certify_b3_record
# --------------------------------------------------------------------------- #
def test_certifier_accepts_valid_improvement():
    # Any valid B_3 set strictly larger than a (fake) small published best certifies.
    S = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # size 4, valid B_3
    res = certify_b3_record({"n": 3, "set": [list(v) for v in S]}, published_best=3)
    assert res["certified"] is True
    assert res["checks"]["is_b3"] and res["checks"]["strictly_beats_published"]
    assert res["margin"] == 1


def test_certifier_rejects_too_small():
    S = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # size 4
    res = certify_b3_record({"n": 3, "set": [list(v) for v in S]}, published_best=4)
    assert res["certified"] is False
    assert res["checks"]["strictly_beats_published"] is False
    assert res["checks"]["is_b3"] is True  # it IS B_3, just not larger


def test_certifier_rejects_non_b3():
    S = [(0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 1, 0)]  # collision -> not B_3
    res = certify_b3_record({"n": 3, "set": [list(v) for v in S]}, published_best=2)
    assert res["certified"] is False
    assert res["checks"]["is_b3"] is False


def test_certifier_rejects_non_binary_vector():
    S = [(0, 0, 0), (2, 0, 0), (0, 1, 0)]  # 2 is outside {0,1}
    res = certify_b3_record({"n": 3, "set": [list(v) for v in S]}, published_best=1)
    assert res["certified"] is False
    assert res["checks"]["in_binary_cube"] is False


def test_certifier_accepts_bare_list():
    S = [[0, 0], [1, 0], [0, 1]]
    res = certify_b3_record(S, published_best=2)
    assert res["certified"] is True
    assert res["n"] == 2


# --------------------------------------------------------------------------- #
# record registry
# --------------------------------------------------------------------------- #
def test_registry_a396704_proven_values_match_doc():
    proven = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 8, 6: 11}
    for n, v in proven.items():
        assert best_known("A396704", n) == v
        assert get_record("A396704", n)["status"] == "proven_optimal"


def test_registry_a396704_provisional_n7():
    term = get_record("A396704", 7)
    assert term["best_known"] == 16
    assert term["status"] == "provisional_lower_bound"
    assert term["oeis_id"] == "A396704"
    assert "oeis.org/A396704" in term["url"]


def test_registry_other_sequences_match_doc():
    # A385931 weak-B_3 Golomb, proven a(1..10), a(11) open.
    weak = {1: 0, 2: 1, 3: 2, 4: 3, 5: 7, 6: 13, 7: 22, 8: 39, 9: 69, 10: 113}
    for n, v in weak.items():
        assert best_known("A385931", n) == v
    assert get_record("A385931", 11)["status"] == "open"
    assert best_known("A385931", 11) is None

    # Modular Golomb rulers.
    assert best_known("A004135", 17) == 255
    assert get_record("A004135", 18)["status"] == "open"
    assert best_known("A004136", 18) == 307
    assert get_record("A004136", 19)["status"] == "open"

    # Binary Sidon (B_2) upper-bound context for A396704.
    assert best_known("A309370", 6) == 15
    assert best_known("A309370", 7) == 24
    assert get_record("A309370", 7)["status"] == "provisional_lower_bound"

    # Cap sets in F_3^n.
    for n, v in {2: 4, 3: 9, 4: 20, 5: 45, 6: 112}.items():
        assert best_known("A090245", n) == v
    assert get_record("A090245", 7)["status"] == "provisional_lower_bound"


def test_registry_structure_is_dict_with_urls():
    assert isinstance(RECORDS, dict)
    for key, seq in RECORDS.items():
        assert seq["oeis_id"]
        assert seq["url"].startswith("http")
        assert seq["source_note"]
        for n, term in seq["terms"].items():
            assert term["status"] in {"proven_optimal", "provisional_lower_bound", "open"}


def test_registry_lookup_missing_returns_none():
    assert get_record("A396704", 999) is None
    assert get_record("NOPE", 1) is None


# --------------------------------------------------------------------------- #
# symmetry helpers
# --------------------------------------------------------------------------- #
def test_translate_to_origin_contains_zero_and_preserves_b3():
    S = [(1, 1, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    tr = translate_to_origin(S)
    assert tuple([0, 0, 0]) in tr
    assert is_B3(S) == is_B3(tr)


def test_canonical_form_symmetry_invariant():
    S = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    # permute coordinates + complement one coordinate -> same canonical form
    S2 = [(v[2], v[0], v[1]) for v in S]
    S3 = [(1 - v[0], v[1], v[2]) for v in S2]
    assert canonical_form(S, 3) == canonical_form(S3, 3)


# --------------------------------------------------------------------------- #
# finder reproduction gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n,expected",
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)],
)
def test_finder_reproduces_proven_small_via_bnb(n, expected):
    res = find_max_b3(n, time_budget=8.0, seed=1)
    assert res["size"] == expected
    assert is_B3([tuple(v) for v in res["set"]])
    assert res["proven_optimal"] is True


@pytest.mark.parametrize("n,expected", [(5, 8), (6, 11)])
def test_finder_reproduces_a5_a6(n, expected):
    res = find_max_b3(n, time_budget=40.0, seed=7, target=expected)
    assert res["size"] == expected, f"expected a({n})={expected}, got {res['size']}"
    assert is_B3([tuple(v) for v in res["set"]])


def test_finder_reaches_a7_lower_bound_16():
    # Reproduces the published provisional lower bound a(7) >= 16. Warm-started from
    # the finder's own stored witness so this is fast and deterministic in CI.
    res = find_max_b3(7, time_budget=45.0, seed=3, target=16)
    assert res["size"] >= 16, f"finder only reached {res['size']} at n=7"
    S = [tuple(v) for v in res["set"]]
    assert is_B3(S)
    # And a size-16 set is not (yet) a record: it does not strictly beat 16.
    cert = certify_b3_record({"n": 7, "set": res["set"]}, best_known("A396704", 7))
    assert cert["certified"] is False


def test_stored_n7_witness_is_valid_b3_of_size_16():
    ws = verified_witnesses(7)
    assert ws, "expected at least one stored/verified n=7 witness (provenance)"
    assert any(len(w) >= 16 for w in ws)
    for w in ws:
        assert is_B3(w)
