"""B_3-binary-cube (OEIS A396704) wiring into the math_combinatorics verifier.

The discovery kernel lives in math_combinatorics/discovery/; this checks it is
correctly ROUTED and gated from run_combinatorics_experiment: a B_3 hypothesis
computes a real, independently-witnessed size, compares honestly to best-known, and
only surfaces a discovery when the paranoid certifier passes.
"""
from __future__ import annotations

from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin
from propab.domain_modules.math_combinatorics.verifier import (
    _extract_b3_dim,
    _is_b3_binary_cube_hypothesis,
    run_combinatorics_experiment,
)

_DISCOVERY = "propab.domain_modules.math_combinatorics.discovery"


def test_detector_positive_and_negative():
    assert _is_b3_binary_cube_hypothesis("Maximize a B_3 set in {0,1}^7")
    assert _is_b3_binary_cube_hypothesis("threefold sums distinct set in {0,1}^8")
    assert _is_b3_binary_cube_hypothesis("improve OEIS A396704 at n=7")
    # A plain Sidon/{0,1}^n claim must NOT mis-route to B_3.
    assert not _is_b3_binary_cube_hypothesis("Maximize a Sidon set in {0,1}^7")
    assert not _is_b3_binary_cube_hypothesis("cap set in F_3^7")


def test_extract_dim():
    assert _extract_b3_dim("B_3 set in {0,1}^7") == 7
    assert _extract_b3_dim("B_3 set with n = 5") == 5
    assert _extract_b3_dim("B_3 set, no dim given") == 7  # default
    assert _extract_b3_dim("B_3 in {0,1}^40") == 9  # clamped to B3_MAX_N


def test_plugin_on_topic_accepts_b3():
    plugin = MathCombinatoricsPlugin()
    assert plugin.hypothesis_on_topic(
        "Find a B_3 set of size 17 in {0,1}^7 (A396704)",
        methodology="iterated local search with symmetry reduction",
    )


def test_runner_matches_proven_value_is_honest_not_discovery():
    # n=4: a(4)=6 is PROVEN optimal. Reproducing it is a rediscovery, never a record.
    res = run_combinatorics_experiment(
        {"statement": "Maximum B_3 set in {0,1}^4", "text": "Maximum B_3 set in {0,1}^4"},
        ["b3_binary_cube_size"],
    )
    assert res["metric_name"] == "b3_binary_cube_size"
    assert res["b3_size"] == 6
    assert res["vs_best_known"] == "matches_best_known"
    assert res["trivial_rediscovery"] is True
    assert res["discovery_worthy"] is False
    assert int(res["verified_true_steps"]) == 0


def test_runner_certified_record_path(monkeypatch):
    # Feed a REAL size-16 B_3 witness (from the finder's provenance store) but pretend
    # the best-known is only 10, so 16 > best-known becomes a candidate record. The
    # certifier runs on a genuine B_3 set, so this exercises the true positive path
    # without a 20s search and without fabricating a witness.
    import propab.domain_modules.math_combinatorics.discovery as disc
    from propab.domain_modules.math_combinatorics.discovery.known_witnesses import WITNESSES

    real16 = [list(v) for v in WITNESSES[7][0]]
    assert len(real16) == 16

    def fake_find_max_b3(n, *, time_budget=25.0, **kw):
        return {"n": 7, "size": 16, "set": real16, "method": "test_stub", "proven_optimal": False}

    monkeypatch.setattr(disc, "find_max_b3", fake_find_max_b3)
    monkeypatch.setattr(disc, "best_known", lambda seq, n: 10)
    monkeypatch.setattr(disc, "record_status", lambda seq, n: "provisional_lower_bound")

    res = run_combinatorics_experiment(
        {"statement": "Find a record B_3 set in {0,1}^7 (A396704)"},
        ["b3_binary_cube_size"],
    )
    assert res["vs_best_known"] == "exceeds_best_known"
    assert res["discovery_worthy"] is True
    assert int(res["verified_true_steps"]) == 1
    assert res["certification"]["certified"] is True
    assert res["record_witness_json"]["claimed_size"] == 16
    # deterministic + explicit proof method => confirmable by the verdict pipeline
    assert res["deterministic"] is True
    assert res["verification_method"] == "combinatorial_computation"


def test_runner_uncertified_over_best_known_not_surfaced(monkeypatch):
    # A size that "beats" best-known but FAILS certification (here: a non-B_3 set)
    # must never be surfaced as a record.
    import propab.domain_modules.math_combinatorics.discovery as disc

    bogus = [[0, 0, 0, 0, 0, 0, 0]] * 3 + [[1, 0, 0, 0, 0, 0, 0]] * 9  # duplicates, not B_3

    monkeypatch.setattr(
        disc, "find_max_b3",
        lambda n, *, time_budget=25.0, **kw: {"n": 7, "size": 12, "set": bogus, "method": "stub", "proven_optimal": False},
    )
    monkeypatch.setattr(disc, "best_known", lambda seq, n: 10)
    monkeypatch.setattr(disc, "record_status", lambda seq, n: "provisional_lower_bound")

    res = run_combinatorics_experiment(
        {"statement": "B_3 set in {0,1}^7"}, ["b3_binary_cube_size"]
    )
    assert res["vs_best_known"] == "exceeds_best_known_uncertified"
    assert res["discovery_worthy"] is False
    assert int(res["verified_true_steps"]) == 0
