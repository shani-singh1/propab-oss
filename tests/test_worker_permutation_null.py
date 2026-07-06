"""Tests for the generic-worker label-permutation null (D2).

Covers the pure permutation-null helper and its interop with the read-only
core gate ``artifact_verification._survives_permutation``:

  (a) two real arrays with a genuine difference -> a low permutation_p is
      computed and _survives_permutation accepts it (at n >= 100);
  (b) two arrays with NO real difference -> high permutation_p, does not survive;
  (c) only one array / no raw arrays -> permutation_p absent (fail-closed, no
      fabricated null);
  (d) determinism under a fixed seed.

``import propab`` resolves to the MAIN checkout (not this worktree); that is fine
because these tests only READ ``_survives_permutation`` — the D2 change lives
entirely under ``services/worker/`` which DOES resolve to the worktree.
"""
from __future__ import annotations

import numpy as np

# Read-only import of the core survival gate (main checkout) — NOT edited by D2.
from propab.artifact_verification import EvidenceContext, _survives_permutation

from propab.verdict_pipeline import classify_evidence_type, run_verdict_pipeline

from services.worker.permutation_null import (
    DEFAULT_N_PERMUTATIONS,
    PermutationNullResult,
    compute_label_permutation_null,
    extract_two_group_arrays,
)
from services.worker.sub_agent_loop import attach_permutation_null_to_evidence


def _big_two_groups(diff: float, *, n: int = 130, seed: int = 3):
    rng = np.random.default_rng(seed)
    a = list(rng.normal(diff, 0.4, n))
    b = list(rng.normal(0.0, 0.4, n))
    return a, b


# ── (a) genuine difference: low p, and _survives_permutation accepts it ───────

def test_genuine_difference_low_p_and_survives():
    a, b = _big_two_groups(diff=1.0, n=130)
    res = compute_label_permutation_null(a, b)
    assert isinstance(res, PermutationNullResult)
    assert res.permutation_p < 0.01
    assert res.n_samples == 260
    assert res.n_permutations >= 1000

    # Interop: attach to an evidence dict exactly as the worker does and confirm
    # the read-only core gate accepts it.
    ev = res.to_evidence_fields()
    ctx = EvidenceContext(
        hypothesis_text="treatment beats baseline",
        p_value=0.002,
        n_samples=res.n_samples,
        metric_value=1.0,
    )
    verification = _survives_permutation(ctx, ev)
    assert verification.survived is True
    assert "outcome-permutation null" in verification.rationale


# ── (b) no difference: high p, does not survive ───────────────────────────────

def test_no_difference_high_p_not_survives():
    a, b = _big_two_groups(diff=0.0, n=120, seed=11)
    res = compute_label_permutation_null(a, b)
    assert res is not None
    assert res.permutation_p > 0.05  # no real effect -> null not beaten

    ev = res.to_evidence_fields()
    ctx = EvidenceContext(hypothesis_text="x", p_value=0.5, n_samples=res.n_samples)
    verification = _survives_permutation(ctx, ev)
    assert verification.survived is False


# ── (c) fail-closed: single / missing arrays never synthesize a null ──────────

def test_single_array_returns_none():
    a, _ = _big_two_groups(diff=1.0)
    assert compute_label_permutation_null(a, None) is None
    assert compute_label_permutation_null(None, a) is None


def test_too_short_array_returns_none():
    assert compute_label_permutation_null([0.1], [0.2, 0.3, 0.4]) is None
    assert compute_label_permutation_null([], [1.0, 2.0]) is None


def test_non_numeric_or_bool_arrays_return_none():
    # bools (bool is an int subclass) and mixed-type lists must be rejected —
    # we never silently coerce to force a null.
    assert compute_label_permutation_null([True, False, True], [1.0, 2.0, 3.0]) is None
    assert compute_label_permutation_null([1.0, "x", 3.0], [1.0, 2.0, 3.0]) is None


def test_extract_two_group_arrays_fail_closed():
    # Only a single array present -> None (cannot ground a two-group null).
    assert extract_two_group_arrays({"results_a": [1.0, 2.0, 3.0]}) is None
    assert extract_two_group_arrays({"values": [1.0, 2.0, 3.0]}) is None  # bootstrap single-array
    assert extract_two_group_arrays({}) is None
    assert extract_two_group_arrays(None) is None
    # Both arrays present under any recognised pair -> returned.
    got = extract_two_group_arrays({"results_a": [1.0, 2.0], "results_b": [3.0, 4.0]})
    assert got == ([1.0, 2.0], [3.0, 4.0])
    got2 = extract_two_group_arrays({"treatment": [5.0, 6.0], "baseline": [1.0, 2.0]})
    assert got2 == ([5.0, 6.0], [1.0, 2.0])
    got3 = extract_two_group_arrays(
        {"our_results": [0.4, 0.5], "baseline_results": [0.1, 0.2]}
    )
    assert got3 == ([0.4, 0.5], [0.1, 0.2])


def test_survives_permutation_fails_closed_without_null():
    # With NO permutation_p in the evidence, the core gate must not survive.
    ctx = EvidenceContext(hypothesis_text="x", p_value=0.001, n_samples=400)
    verification = _survives_permutation(ctx, {})
    assert verification.survived is False


def test_survives_permutation_small_n_fails_closed():
    # A strict permutation p at too-small n must still fail closed in the core gate.
    a, b = _big_two_groups(diff=1.0, n=20)
    res = compute_label_permutation_null(a, b)
    assert res is not None
    assert res.n_samples == 40  # < 100
    ctx = EvidenceContext(hypothesis_text="x", p_value=0.001, n_samples=res.n_samples)
    verification = _survives_permutation(ctx, res.to_evidence_fields())
    assert verification.survived is False


# ── (d) determinism under a fixed seed ────────────────────────────────────────

def test_determinism_under_seed():
    a, b = _big_two_groups(diff=0.6, n=100, seed=42)
    r1 = compute_label_permutation_null(a, b, seed=99)
    r2 = compute_label_permutation_null(a, b, seed=99)
    assert r1 is not None and r2 is not None
    assert r1.permutation_p == r2.permutation_p
    assert r1.observed_stat == r2.observed_stat
    # A different seed may give a (slightly) different Monte-Carlo estimate but
    # must still be a valid probability.
    r3 = compute_label_permutation_null(a, b, seed=100)
    assert 0.0 < r3.permutation_p <= 1.0


def test_default_n_permutations_is_at_least_1000():
    assert DEFAULT_N_PERMUTATIONS >= 1000
    a, b = _big_two_groups(diff=0.5, n=60)
    res = compute_label_permutation_null(a, b, n_permutations=100)  # floor-clamped
    assert res is not None
    assert res.n_permutations >= 1000


# ── Evidence-attachment wiring (attach_permutation_null_to_evidence) ──────────

def _stat_evidence_base() -> dict:
    """Evidence _build_evidence would produce for a two-group compare: a real
    significance p_value/effect_size but no accuracy-style metric yet."""
    return {
        "metric_value": None,
        "baseline_value": None,
        "p_value": 0.002,
        "effect_size": 1.2,
        "relevance_score": 0.5,
        "n_metric_steps": 0,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "stat_input_provenance": "computed",
    }


def test_attach_null_confirms_generic_two_group_experiment():
    a, b = _big_two_groups(diff=1.0, n=140)
    ev = _stat_evidence_base()
    attach_permutation_null_to_evidence(ev, [(a, b)])

    # Null attached from the real arrays.
    assert ev["permutation_p"] < 0.01
    assert ev["n_samples"] == 280
    # Group-mean metric filled honestly (no accuracy metric existed).
    assert ev["metric_value"] is not None
    assert ev["metric_from_permutation_groups"] is True
    # Provenance tag preserved, not stripped.
    assert ev["stat_input_provenance"] == "computed"

    # Full verdict pipeline confirms a generic statistical result WITH a real null.
    assert classify_evidence_type(ev) == "statistical"
    verdict, _c, _r = run_verdict_pipeline(ev, campaign_context={"min_metric_steps": 1})
    assert verdict == "confirmed"


def test_attach_null_absent_arrays_stays_inconclusive():
    ev = _stat_evidence_base()
    ev["metric_value"] = 0.9
    ev["baseline_value"] = 0.8
    ev["delta"] = 0.1
    ev["n_metric_steps"] = 3
    # No captured arrays -> fail-closed: nothing attached.
    attach_permutation_null_to_evidence(ev, [])
    assert "permutation_p" not in ev

    verdict, _c, reason = run_verdict_pipeline(ev, campaign_context={"min_metric_steps": 1})
    assert verdict == "inconclusive"
    assert "holdout" in reason.lower() or "null" in reason.lower()


def test_attach_null_no_difference_does_not_confirm():
    # Two groups drawn from the SAME distribution: the observed mean difference is
    # small, so the permutation null is not beaten and the result must not confirm.
    a, b = _big_two_groups(diff=0.0, n=130, seed=0)
    ev = _stat_evidence_base()
    ev["p_value"] = 0.4  # honest: no real effect
    attach_permutation_null_to_evidence(ev, [(a, b)])
    assert ev["permutation_p"] >= 0.01  # does NOT clear the < 0.01 survival bar
    verdict, _c, _r = run_verdict_pipeline(ev, campaign_context={"min_metric_steps": 1})
    assert verdict == "inconclusive"


def test_attach_null_preserves_agent_literal_provenance():
    a, b = _big_two_groups(diff=1.0, n=120)
    ev = _stat_evidence_base()
    ev["stat_input_provenance"] = "agent_literal"
    attach_permutation_null_to_evidence(ev, [(a, b)])
    # The null is still computed on the (untrusted) arrays, but the provenance tag
    # MUST remain so a later gate can reject the verdict.
    assert ev["permutation_p"] < 0.01
    assert ev["stat_input_provenance"] == "agent_literal"


def test_attach_null_does_not_overwrite_real_metric():
    a, b = _big_two_groups(diff=1.0, n=120)
    ev = _stat_evidence_base()
    ev["metric_value"] = 0.93  # a real accuracy metric already found
    ev["baseline_value"] = 0.90
    attach_permutation_null_to_evidence(ev, [(a, b)])
    assert ev["metric_value"] == 0.93  # untouched
    assert "metric_from_permutation_groups" not in ev
    assert ev["permutation_p"] < 0.01  # null still attached


def test_attach_null_picks_largest_comparison():
    small_a, small_b = _big_two_groups(diff=1.0, n=10)
    big_a, big_b = _big_two_groups(diff=1.0, n=150, seed=8)
    ev = _stat_evidence_base()
    attach_permutation_null_to_evidence(ev, [(small_a, small_b), (big_a, big_b)])
    assert ev["n_samples"] == 300  # the 150+150 comparison, not 10+10
