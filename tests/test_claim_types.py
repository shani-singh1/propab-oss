"""Tests for claim typing (fixes.md P0.1)."""

from propab.claim_types import (
    CLAIM_COUNTEREXAMPLE,
    CLAIM_FINITE_VERIFIED,
    CLAIM_PERFORMANCE,
    CLAIM_STATISTICAL,
    CLAIM_SYMBOLIC,
    build_finding_object,
    classify_claim_type,
    extract_mechanism,
)


def test_classify_finite_verification() -> None:
    ev = {"verified_true_steps": 2, "verdict_reason": "scan up to n=1000000"}
    assert classify_claim_type(ev, "confirmed", hypothesis_text="exhaustive scan for n ≤ 1e6") == (
        CLAIM_FINITE_VERIFIED
    )


def test_classify_counterexample() -> None:
    ev = {"verified_false_steps": 1}
    assert classify_claim_type(ev, "refuted") == CLAIM_COUNTEREXAMPLE


def test_classify_statistical_vs_performance() -> None:
    stat_ev = {"n_metric_steps": 3, "p_value": 0.01, "verdict_reason": "significant"}
    assert classify_claim_type(stat_ev, "confirmed", hypothesis_text="effect on accuracy") == (
        CLAIM_STATISTICAL
    )
    perf_ev = {"n_metric_steps": 2, "verdict_reason": "faster"}
    assert classify_claim_type(perf_ev, "confirmed", hypothesis_text="SIMD cache throughput") == (
        CLAIM_PERFORMANCE
    )


def test_classify_symbolic_default() -> None:
    ev = {"verified_true_steps": 1, "verdict_reason": "check passed"}
    assert classify_claim_type(ev, "confirmed", hypothesis_text="algebraic congruence mod 8") == CLAIM_SYMBOLIC


def test_finding_object_and_mechanism() -> None:
    ev = {"verified_true_steps": 2, "verdict_reason": "reproduced", "p_value": None}
    mech = extract_mechanism(ev, claim_type=CLAIM_SYMBOLIC, hypothesis_text="mod 4 pattern")
    assert mech and "Deterministic verification" in mech
    finding = build_finding_object(
        claim="mod 4 residue class",
        claim_type=CLAIM_SYMBOLIC,
        evidence=ev,
        confidence=0.95,
        verification_method="symbolic_identity",
        theme="residue_class",
        mechanism=mech,
    )
    assert finding["claim_type"] == CLAIM_SYMBOLIC
    assert finding["theme"] == "residue_class"
