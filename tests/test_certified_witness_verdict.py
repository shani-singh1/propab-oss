"""Certified-witness evidence -> verdict (the S0 false-negative fix).

A general-agent worker running the trusted B_3 tools emits {certified, size,
is_record, best_known}, not the ML metric shape. enrich_certified_witness_evidence
maps that onto the deterministic evidence shape so the orchestrator's authoritative
verdict (verdict_pipeline.compute_authoritative_verdict, math plugin) CONFIRMS a
genuine record and stays inconclusive for a rediscovery / below-record set — no false
positives, and (the bug this fixes) no false negatives that would MISS a real record.
"""
from __future__ import annotations

from propab.domain_modules.registry import resolve_domain_plugin
from propab.verdict_pipeline import (
    classify_evidence_type,
    compute_authoritative_verdict,
    is_recomputable_evidence,
)
from services.worker.sub_agent_loop import enrich_certified_witness_evidence

B3_Q = "Construct a B_3 Sidon-type set in the binary cube {0,1}^n beating the best-known size a(7)=16."


def _witness(size: int, is_record: bool, certified: bool = True, best_known: int | None = None) -> dict:
    return {
        "object": "b3_binary_cube", "n": 8, "size": size,
        "certified": certified, "is_record": is_record, "best_known": best_known,
    }


def _enrich(outputs: list[dict]) -> dict:
    ev = {"n_metric_steps": 0, "verified_true_steps": 0, "verified_false_steps": 0, "relevance_score": 0.5}
    enrich_certified_witness_evidence(ev, outputs)
    return ev


# ── shape mapping ─────────────────────────────────────────────────────────────

def test_record_maps_to_deterministic_metric():
    ev = _enrich([_witness(20, is_record=True, best_known=19)])
    assert ev["verification_method"] == "combinatorial_computation"
    assert ev["metric_value"] == 20.0
    assert ev["n_metric_steps"] >= 1
    assert ev["verified_true_steps"] >= 1
    assert ev["discovery_worthy"] is True
    assert classify_evidence_type(ev) == "deterministic"
    assert is_recomputable_evidence(ev) is True


def test_rediscovery_never_verified_true():
    ev = _enrich([_witness(16, is_record=False, best_known=16)])
    assert ev["verified_true_steps"] == 0
    assert ev.get("discovery_worthy") is False
    assert ev.get("trivial_rediscovery") is True
    assert ev["metric_value"] == 16.0  # still metric-bearing (not "no metric steps")


def test_below_known_never_verified_true():
    ev = _enrich([_witness(18, is_record=False, best_known=19)])
    assert ev["verified_true_steps"] == 0
    assert ev.get("discovery_worthy") is False
    assert ev.get("trivial_rediscovery") is not True
    assert ev["metric_value"] == 18.0


def test_uncertified_never_verified_true():
    ev = _enrich([_witness(21, is_record=False, certified=False, best_known=19)])
    assert ev["verified_true_steps"] == 0
    assert ev.get("discovery_worthy") is False


def test_prefers_the_record_witness_among_several():
    ev = _enrich([
        _witness(16, is_record=False, best_known=16),
        _witness(20, is_record=True, best_known=19),
        _witness(18, is_record=False, best_known=19),
    ])
    assert ev["verified_true_steps"] >= 1
    assert ev["witness_size"] == 20


def test_no_witness_is_noop():
    ev = {"n_metric_steps": 0, "verified_true_steps": 0}
    enrich_certified_witness_evidence(ev, [{"val_accuracy": 0.9}, {"p_value": 0.01}])
    assert "verification_method" not in ev
    assert ev["verified_true_steps"] == 0


# ── authoritative verdict (exactly what the orchestrator computes) ─────────────

def test_b3_question_routes_to_math_plugin():
    assert resolve_domain_plugin(question=B3_Q) is not None


def _authoritative(ev: dict) -> tuple[str, float, str]:
    return compute_authoritative_verdict(
        plugin=resolve_domain_plugin(question=B3_Q),
        hypothesis={"text": B3_Q, "test_methodology": ""},
        evidence=dict(ev),
        campaign_context={"question": B3_Q, "min_metric_steps": 2},
    )


def test_authoritative_confirms_a_real_record():
    # THE FIX: a certified set that beats the best-known must be CONFIRMED, not a
    # false-negative inconclusive.
    verdict, conf, reason = _authoritative(_enrich([_witness(20, is_record=True, best_known=19)]))
    assert verdict == "confirmed", (verdict, reason)
    assert conf >= 0.9


def test_authoritative_inconclusive_for_rediscovery():
    verdict, _c, reason = _authoritative(_enrich([_witness(16, is_record=False, best_known=16)]))
    assert verdict == "inconclusive", (verdict, reason)


def test_authoritative_inconclusive_for_below_known():
    verdict, _c, reason = _authoritative(_enrich([_witness(18, is_record=False, best_known=19)]))
    assert verdict == "inconclusive", (verdict, reason)


def test_authoritative_not_confirmed_for_uncertified_claim():
    # A tool that claims a big size but is NOT certified must never confirm.
    verdict, _c, reason = _authoritative(_enrich([_witness(21, is_record=False, certified=False, best_known=19)]))
    assert verdict != "confirmed", (verdict, reason)
