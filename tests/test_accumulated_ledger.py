from __future__ import annotations

import pytest

from services.orchestrator.accumulated_ledger import AccumulatedLedger, RoundSummary


def _result(hid: str, verdict: str, confidence: float = 0.7) -> dict:
    return {
        "hypothesis_id": hid,
        "verdict": verdict,
        "confidence": confidence,
        "key_finding": f"finding for {hid}" if verdict == "confirmed" else None,
        "evidence_summary": f"evidence for {hid}",
    }


def test_add_confirmed():
    ledger = AccumulatedLedger()
    ledger.add_result(_result("h1", "confirmed"))
    assert "h1" in ledger.confirmed
    assert ledger.total_confirmed == 1


def test_add_refuted():
    ledger = AccumulatedLedger()
    ledger.add_result(_result("h1", "refuted"))
    assert "h1" in ledger.refuted
    assert ledger.total_refuted == 1


def test_add_inconclusive():
    ledger = AccumulatedLedger()
    ledger.add_result(_result("h1", "inconclusive"))
    assert "h1" in ledger.inconclusive
    assert ledger.total_inconclusive == 1


def test_no_duplicate_entries():
    ledger = AccumulatedLedger()
    ledger.add_result(_result("h1", "confirmed"))
    ledger.add_result(_result("h1", "confirmed"))  # duplicate
    assert ledger.confirmed.count("h1") == 1


def test_merge_round_basic():
    ledger = AccumulatedLedger()
    results = [
        _result("h1", "confirmed"),
        _result("h2", "inconclusive"),
        _result("h3", "refuted"),
    ]
    summary = ledger.merge_round(0, "round-0", results)
    assert len(summary.confirmed) == 1
    assert len(summary.inconclusive) == 1
    assert len(summary.refuted) == 1
    assert not summary.all_inconclusive()


def test_merge_round_all_inconclusive():
    ledger = AccumulatedLedger()
    results = [_result("h1", "inconclusive"), _result("h2", "inconclusive")]
    summary = ledger.merge_round(0, "round-0", results)
    assert summary.all_inconclusive()


def test_marginal_return_first_round():
    ledger = AccumulatedLedger()
    ledger.merge_round(0, "round-0", [_result("h1", "confirmed")])
    assert ledger.marginal_return(0) == 1.0


def test_marginal_return_improves_with_new_confirmed():
    ledger = AccumulatedLedger()
    ledger.merge_round(0, "r0", [_result("h1", "inconclusive", 0.3)])
    ledger.merge_round(1, "r1", [_result("h2", "confirmed", 0.8)])
    mr = ledger.marginal_return(1)
    assert mr > 0


def test_marginal_return_zero_on_stale():
    ledger = AccumulatedLedger()
    ledger.merge_round(0, "r0", [_result("h1", "inconclusive", 0.3)])
    ledger.merge_round(1, "r1", [_result("h2", "inconclusive", 0.3)])
    mr = ledger.marginal_return(1)
    assert mr >= 0.0


def test_summary_for_hypothesis_generator_empty():
    ledger = AccumulatedLedger()
    summary = ledger.summary_for_hypothesis_generator()
    assert "No prior round" in summary


def test_summary_for_hypothesis_generator_with_rounds():
    ledger = AccumulatedLedger()
    ledger.merge_round(0, "r0", [_result("h1", "confirmed"), _result("h2", "inconclusive")])
    summary = ledger.summary_for_hypothesis_generator()
    assert "Round 0" in summary
    assert "confirmed" in summary


def test_to_dict_and_from_dict_roundtrip():
    ledger = AccumulatedLedger()
    ledger.merge_round(0, "r0", [_result("h1", "confirmed"), _result("h2", "refuted")])
    data = ledger.to_dict()
    restored = AccumulatedLedger.from_dict(data)
    assert restored.total_confirmed == 1
    assert restored.total_refuted == 1
    assert len(restored.round_summaries) == 1
