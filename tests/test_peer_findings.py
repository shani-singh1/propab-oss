from __future__ import annotations

import pytest

from services.worker.peer_findings import build_peer_finding_payload


def test_build_peer_finding_payload_confirmed():
    result = {
        "hypothesis_id": "h1",
        "verdict": "confirmed",
        "confidence": 0.82,
        "key_finding": "Pre-norm significantly improves gradient stability.",
        "learned": "Use statistical_significance with repeated runs.",
        "failure_reason": None,
    }
    payload = build_peer_finding_payload(result)
    assert payload["verdict"] == "confirmed"
    assert payload["confidence"] == 0.82
    assert "Pre-norm" in payload["key_finding"]
    assert payload["failure_reason"] is None


def test_build_peer_finding_payload_inconclusive():
    result = {
        "hypothesis_id": "h2",
        "verdict": "inconclusive",
        "confidence": 0.3,
        "key_finding": None,
        "learned": None,
        "failure_reason": "no significance evidence",
    }
    payload = build_peer_finding_payload(result)
    assert payload["verdict"] == "inconclusive"
    assert payload["key_finding"] == ""
    assert payload["failure_reason"] == "no significance evidence"


def test_build_peer_finding_payload_truncates_long_fields():
    result = {
        "hypothesis_id": "h3",
        "verdict": "refuted",
        "confidence": 0.7,
        "key_finding": "x" * 500,
        "learned": "y" * 500,
        "failure_reason": None,
    }
    payload = build_peer_finding_payload(result)
    assert len(payload["key_finding"]) <= 300
    assert len(payload["learned"]) <= 300
