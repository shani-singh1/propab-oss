"""Regression: classify_verification_method must read structured counters, not substrings.

The old implementation did ``"verified_true" in low``, which matched
``"verified_true_steps": 0`` (which means NOT verified) and mislabeled refuted /
statistical nodes as ``symbolic_identity``. Likewise ``"verified_false" in low``
matched ``"verified_false_steps": 0``.
"""
from __future__ import annotations

import json

from services.orchestrator.campaign_diagnostics import classify_verification_method


def test_statistical_node_with_zero_verified_steps_is_not_symbolic_identity():
    ev = json.dumps({
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "p_value": 0.2,
        "metric_value": 0.31,
        "label_shuffle_null_p": 0.2,
    })
    assert classify_verification_method(ev) == "statistical"


def test_zero_verified_false_steps_is_not_counterexample():
    ev = json.dumps({"verified_true_steps": 0, "verified_false_steps": 0, "metric_value": 0.1})
    # Neither a counterexample (vf==0) nor symbolic_identity (vt==0).
    assert classify_verification_method(ev) in {"statistical", "unknown"}
    assert classify_verification_method(ev) != "counterexample"


def test_genuine_verified_true_is_symbolic_or_finite_scan():
    ev = json.dumps({"verified_true_steps": 1, "verified_false_steps": 0, "sweep": "exhaustive scan up to n"})
    assert classify_verification_method(ev) in {"symbolic_identity", "finite_scan"}


def test_genuine_verified_false_is_counterexample():
    ev = json.dumps({"verified_true_steps": 0, "verified_false_steps": 1, "counterexample": [1, 2, 3]})
    assert classify_verification_method(ev) == "counterexample"


def test_non_json_evidence_falls_back_to_literal():
    assert classify_verification_method('result "verified": true via certificate') == "symbolic_identity"
    assert classify_verification_method("p_value=0.01 effect_size=0.5") == "statistical"
    assert classify_verification_method("") == "unknown"
