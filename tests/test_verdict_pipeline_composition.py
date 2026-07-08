"""Composition tests for run_verdict_pipeline (fixes.md Task 0)."""
from __future__ import annotations

from propab.verdict_pipeline import (
    ood_gate_stage,
    run_verdict_pipeline,
    scope_integrity_stage,
)


def test_ml_evidence_with_permutation_null_confirms():
    """A statistical result confirms ONLY when it carries a passing, independent
    adversarial null. This case supplies an outcome-permutation null (p<0.01) at
    large n, so it survives the artifact gate and confirms.

    (Replaces the old ``test_ml_evidence_confirms``, which asserted the V2 bug:
    a bare ``verified_true_steps>=2`` counter with no proof method and no null
    used to bypass the artifact gate and "confirm". That is no longer honest —
    such evidence is now routed through the gate and fails closed.)
    """
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.001,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "permutation_p": 0.002,
        "n_samples": 400,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "metric_direction": "higher_is_better",
        "verdict_reason": "significance gate passed",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
    assert confidence >= 0.85


def test_bare_counter_no_longer_bypasses_artifact_gate():
    """V2: verified_true_steps>=2 with method 'significance' (or no method) must
    NOT be treated as deterministic and must NOT bypass the artifact gate. With
    no adversarial null present, it fails closed to inconclusive."""
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "verified_true_steps": 3,
        "verified_false_steps": 0,
        "verification_method": "significance",
        "metric_direction": "higher_is_better",
    }
    verdict, _confidence, _reason = run_verdict_pipeline(evidence)
    assert verdict != "confirmed"
    assert verdict == "inconclusive"


def test_statistical_with_permutation_null_absent_stays_inconclusive():
    """A statistical result WITHOUT any real null stays inconclusive (fail-closed):
    a bare significance p-value can never confirm on its own."""
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.001,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, _confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"
    assert "holdout" in reason.lower() or "null" in reason.lower()


def test_statistical_permutation_null_small_n_stays_inconclusive():
    """Even a strict permutation p fails closed when n is too small to trust it."""
    evidence = {
        "metric_value": 0.94,
        "p_value": 0.001,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "permutation_p": 0.001,
        "n_samples": 30,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, _confidence, _reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"


def test_deterministic_math_proof_confirms():
    evidence = {
        "verified_true_steps": 2,
        "verified_false_steps": 0,
        "deterministic": True,
        "verification_method": "symbolic_proof",
        "verdict_reason": "proof verified",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
    assert confidence >= 0.90


# ── Deterministic proof must survive the OOD / scope gates ───────────────────
# Regression: the OOD gate (and the scope-integrity gate) used to fire on ANY
# confirmed verdict once a hypothesis text was present, with no shape guard. A
# deterministic proof carries no OOD/transfer surface, so parse_scope found no
# scope and check_ood_evidence returned "no OOD evidence" -> the proof was
# silently downgraded to inconclusive. With no hyp text it confirmed; with one it
# broke. The plugin verdict path already restricts OOD/scope to lofo/statistical
# evidence (F1 block); run_verdict_pipeline must do the same so the deterministic
# (math/coding_theory) confirm path stays confirmable end-to-end.


def _deterministic_proof_evidence() -> dict:
    return {
        "verified_true_steps": 2,
        "verified_false_steps": 0,
        "verification_method": "symbolic_proof",
        "verdict_reason": "proof verified",
    }


def test_deterministic_proof_confirms_with_hypothesis_text():
    """The core regression: a proof + a hypothesis text must STILL confirm
    (previously the OOD gate downgraded it to inconclusive)."""
    ev = _deterministic_proof_evidence()
    verdict, confidence, _reason = run_verdict_pipeline(
        ev, hypothesis={"text": "For all n >= 1, the construction achieves the bound"},
    )
    assert verdict == "confirmed"
    assert confidence >= 0.90


def test_deterministic_proof_confirms_with_campaign_context_hyp_text():
    """Same via campaign_context.hyp_text (the plugin except-branch shape)."""
    ev = {
        "verified_true_steps": 3,
        "verification_method": "exhaustive_enumeration",
        "verdict_reason": "enumerated",
    }
    verdict, _confidence, _reason = run_verdict_pipeline(
        ev,
        campaign_context={"hyp_text": "A [12,4,6] code exists", "test_methodology": ""},
    )
    assert verdict == "confirmed"


def test_ood_stage_skips_deterministic_evidence():
    """Unit: ood_gate_stage is a no-op for deterministic evidence even with a
    hypothesis text + methodology present."""
    ev = _deterministic_proof_evidence()
    verdict, conf, reason = ood_gate_stage(
        ev, "confirmed", 0.95, "proof verified",
        hypothesis={"text": "some claim with no scope labels"},
        campaign_context={"test_methodology": "run the proof"},
    )
    assert verdict == "confirmed"
    assert conf == 0.95
    assert reason == "proof verified"


def test_scope_integrity_stage_skips_deterministic_evidence():
    """Unit: a pre-attached scope FAIL must not collapse a deterministic proof."""
    ev = dict(_deterministic_proof_evidence())
    ev["scope_gate_result"] = "FAIL"
    ev["scope_integrity"] = {"reason": "missing declared scope"}
    verdict, _conf, _reason = scope_integrity_stage(ev, "confirmed", 0.95, "proof verified")
    assert verdict == "confirmed"


def test_ood_stage_still_gates_statistical_without_ood():
    """Over-scoping guard: a statistical confirm with a declared OOD test but no
    passing OOD evidence is STILL downgraded (the deterministic skip must not leak
    into distributional evidence)."""
    ev = {
        "metric_value": 0.94,
        "p_value": 0.001,
        "n_metric_steps": 3,
    }
    verdict, _conf, _reason = ood_gate_stage(
        ev, "confirmed", 0.9, "sig passed",
        campaign_context={
            "hyp_text": "Accuracy improves",
            "test_methodology": "OOD test: hold out family B and evaluate transfer",
        },
    )
    assert verdict == "inconclusive"


def test_lofo_evidence_with_null_stats_confirms():
    evidence = {
        "lofo_r2": 0.15,
        "label_shuffle_null_p95": 0.11,
        "label_shuffle_permutation_p": 0.02,
        "lofo_gap": 0.45,
        "family_leakage_confirmed": False,
        "verified_true_steps": 2,
        "p_value": 0.03,
        "metric_value": 0.15,
        "n_samples": 120,
        "n_families": 7,
        "methodology": "LOFO",
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"


def test_generic_significance_only_goes_inconclusive_not_refuted():
    evidence = {
        "metric_value": 0.88,
        "p_value": 0.03,
        "verified_true_steps": 1,
        "verified_false_steps": 0,
        "n_metric_steps": 3,
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"
    assert "replication" in reason.lower() or "holdout" in reason.lower()


def test_statistical_confirmed_downgrades_without_holdout():
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, confidence, reason = run_verdict_pipeline(evidence)
    assert verdict == "inconclusive"
    assert "holdout" in reason.lower()


def test_worker_path_integration_composition():
    """End-to-end through pipeline with worker-like campaign context."""
    evidence = {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.01,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
    }
    verdict, confidence, reason = run_verdict_pipeline(
        evidence,
        hypothesis={"text": "MNIST accuracy improves with deeper CNN"},
        campaign_context={
            "hyp_text": "MNIST accuracy improves with deeper CNN",
            "domain_bucket": "ml",
            "min_metric_steps": 2,
        },
    )
    assert verdict == "inconclusive"
    assert verdict != "refuted"


# ── W1b: stat_input_provenance enforcement ───────────────────────────────────
# A statistical confirm requires BOTH a passing adversarial null AND that the
# numbers the significance/permutation test ran on were NOT typed directly by
# the LLM agent ("agent_literal"). The guard lives in the statistical branch of
# artifact_gate_stage (verdict_pipeline.py) and is scoped strictly to that
# branch — deterministic-proof and LOFO paths are intentionally unaffected.


def _stat_confirming_evidence() -> dict:
    """Statistical evidence that WOULD confirm: passing outcome-permutation null
    (p<0.01) at large n, supporting metric direction, replicated."""
    return {
        "metric_value": 0.94,
        "baseline_value": 0.90,
        "p_value": 0.001,
        "effect_size": 0.8,
        "delta": 0.04,
        "relevance_score": 0.5,
        "n_metric_steps": 3,
        "permutation_p": 0.001,
        "n_samples": 400,
        "verified_true_steps": 0,
        "verified_false_steps": 0,
        "metric_direction": "higher_is_better",
    }


def test_w1b_statistical_agent_literal_inputs_blocked():
    """Passing null + n>=100 but inputs typed by the agent -> fail closed."""
    evidence = _stat_confirming_evidence()
    evidence["stat_input_provenance"] = "agent_literal"
    verdict, _confidence, reason = run_verdict_pipeline(
        evidence, campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "inconclusive"
    assert "stat_inputs_agent_literal_untrusted" in reason


def test_w1b_statistical_computed_inputs_confirm():
    """Same passing null, but inputs were computed in-sandbox -> confirms."""
    evidence = _stat_confirming_evidence()
    evidence["stat_input_provenance"] = "computed"
    verdict, confidence, _reason = run_verdict_pipeline(
        evidence, campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "confirmed"
    assert confidence >= 0.85


def test_w1b_statistical_unknown_inputs_confirm():
    """Deliberate policy: we fail closed ONLY on the KNOWN-untrusted case. An
    'unknown' provenance is not evidence of fabrication, so it still confirms
    when the null passes (documents the choice not to over-block)."""
    evidence = _stat_confirming_evidence()
    evidence["stat_input_provenance"] = "unknown"
    verdict, _confidence, _reason = run_verdict_pipeline(
        evidence, campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "confirmed"


def test_w1b_statistical_missing_provenance_confirms():
    """Absent provenance field (legacy evidence) behaves like 'unknown': not
    blocked, confirms on a passing null."""
    evidence = _stat_confirming_evidence()
    assert "stat_input_provenance" not in evidence
    verdict, _confidence, _reason = run_verdict_pipeline(
        evidence, campaign_context={"min_metric_steps": 1},
    )
    assert verdict == "confirmed"


def test_w1b_deterministic_proof_unaffected_by_agent_literal():
    """Guard is scoped to the statistical branch only: a deterministic proof
    confirms even when stat_input_provenance == 'agent_literal' (over-scoping
    guard)."""
    evidence = {
        "verified_true_steps": 2,
        "verified_false_steps": 0,
        "deterministic": True,
        "verification_method": "symbolic_proof",
        "stat_input_provenance": "agent_literal",
    }
    verdict, confidence, _reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
    assert confidence >= 0.90


def test_w1b_lofo_unaffected_by_agent_literal():
    """LOFO (label-shuffle) evidence computes in-sandbox and must confirm even
    with stat_input_provenance == 'agent_literal' (over-scoping guard)."""
    evidence = {
        "lofo_r2": 0.15,
        "label_shuffle_null_p95": 0.11,
        "label_shuffle_permutation_p": 0.02,
        "lofo_gap": 0.45,
        "family_leakage_confirmed": False,
        "verified_true_steps": 2,
        "p_value": 0.03,
        "metric_value": 0.15,
        "n_samples": 120,
        "n_families": 7,
        "methodology": "LOFO",
        "stat_input_provenance": "agent_literal",
    }
    verdict, _confidence, _reason = run_verdict_pipeline(evidence)
    assert verdict == "confirmed"
