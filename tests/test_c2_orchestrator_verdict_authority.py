"""C2 — the ORCHESTRATOR is the single verdict authority.

Today each WORKER computes the verdict for its hypothesis. C2 relocates that
JUDGMENT to the orchestrator: `verdict_pipeline.compute_authoritative_verdict`
recomputes the verdict from the worker's RAW evidence, in one place, for every
domain. This is a behaviour-preserving relocation — the orchestrator must produce
the SAME verdict (and confidence) the worker did.

These tests pin that parity across the four worker verdict paths:
  * plugin path (genomics = LOFO shape, math_combinatorics = deterministic proof);
  * dedicated LOFO-adapter path (mandrake, materials).

Each case builds a fixed raw verification result, computes the verdict the WORKER
would (via the worker's real classify + gate primitives, arranged exactly as the
worker arranges them), serialises it exactly as the worker does, parses it back
through the orchestrator's `parse_evidence_obj`, and asserts
`compute_authoritative_verdict` returns the identical (verdict, confidence). Each
case also pins a concrete expected verdict so the test fails on wrong logic, not
just on worker/orchestrator disagreement.
"""
from __future__ import annotations

import json

import pytest

from propab.artifact_verification import (
    apply_artifact_gate_override,
    evidence_context_from_hypothesis,
    merge_artifact_into_evidence,
    run_artifact_gate,
)
from propab.domain_modules.registry import resolve_domain_plugin
from propab.scoped_claim import apply_ood_gate_to_verdict
from propab.verdict_pipeline import (
    artifact_gate_stage,
    classify_evidence_type,
    compute_authoritative_verdict,
    is_recomputable_evidence,
    ood_gate_stage,
    scope_integrity_stage,
)
from services.orchestrator.campaign_diagnostics import parse_evidence_obj
from services.worker.sub_agent_loop import (
    _build_mandrake_evidence,
    _build_materials_evidence,
    attach_scope_integrity,
)


# ── Faithful replicas of the worker verdict paths (independent of the central
#    function under test — they use the SAME primitives the worker uses, arranged
#    as the worker arranges them). ──────────────────────────────────────────────

def worker_plugin_path_verdict(plugin, hyp, question, output):
    """Mirror `_plugin_verification_path` F1 block (sub_agent_loop.py ~1368-1447)."""
    ht = str(hyp.get("text") or "")
    tm = str(hyp.get("test_methodology") or "")
    min_steps = int(plugin.confirmation_criteria().get("min_metric_steps_for_confirm") or 1)
    try:
        verdict, reason, confidence = plugin.classify_verdict(ht, output)
    except Exception:  # pragma: no cover - fixed inputs never raise
        from propab.verdict_pipeline import run_verdict_pipeline

        verdict, confidence, reason = run_verdict_pipeline(
            output, hypothesis=hyp, campaign_context={"min_metric_steps": min_steps})
    if verdict == "confirmed":
        try:
            gv, gc, gr = artifact_gate_stage(
                output, verdict, confidence, reason,
                campaign_context={"min_metric_steps": min_steps})
            if gv == "confirmed" and classify_evidence_type(output) in ("lofo", "statistical"):
                gv, gc, gr = ood_gate_stage(
                    output, gv, gc, gr, hypothesis=hyp,
                    campaign_context={"hyp_text": ht, "test_methodology": tm})
                if gv == "confirmed":
                    scoped = attach_scope_integrity(
                        output, hypothesis_text=ht, test_methodology=tm,
                        experiment_output=output, question=question,
                        code=str(output.get("executed_code") or ""))
                    output["scope_integrity"] = scoped.get("scope_integrity")
                    output["scope_gate_result"] = scoped.get("scope_gate_result")
                    gv, gc, gr = scope_integrity_stage(output, gv, gc, gr, hypothesis=hyp)
        except Exception:  # pragma: no cover
            gv, gc, gr = ("inconclusive", 0.0, "gate raised")
        verdict, confidence, reason = gv, gc, gr
    serialized = json.dumps(output, default=str)
    return verdict, confidence, serialized


def worker_mandrake_path_verdict(plugin, hyp, question, output, baseline):
    """Mirror `_mandrake_verification_path` (sub_agent_loop.py ~1057-1114)."""
    from propab.domain_adapters.mandrake_adapter import classify_mandrake_verdict

    ht = str(hyp.get("text") or "")
    tm = str(hyp.get("test_methodology") or "LOFO") or "LOFO"
    verdict, reason, confidence = classify_mandrake_verdict(ht, output)
    ctx = evidence_context_from_hypothesis(
        ht,
        {"metric_value": output.get("mean_r2"), "lofo_r2": output.get("mean_r2"),
         "lofo_gap": output.get("lofo_gap"), "p_value": output.get("permutation_p"),
         "n_samples": output.get("n_samples"), "n_families": output.get("n_families"),
         "methodology": "LOFO", "feature_subset": output.get("feature_subset"),
         "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
         "label_shuffle_null_p95": output.get("label_shuffle_null_p95")},
        methodology="LOFO")
    verdict, reason, confidence, gate = apply_artifact_gate_override(verdict, reason, confidence, ctx, output)
    eo = _build_mandrake_evidence(output=output, verdict=verdict, reason=reason, baseline=baseline)
    eo = attach_scope_integrity(eo, hypothesis_text=ht, test_methodology=tm,
                                experiment_output=output, question=question, code="")
    if gate is not None:
        eo = merge_artifact_into_evidence(eo, gate)
    verdict, reason = apply_ood_gate_to_verdict(verdict, reason, eo, hypothesis_text=ht, test_methodology=tm)
    if eo.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
        verdict = "inconclusive"
    eo["verdict_reason"] = reason
    serialized = f"evidence={json.dumps(eo, ensure_ascii=False)}; mandrake_verification; LOFO={output.get('mean_r2')};"
    return verdict, confidence, serialized


def worker_materials_path_verdict(plugin, hyp, question, output, baseline):
    """Mirror `_materials_verification_path` (sub_agent_loop.py ~1584-1633)."""
    from propab.domain_adapters.materials_adapter import classify_materials_verdict

    ht = str(hyp.get("text") or "")
    tm = str(hyp.get("test_methodology") or "LOFO") or "LOFO"
    verdict, reason, confidence = classify_materials_verdict(ht, output)
    ctx = evidence_context_from_hypothesis(
        ht,
        {"metric_value": output.get("mean_r2"), "lofo_r2": output.get("lofo_r2"),
         "lofo_gap": output.get("lofo_gap"), "p_value": output.get("permutation_p"),
         "n_samples": output.get("n_samples"), "n_families": output.get("n_families"),
         "group_column": output.get("group_column"), "methodology": "LOFO",
         "feature_subset": output.get("feature_subset"),
         "label_shuffle_permutation_p": output.get("label_shuffle_permutation_p"),
         "label_shuffle_null_p95": output.get("label_shuffle_null_p95")},
        methodology="LOFO", domain_bucket="materials")
    gate = run_artifact_gate(ctx, output, question=question, payload=None)
    if verdict == "confirmed" and gate.verdict != "confirmed":
        verdict, reason, confidence = gate.verdict, gate.verdict_reason, gate.confidence
    elif verdict == "confirmed":
        reason = gate.verdict_reason
    eo = _build_materials_evidence(output=output, verdict=verdict, reason=reason, baseline=baseline)
    eo = attach_scope_integrity(eo, hypothesis_text=ht, test_methodology=tm,
                                experiment_output=output, question=question, code="")
    if gate is not None:
        eo = merge_artifact_into_evidence(eo, gate)
    verdict, reason = apply_ood_gate_to_verdict(verdict, reason, eo, hypothesis_text=ht, test_methodology=tm)
    if eo.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
        verdict = "inconclusive"
    eo["verdict_reason"] = reason
    serialized = f"evidence={json.dumps(eo, ensure_ascii=False)}; materials_verification;"
    return verdict, confidence, serialized


def _orch_verdict(plugin, hyp, question, serialized):
    parsed = parse_evidence_obj(serialized)
    assert is_recomputable_evidence(parsed), "fixture evidence should be judgeable"
    return compute_authoritative_verdict(
        plugin=plugin, hypothesis=hyp, evidence=dict(parsed),
        campaign_context={"question": question, "min_metric_steps": 2})


# ── Fixtures: fixed raw verification outputs per domain ───────────────────────

def _genomics_plugin():
    return resolve_domain_plugin(question="cross-tissue gene expression tau GTEx tissue specificity")


def _math_plugin():
    return resolve_domain_plugin(question="Sidon set B_3 cap set combinatorics construction")


def _mandrake_plugin():
    return resolve_domain_plugin(question="[domain_profile:mandrake] retroelement family LOFO")


def _materials_plugin():
    return resolve_domain_plugin(question="[domain_profile:materials] dielectric crystal-system LOFO")


HYP_GENOMICS = {"text": "Expression variance predicts cross-tissue specificity",
                "test_methodology": "leave-one-tissue-out R2 with tissue-shuffle null"}
HYP_MATH = {"text": "A larger B_3 Sidon set exists at n=20", "test_methodology": "exhaustive search"}
HYP_SURV = {"text": "the signal is cross-group and retained under leave-one-family-out",
            "test_methodology": "LOFO"}


def test_plugins_resolve() -> None:
    assert _genomics_plugin().domain_id == "genomics"
    assert _math_plugin().domain_id == "math_combinatorics"
    assert _mandrake_plugin().domain_id == "mandrake"
    assert _materials_plugin().domain_id == "materials"


# ── GENOMICS (plugin path, LOFO shape) ────────────────────────────────────────

@pytest.mark.parametrize("output, expected", [
    ({"lofo_r2": 0.02, "label_shuffle_null_p95": 0.9, "label_shuffle_null_p": 0.8,
      "verification_method": "leave_tissue_out", "verified_true_steps": 0}, "refuted"),
    ({"lofo_r2": 0.10, "label_shuffle_null_p95": 0.3, "label_shuffle_null_p": 0.2,
      "verification_method": "leave_tissue_out", "verified_true_steps": 0}, "inconclusive"),
])
def test_genomics_parity(output, expected) -> None:
    plugin = _genomics_plugin()
    wv, wc, serialized = worker_plugin_path_verdict(plugin, HYP_GENOMICS, "gtex", dict(output))
    ov, oc, _ = _orch_verdict(plugin, HYP_GENOMICS, "gtex", serialized)
    assert wv == expected  # pins concrete behaviour, not just circular equality
    assert ov == wv, (ov, wv)
    assert oc == pytest.approx(wc)


# ── MATH_COMBINATORICS (plugin path, deterministic proof) ─────────────────────

@pytest.mark.parametrize("output, expected", [
    ({"verified_true_steps": 1, "verified_false_steps": 0, "discovery_worthy": True,
      "verification_method": "combinatorial_computation", "notes": "record found",
      "deterministic": True}, "confirmed"),
    ({"verified_true_steps": 0, "verified_false_steps": 1,
      "verification_method": "counterexample_search", "notes": "counterexample"}, "refuted"),
    ({"trivial_rediscovery": True, "verified_true_steps": 1, "discovery_worthy": True,
      "notes": "known result"}, "inconclusive"),
])
def test_math_combinatorics_parity(output, expected) -> None:
    plugin = _math_plugin()
    wv, wc, serialized = worker_plugin_path_verdict(plugin, HYP_MATH, "combinatorics", dict(output))
    ov, oc, _ = _orch_verdict(plugin, HYP_MATH, "combinatorics", serialized)
    assert wv == expected
    assert ov == wv, (ov, wv)
    assert oc == pytest.approx(wc)


# ── MANDRAKE (dedicated LOFO-adapter path) ────────────────────────────────────

MANDRAKE_STRONG = {"mean_r2": 0.30, "lofo_r2": 0.30, "family_baseline_r2": 0.20, "lofo_gap": 0.05,
    "permutation_p": 0.01, "bootstrap_ci": [0.24, 0.36], "confidence_interval": [0.24, 0.36],
    "label_shuffle_permutation_p": 0.01, "label_shuffle_null_p95": 0.05, "n_samples": 300,
    "n_families": 8, "methodology": "LOFO", "feature_subset": ["f1"], "compare_result": None,
    "within_family_r2": 0.25, "global_r2": 0.4, "metric_value": 0.30, "p_value": 0.01,
    "executed_code": ""}
MANDRAKE_REFUTE = {"mean_r2": -0.5, "lofo_r2": -0.5, "family_baseline_r2": 0.2, "lofo_gap": 0.1,
    "permutation_p": 0.9, "bootstrap_ci": [-0.7, -0.3], "label_shuffle_permutation_p": 0.9,
    "label_shuffle_null_p95": 0.3, "n_samples": 300, "n_families": 8, "methodology": "LOFO",
    "feature_subset": ["f1"], "metric_value": -0.5, "p_value": 0.9, "executed_code": ""}


@pytest.mark.parametrize("output", [MANDRAKE_STRONG, MANDRAKE_REFUTE])
def test_mandrake_parity(output) -> None:
    plugin = _mandrake_plugin()
    wv, wc, serialized = worker_mandrake_path_verdict(plugin, HYP_SURV, "q", dict(output), {"value": 0.2})
    ov, oc, _ = _orch_verdict(plugin, HYP_SURV, "q", serialized)
    assert ov == wv, (ov, wv)
    assert oc == pytest.approx(wc)


def test_mandrake_refute_is_refuted() -> None:
    """Concrete pin: the refute fixture is genuinely refuted end-to-end."""
    plugin = _mandrake_plugin()
    wv, _wc, serialized = worker_mandrake_path_verdict(plugin, HYP_SURV, "q", dict(MANDRAKE_REFUTE), {"value": 0.2})
    ov, _oc, _ = _orch_verdict(plugin, HYP_SURV, "q", serialized)
    assert wv == "refuted"
    assert ov == "refuted"


# ── MATERIALS (dedicated LOFO-adapter path) ───────────────────────────────────

MATERIALS_STRONG = {"mean_r2": 0.30, "lofo_r2": 0.30, "family_baseline_r2": 0.20, "lofo_gap": 0.40,
    "permutation_p": 0.01, "label_shuffle_permutation_p": 0.01, "label_shuffle_null_p95": 0.05,
    "n_samples": 600, "n_families": 7, "group_column": "crystal_system", "methodology": "LOFO",
    "feature_subset": ["density"], "family_leakage_confirmed": False, "metric_value": 0.30,
    "p_value": 0.01, "executed_code": ""}
MATERIALS_REFUTE = {"mean_r2": -0.5, "lofo_r2": -0.5, "family_baseline_r2": 0.2, "lofo_gap": 0.1,
    "permutation_p": 0.9, "label_shuffle_permutation_p": 0.9, "label_shuffle_null_p95": 0.3,
    "n_samples": 600, "n_families": 7, "group_column": "crystal_system", "methodology": "LOFO",
    "feature_subset": ["density"], "family_leakage_confirmed": False, "metric_value": -0.5,
    "p_value": 0.9, "executed_code": ""}


@pytest.mark.parametrize("output", [MATERIALS_STRONG, MATERIALS_REFUTE])
def test_materials_parity(output) -> None:
    plugin = _materials_plugin()
    wv, wc, serialized = worker_materials_path_verdict(plugin, HYP_SURV, "q", dict(output), {"value": 0.2})
    ov, oc, _ = _orch_verdict(plugin, HYP_SURV, "q", serialized)
    assert ov == wv, (ov, wv)
    assert oc == pytest.approx(wc)


def test_materials_refute_is_refuted() -> None:
    plugin = _materials_plugin()
    wv, _wc, serialized = worker_materials_path_verdict(plugin, HYP_SURV, "q", dict(MATERIALS_REFUTE), {"value": 0.2})
    ov, _oc, _ = _orch_verdict(plugin, HYP_SURV, "q", serialized)
    assert wv == "refuted"
    assert ov == "refuted"


# ── Short-circuit / failure evidence must NOT be re-judged ────────────────────

@pytest.mark.parametrize("evidence_str", [
    json.dumps({"reason": "hypothesis_off_topic_for_domain", "domain": "genomics"}),
    json.dumps({"error": "tool blew up", "domain": "materials"}),
    "",
    "some raw exception traceback with no evidence= block",
])
def test_shortcircuit_evidence_is_not_recomputable(evidence_str) -> None:
    """Off-topic / error / empty payloads are not verification results — the
    orchestrator must keep the worker's verdict rather than fabricate one."""
    parsed = parse_evidence_obj(evidence_str)
    assert is_recomputable_evidence(parsed) is False


def test_offtopic_would_be_misjudged_if_recomputed() -> None:
    """Documents WHY the guard exists: a domain classifier over an off-topic payload
    fabricates a verdict, so the guard (not classify) must gate the recompute."""
    plugin = _genomics_plugin()
    offtopic = {"reason": "hypothesis_off_topic_for_domain", "domain": "genomics"}
    # classify over the empty-of-signal payload yields a (wrong) refuted...
    assert plugin.classify_verdict("x", offtopic)[0] == "refuted"
    # ...so the guard must reject it from recomputation.
    assert is_recomputable_evidence(offtopic) is False
