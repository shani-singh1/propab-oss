"""Composed verdict pipeline — pure, testable, no I/O."""
from __future__ import annotations

from typing import Any

from propab.artifact_verification import (
    evidence_context_from_hypothesis,
    merge_artifact_into_evidence,
    run_artifact_gate,
)
from propab.scoped_claim import apply_ood_gate_to_verdict, check_ood_evidence, parse_scope_from_methodology
from propab.significance import SignificanceResult, check_significance, classify_verdict


def classify_evidence_type(evidence: dict[str, Any]) -> str:
    """
    "deterministic" — verified proof / exact check
    "lofo"          — lofo_r2 + label_shuffle_null_p95 present
    "statistical"   — p_value + metric_value, no lofo
    "unknown"       — none of the above
    """
    vt = int(evidence.get("verified_true_steps") or 0)
    vf = int(evidence.get("verified_false_steps") or 0)

    has_lofo = (
        evidence.get("lofo_r2") is not None
        and evidence.get("label_shuffle_null_p95") is not None
    )
    if has_lofo:
        return "lofo"

    # V2: deterministic gate-bypass requires an actual proof method. A bare
    # `verified_true_steps` counter is NOT sufficient — a counter without an
    # adversarial control must be routed through the artifact gate like any
    # other result, so it cannot silently bypass artifact verification.
    _PROOF_METHODS = {
        "symbolic_proof",
        "exact_check",
        "counterexample_search",
        "combinatorial_verification",
        "combinatorial_computation",
        "counterexample",
    }
    method = evidence.get("verification_method")
    # V3: a "deterministic" (gate-bypassing) classification requires an EXPLICIT
    # proof method or an explicit `deterministic` flag. The former "any
    # non-significance method name + a verified_true counter" clause was a
    # gate-bypass hole — a domain emitting e.g. verification_method=
    # "cross_network_lofo" with verified_true_steps=1 earned a free "confirmed"
    # with NO adversarial null. A genuine statistical/lofo result that carries real
    # null statistics is already classified as "lofo" above (has_lofo) or
    # "statistical" below (has_stats); anything left with only a step counter and
    # an unrecognized method must go THROUGH the artifact gate, not around it, so
    # it falls to "unknown" and is gated. Domain-general: keyed on evidence shape,
    # never on domain id.
    is_deterministic = (
        evidence.get("deterministic") is True
        or method in _PROOF_METHODS
    )
    if is_deterministic:
        return "deterministic"

    has_stats = (
        evidence.get("p_value") is not None
        and evidence.get("metric_value") is not None
    )
    if has_stats:
        return "statistical"

    return "unknown"


def _sig_result_from_context(
    evidence: dict[str, Any],
    campaign_context: dict[str, Any] | None,
) -> SignificanceResult:
    ctx = campaign_context or {}
    sig = ctx.get("sig_result")
    if isinstance(sig, SignificanceResult):
        return sig
    return check_significance([evidence])


def _compute_pipeline_confidence(
    evidence: dict[str, Any],
    verdict: str,
) -> float:
    if int(evidence.get("verified_true_steps") or 0) > 0 and verdict == "confirmed":
        return 0.95
    if int(evidence.get("verified_false_steps") or 0) > 0 and verdict == "refuted":
        return 0.95

    score = 0.0
    if evidence.get("metric_value") is not None:
        score += 0.20
    if evidence.get("baseline_value") is not None:
        score += 0.20
    p = evidence.get("p_value")
    if p is not None and float(p) < 0.05:
        score += 0.25
    es = evidence.get("effect_size")
    if es is not None and abs(float(es)) > 0.2:
        score += 0.15
    if int(evidence.get("n_metric_steps") or 0) >= 3:
        score += 0.10
    if float(evidence.get("relevance_score") or 0.0) > 0.30:
        score += 0.10
    return min(max(score, 0.0), 0.95)


def classify_verdict_stage(
    evidence: dict[str, Any],
    campaign_context: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    ctx = campaign_context or {}
    min_metric_steps = int(ctx.get("min_metric_steps") or 2)
    sig = _sig_result_from_context(evidence, ctx)
    verdict, reason = classify_verdict(
        evidence,
        sig,
        min_metric_steps_for_confirm=min_metric_steps,
    )
    confidence = _compute_pipeline_confidence(evidence, verdict)
    return verdict, confidence, reason


def run_full_artifact_gate(
    evidence: dict[str, Any],
    *,
    campaign_context: dict[str, Any] | None = None,
) -> Any:
    ctx_data = campaign_context or {}
    hyp_text = str(ctx_data.get("hyp_text") or "")
    tools = list(ctx_data.get("tools_used") or [])
    domain = str(ctx_data.get("domain_bucket") or "")
    question = str(ctx_data.get("question") or "")
    payload = ctx_data.get("payload") if isinstance(ctx_data.get("payload"), dict) else None

    gate_ctx = evidence_context_from_hypothesis(
        hyp_text,
        evidence,
        tools_used=tools,
        domain_bucket=domain or None,
    )
    return run_artifact_gate(
        gate_ctx,
        dict(evidence),
        question=question,
        payload=payload,
    )


def artifact_gate_stage(
    evidence: dict[str, Any],
    verdict: str,
    confidence: float,
    reason: str,
    *,
    campaign_context: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    if verdict != "confirmed":
        return verdict, confidence, reason

    evidence_type = classify_evidence_type(evidence)

    if evidence_type == "deterministic":
        return verdict, confidence, reason

    if evidence_type == "lofo":
        gate_result = run_full_artifact_gate(evidence, campaign_context=campaign_context)
        merge_artifact_into_evidence(evidence, gate_result)
        if gate_result.verdict != "confirmed":
            return gate_result.verdict, 0.0, gate_result.verdict_reason
        return verdict, confidence, gate_result.verdict_reason or reason

    if evidence_type == "statistical":
        # W1b: a statistical confirm requires BOTH a real passing null AND that
        # the numbers the significance/permutation test ran on were NOT typed
        # directly by the LLM agent. The worker records the provenance of those
        # inputs on the evidence dict as `stat_input_provenance`:
        #   "computed"      — produced inside the sandbox (trusted);
        #   "agent_literal" — the agent typed the numbers into the tool call
        #                     (UNTRUSTED / possibly fabricated);
        #   "unknown"       — could not be determined.
        # A permutation null computed on agent-fabricated arrays looks perfectly
        # "significant", so this is the only choke point that can stop a
        # fabricated statistical result from confirming.
        #
        # Fail closed ONLY on the KNOWN-untrusted "agent_literal" case. We do NOT
        # block "unknown" here: absence of provenance is not evidence of
        # fabrication, and many legitimate paths (older evidence, third-party
        # tools) leave it unset/unknown — blocking those would falsely downgrade
        # honest confirmations. If a future audit shows "unknown" is being abused,
        # tighten it here. This guard is scoped strictly to the statistical
        # branch; the deterministic-proof and lofo (label-shuffle) paths above
        # compute in-sandbox and are intentionally untouched.
        stat_provenance = evidence.get("stat_input_provenance")
        if stat_provenance == "agent_literal":
            return (
                "inconclusive",
                confidence * 0.5,
                "stat_inputs_agent_literal_untrusted: significance/permutation "
                "inputs were typed by the agent (not sandbox-computed); a "
                "statistical confirm requires non-fabricated inputs",
            )

        # V1: a purely statistical result is confirmable only when it carries a
        # real, independent adversarial null (an outcome permutation null or a
        # label-shuffle null) that PASSES. Route it through the artifact gate's
        # permutation-null path instead of auto-downgrading. If the required null
        # statistics are absent, the gate cannot mark it as surviving, so a bare
        # significance number can never confirm.
        #
        # Fail-closed policy: only a genuinely PASSING null yields "confirmed".
        # Any non-confirm gate outcome (including a hard "refuted" that the gate
        # emits merely because no null was supplied) is collapsed to
        # "inconclusive": absence of an adversarial control is not positive
        # evidence of an artifact, so we must not manufacture a "refuted".
        gate_result = run_full_artifact_gate(evidence, campaign_context=campaign_context)
        merge_artifact_into_evidence(evidence, gate_result)
        if gate_result.verdict == "confirmed":
            return verdict, confidence, gate_result.verdict_reason or reason
        return (
            "inconclusive",
            confidence * 0.7,
            "significance gate passed but no passing adversarial null / holdout "
            f"available to rule out artifact; treat as preliminary ({gate_result.verdict_reason})",
        )

    return "inconclusive", 0.0, f"unrecognized evidence type: {evidence_type}"


def ood_gate_stage(
    evidence: dict[str, Any],
    verdict: str,
    confidence: float,
    reason: str,
    *,
    hypothesis: dict[str, Any] | None = None,
    campaign_context: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    if verdict != "confirmed":
        return verdict, confidence, reason

    ctx = campaign_context or {}
    hyp_text = str(
        (hypothesis or {}).get("text")
        or ctx.get("hyp_text")
        or ""
    )
    test_methodology = str(
        (hypothesis or {}).get("test_methodology")
        or ctx.get("test_methodology")
        or ""
    )

    if not hyp_text and not test_methodology:
        if classify_evidence_type(evidence) == "lofo":
            ok, ood_reason = check_ood_evidence(evidence, None)
            evidence["ood_passed"] = ok
            evidence["ood_reason"] = ood_reason
            if not ok:
                return "inconclusive", confidence * 0.7, f"OOD gate: {ood_reason}"
        return verdict, confidence, reason

    scope = parse_scope_from_methodology(hyp_text, test_methodology)
    ok, ood_reason = check_ood_evidence(evidence, scope)
    evidence["ood_passed"] = ok
    evidence["ood_reason"] = ood_reason
    new_verdict, new_reason = apply_ood_gate_to_verdict(
        verdict,
        reason,
        evidence,
        hypothesis_text=hyp_text,
        test_methodology=test_methodology,
    )
    if new_verdict != verdict:
        return new_verdict, confidence * 0.7, new_reason
    return verdict, confidence, new_reason


def scope_integrity_stage(
    evidence: dict[str, Any],
    verdict: str,
    confidence: float,
    reason: str,
    *,
    hypothesis: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    if verdict != "confirmed":
        return verdict, confidence, reason
    if evidence.get("scope_gate_result") == "FAIL":
        scope_reason = evidence.get("scope_integrity", {}).get("reason", "?")
        return (
            "inconclusive",
            confidence * 0.7,
            f"scope integrity fail: {scope_reason}",
        )
    return verdict, confidence, reason


def run_verdict_pipeline(
    evidence: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
    campaign_context: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    """
    Pure composition of verdict stages (mutates evidence for artifact/OOD annotations).

    Returns: (verdict, confidence, reason)
    """
    verdict, confidence, reason = classify_verdict_stage(evidence, campaign_context)
    verdict, confidence, reason = artifact_gate_stage(
        evidence, verdict, confidence, reason, campaign_context=campaign_context,
    )
    verdict, confidence, reason = ood_gate_stage(
        evidence, verdict, confidence, reason,
        hypothesis=hypothesis,
        campaign_context=campaign_context,
    )
    verdict, confidence, reason = scope_integrity_stage(
        evidence, verdict, confidence, reason, hypothesis=hypothesis,
    )
    evidence["verdict_reason"] = reason
    return verdict, confidence, reason
