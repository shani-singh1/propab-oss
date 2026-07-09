"""Composed verdict pipeline — pure, testable, no I/O."""
from __future__ import annotations

from typing import Any

from propab.artifact_verification import (
    apply_artifact_gate_override,
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
        "exhaustive_enumeration",
    }
    method = evidence.get("verification_method")
    # V3: a "deterministic" (gate-bypassing) classification requires an EXPLICIT
    # proof method. The former "any non-significance method name + a verified_true
    # counter" clause was a gate-bypass hole — a domain emitting e.g.
    # verification_method="cross_network_lofo" with verified_true_steps=1 earned a
    # free "confirmed" with NO adversarial null.
    #
    # V4: a bare ``deterministic: True`` flag is NO LONGER a standalone gate
    # bypass. Previously ``deterministic is True`` alone forced this branch, so any
    # future/refactored plugin that stamped ``deterministic: True`` on a
    # thresholded/statistical result earned an automatic artifact-gate bypass with
    # no shape cross-check. The flag is now honored ONLY when it co-occurs with a
    # recognized proof ``verification_method`` — and a proof method already
    # suffices on its own, so the condition reduces to ``method in _PROOF_METHODS``.
    # A genuine statistical/lofo result that carries real null statistics is already
    # classified as "lofo" above (has_lofo) or "statistical" below (has_stats);
    # anything left with only a ``deterministic`` flag / step counter and an
    # unrecognized (or absent) method must go THROUGH the artifact gate, not around
    # it, so it falls to "unknown" and is gated. Domain-general: keyed on evidence
    # shape, never on domain id. (math_combinatorics/coding_theory still classify as
    # deterministic — they set verification_method to "combinatorial_computation" /
    # "exhaustive_enumeration", both recognized proof methods.)
    is_deterministic = method in _PROOF_METHODS
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


def compute_confidence(
    evidence: dict[str, Any],
    verdict: str,
) -> float:
    """Canonical confidence score for a hypothesis verdict.

    Single source of truth shared by the core verdict pipeline and the worker
    (``services.worker.sub_agent_loop._compute_confidence`` is a thin adapter that
    maps its ``HypothesisEvidence`` TypedDict onto this). Domain-general: keyed only
    on evidence shape, never on a domain id.
    """
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


# Backwards-compatible alias — this was the private name before the C1
# consolidation collapsed the worker's duplicate into this single implementation.
_compute_pipeline_confidence = compute_confidence


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
    confidence = compute_confidence(evidence, verdict)
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

    # A deterministic proof (symbolic_proof / exact_check / exhaustive_enumeration
    # / counterexample_search) is NOT a distributional claim: an OOD/transfer gate
    # has no meaning for it and wrongly collapses a genuine proof to "inconclusive"
    # the moment a hypothesis text is present (parse_scope finds no scope, so
    # check_ood_evidence returns "no OOD evidence" and downgrades). The OOD gate must
    # apply ONLY to distributional (lofo/statistical) evidence — this mirrors the
    # plugin verdict path's F1 block, which already restricts OOD to
    # classify_evidence_type in ("lofo", "statistical"). Keyed on evidence SHAPE,
    # never on domain id; the deterministic (math/coding_theory) confirm path stays
    # confirmable through the full pipeline regardless of whether a hyp text is set.
    if classify_evidence_type(evidence) == "deterministic":
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
    # Same shape contract as ood_gate_stage: a scope-integrity FAIL (declared vs
    # executed OOD mismatch) is meaningless for a deterministic proof, which has no
    # OOD/transfer surface. Never let a pre-attached scope FAIL collapse a proof —
    # the deterministic confirm path must survive the full pipeline. Distributional
    # (lofo/statistical) evidence is still fully scope-gated below.
    if classify_evidence_type(evidence) == "deterministic":
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


# ── Central verdict authority (C2) ───────────────────────────────────────────
# The orchestrator — not the worker — owns the verdict. `compute_authoritative_
# verdict` is the single entry the orchestrator invokes on the worker's RAW
# evidence, reproducing the exact verdict each worker path used to compute
# locally (behaviour-preserving relocation). Domains keep their `classify_verdict`
# (domain-independence rule §3.5); core only *invokes* an already-resolved plugin —
# it never inspects the question for domain keywords or imports a domain module.


def plugin_overrides_classify_verdict(plugin: Any) -> bool:
    """True iff a resolved DomainPlugin ships its own ``classify_verdict``.

    Duck-typed: compares the bound method's function against the base default so a
    plugin that only implements ``run_verification`` (base ``classify_verdict``
    returns neutral/inconclusive) routes through the generic ``run_verdict_pipeline``
    instead. Never imports a specific domain module.
    """
    if plugin is None:
        return False
    try:
        from propab.domain_modules.base import DomainPlugin

        return type(plugin).classify_verdict is not DomainPlugin.classify_verdict
    except Exception:  # noqa: BLE001 — a resolution error must never mislabel routing
        return False


def _dedicated_lofo_verdict(
    plugin: Any,
    hypothesis: dict[str, Any] | None,
    evidence: dict[str, Any],
    question: str,
    tool: str,
) -> tuple[str, float, str]:
    """Reproduce the worker's mandrake/materials dedicated LOFO-adapter gate chain.

    Keyed on the evidence's own ``verification_tool`` provenance (which verifier
    produced it), never on a domain id — this mirrors the two legacy dedicated
    worker paths (`_mandrake_verification_path` / `_materials_verification_path`)
    exactly, including their (mutually different) confidence handling, so the
    relocation changes no verdict AND no confidence. The mandrake path takes the
    artifact-gate OVERRIDE (adopts the gate's confidence); the materials path keeps
    the classifier confidence on gate survival. These per-path quirks are legacy —
    they collapse into the single gate chain in C3 — but C2 must preserve them.
    """
    ht = str((hypothesis or {}).get("text") or "")
    tm = str((hypothesis or {}).get("test_methodology") or "") or str(evidence.get("methodology") or "") or "LOFO"
    verdict, reason, confidence = plugin.classify_verdict(ht, evidence)
    ctx = evidence_context_from_hypothesis(
        ht,
        {
            "metric_value": evidence.get("mean_r2", evidence.get("lofo_r2")),
            "lofo_r2": evidence.get("lofo_r2", evidence.get("mean_r2")),
            "lofo_gap": evidence.get("lofo_gap"),
            "p_value": evidence.get("permutation_p", evidence.get("p_value")),
            "n_samples": evidence.get("n_samples"),
            "n_families": evidence.get("n_families"),
            "group_column": evidence.get("group_column"),
            "methodology": "LOFO",
            "feature_subset": evidence.get("feature_subset"),
            "label_shuffle_permutation_p": evidence.get("label_shuffle_permutation_p"),
            "label_shuffle_null_p95": evidence.get("label_shuffle_null_p95"),
        },
        methodology="LOFO",
        domain_bucket="materials" if tool == "materials_verification" else None,
    )
    gate = None
    if tool == "materials_verification":
        gate = run_artifact_gate(ctx, evidence, question=question, payload=None)
        if verdict == "confirmed" and gate.verdict != "confirmed":
            verdict, reason, confidence = gate.verdict, gate.verdict_reason, gate.confidence
        elif verdict == "confirmed":
            reason = gate.verdict_reason
    else:  # mandrake: artifact-gate override (adopts gate confidence on confirm)
        verdict, reason, confidence, gate = apply_artifact_gate_override(
            verdict, reason, confidence, ctx, evidence,
        )
    if gate is not None:
        evidence = merge_artifact_into_evidence(evidence, gate)
    verdict, reason = apply_ood_gate_to_verdict(
        verdict, reason, evidence, hypothesis_text=ht, test_methodology=tm,
    )
    if evidence.get("scope_gate_result") == "FAIL" and verdict == "confirmed":
        verdict = "inconclusive"
        reason = f"scope integrity fail: {evidence.get('scope_integrity', {}).get('reason', '?')}"
    return verdict, confidence, reason


def _plugin_chain_verdict(
    plugin: Any,
    hypothesis: dict[str, Any] | None,
    evidence: dict[str, Any],
    min_steps: int,
) -> tuple[str, float, str]:
    """Reproduce the worker generic plugin path: plugin ``classify_verdict`` then the
    shape-aware artifact -> OOD -> scope gate chain (F1 block in
    ``_plugin_verification_path``). Fail-closed on any gate error."""
    ht = str((hypothesis or {}).get("text") or "")
    tm = str((hypothesis or {}).get("test_methodology") or "")
    verdict, reason, confidence = plugin.classify_verdict(ht, evidence)
    if verdict == "confirmed":
        try:
            gv, gc, gr = artifact_gate_stage(
                evidence, verdict, confidence, reason,
                campaign_context={"min_metric_steps": min_steps},
            )
            if gv == "confirmed" and classify_evidence_type(evidence) in ("lofo", "statistical"):
                gv, gc, gr = ood_gate_stage(
                    evidence, gv, gc, gr, hypothesis=hypothesis,
                    campaign_context={"hyp_text": ht, "test_methodology": tm},
                )
                if gv == "confirmed":
                    gv, gc, gr = scope_integrity_stage(
                        evidence, gv, gc, gr, hypothesis=hypothesis,
                    )
        except Exception:  # noqa: BLE001 — FAIL CLOSED: never leave a confirm on gate error
            gv, gc, gr = (
                "inconclusive",
                0.0,
                "artifact/scope gate raised; failing closed (verdict downgraded to inconclusive)",
            )
        verdict, confidence, reason = gv, gc, gr
    return verdict, confidence, reason


def is_recomputable_evidence(evidence: dict[str, Any]) -> bool:
    """True when raw evidence carries a real verification result to judge.

    Guards the orchestrator against re-deriving a verdict from a short-circuit /
    failure payload (off-topic ``{"reason": ...}``, tool error ``{"error": ...}``,
    empty exception blobs) — those are NOT verification evidence and running a
    domain classifier over them would fabricate a verdict. Keyed on evidence
    shape, never on a domain id.
    """
    if not isinstance(evidence, dict) or not evidence:
        return False
    if evidence.get("error") is not None:
        return False
    if classify_evidence_type(evidence) != "unknown":
        return True
    from propab.research_quality import is_valid_evidence_for_hash

    return is_valid_evidence_for_hash(evidence)


def compute_authoritative_verdict(
    *,
    plugin: Any,
    hypothesis: dict[str, Any] | None,
    evidence: dict[str, Any],
    campaign_context: dict[str, Any] | None = None,
) -> tuple[str, float, str]:
    """Single central verdict authority the ORCHESTRATOR invokes on raw evidence.

    Reproduces the verdict path each worker used so the C2 relocation is
    behaviour-preserving (verified parity across genomics, math_combinatorics,
    mandrake and materials):

    * ``verification_tool`` == mandrake/materials -> the dedicated adapter classifier
      + adapter gate sequence (:func:`_dedicated_lofo_verdict`);
    * any other domain whose plugin overrides ``classify_verdict`` -> classify + the
      shape-aware artifact/OOD/scope gate chain (:func:`_plugin_chain_verdict`);
    * no plugin / no override -> the generic :func:`run_verdict_pipeline`.

    ``evidence`` may be mutated (artifact/OOD/scope annotations) — pass a copy if the
    caller needs the original. Returns ``(verdict, confidence, reason)``.
    """
    cc = campaign_context or {}
    question = str(cc.get("question") or "")
    min_steps = int(cc.get("min_metric_steps") or 2)
    tool = str(evidence.get("verification_tool") or "")

    if tool in ("mandrake_verification", "materials_verification") and plugin_overrides_classify_verdict(plugin):
        return _dedicated_lofo_verdict(plugin, hypothesis, evidence, question, tool)

    if plugin_overrides_classify_verdict(plugin):
        try:
            return _plugin_chain_verdict(plugin, hypothesis, evidence, min_steps)
        except Exception:  # noqa: BLE001 — a broken plugin classifier must not crash the loop
            return run_verdict_pipeline(evidence, hypothesis=hypothesis, campaign_context=cc)

    return run_verdict_pipeline(evidence, hypothesis=hypothesis, campaign_context=cc)
