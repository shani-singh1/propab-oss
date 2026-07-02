"""
Artifact-aware verification (fixes.md P0–P8).

Confirmation requires surviving adversarial tests against plausible artifact models,
not merely passing a significance gate.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

# ── P0: ArtifactModel ────────────────────────────────────────────────────────

ARTIFACT_FAMILY_LEAKAGE = "family_leakage"
ARTIFACT_TOPOLOGY_DEPENDENCE = "topology_dependence"
ARTIFACT_SAMPLE_SIZE = "sample_size_artifact"
ARTIFACT_OVERFITTING = "overfitting"
ARTIFACT_SIMULATOR = "simulator_specific"
ARTIFACT_MEASUREMENT = "measurement_bias"
ARTIFACT_FEATURE_REDUNDANCY = "feature_redundancy"
ARTIFACT_DISTRIBUTION_SHIFT = "distribution_shift"
ARTIFACT_SIGNIFICANCE_ONLY = "significance_only"

TEST_LABEL_SHUFFLE_LOFO = "label_shuffle_lofo"
TEST_PERMUTATION_NULL = "permutation_null"
TEST_BOOTSTRAP_STABILITY = "bootstrap_stability"
TEST_HELD_OUT_GROUP = "held_out_group"
TEST_ALTERNATE_SIMULATOR = "alternate_simulator"
TEST_ROBUSTNESS = "robustness_analysis"

_NETWORK_MARKERS = (
    "network", "graph", "contagion", "sis", "sir", "topology", "modular",
    "scale-free", "barab", "erdős", "erdos", "k-shell", "k-core", "diffusion",
)
_SIM_MARKERS = ("simulation", "simulator", "monte carlo", "synthetic", "sandbox")
_GROUP_MARKERS = ("family", "group", "lofo", "logo", "leave-one", "cross-group", "topology")


@dataclass
class ArtifactModel:
    artifact_id: str
    description: str
    plausibility_score: float
    why_plausible: str
    affected_components: list[str]
    proposed_test: str
    artifact_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactVerification:
    artifact_model: ArtifactModel
    test_used: str
    survived: bool
    effect_size: float | None
    confidence: float
    observed_stat: float | None = None
    null_p95: float | None = None
    empirical_p: float | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifact_model"] = self.artifact_model.to_dict()
        return d


@dataclass
class EvidenceContext:
    hypothesis_text: str
    evidence_generation_method: str = "unknown"
    n_samples: int | None = None
    n_groups: int | None = None
    group_column: str | None = None
    feature_count: int | None = None
    domain_bucket: str | None = None
    p_value: float | None = None
    metric_value: float | None = None
    effect_size: float | None = None
    lofo_r2: float | None = None
    lofo_gap: float | None = None
    tools_used: list[str] = field(default_factory=list)
    claim_type: str | None = None
    cited_prior_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactGateResult:
    verdict: str
    verdict_reason: str
    confidence: float
    ranked_artifacts: list[ArtifactModel]
    verifications: list[ArtifactVerification]
    top_artifact_survived: bool
    second_artifact_trivial: bool
    artifact_survival_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "confidence": self.confidence,
            "ranked_artifacts": [a.to_dict() for a in self.ranked_artifacts],
            "verifications": [v.to_dict() for v in self.verifications],
            "top_artifact_survived": self.top_artifact_survived,
            "second_artifact_trivial": self.second_artifact_trivial,
            "artifact_survival_rate": self.artifact_survival_rate,
        }


def _base_artifact(
    artifact_id: str,
    description: str,
    plausibility: float,
    why: str,
    components: list[str],
    test: str,
) -> ArtifactModel:
    return ArtifactModel(
        artifact_id=artifact_id,
        description=description,
        plausibility_score=plausibility,
        why_plausible=why,
        affected_components=components,
        proposed_test=test,
    )


def generate_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    """P1 — assume claim is false; list plausible alternative explanations."""
    text = (ctx.hypothesis_text or "").lower()
    method = (ctx.evidence_generation_method or "").lower()
    models: list[ArtifactModel] = []

    has_groups = (ctx.n_groups or 0) >= 2 or ctx.group_column is not None
    is_network = any(m in text for m in _NETWORK_MARKERS) or ctx.domain_bucket == "graphs"
    is_lofo = "lofo" in method or "leave-one" in method or "logo" in method
    is_sim = any(m in text for m in _SIM_MARKERS) or any("code" in t for t in ctx.tools_used)
    n = ctx.n_samples or 0
    n_feat = ctx.feature_count or 0

    if has_groups or is_lofo or any(m in text for m in _GROUP_MARKERS):
        models.append(_base_artifact(
            ARTIFACT_FAMILY_LEAKAGE,
            "Signal tracks group identity rather than cross-group structure",
            0.92 if is_lofo else 0.78,
            "Grouped data with within-group fit often masquerades as discovery",
            ["group_labels", "features", "target"],
            TEST_LABEL_SHUFFLE_LOFO,
        ))

    if is_network:
        models.append(_base_artifact(
            ARTIFACT_TOPOLOGY_DEPENDENCE,
            "Effect is specific to one network topology family",
            0.88,
            "Graph claims often fail to generalize across ER/BA/WS/SBM ensembles",
            ["topology_family", "structural_features", "outcome"],
            TEST_LABEL_SHUFFLE_LOFO,
        ))

    if n > 0 and n < 80:
        models.append(_base_artifact(
            ARTIFACT_SAMPLE_SIZE,
            "Small sample size produces unstable significance",
            0.75 if n < 40 else 0.55,
            f"Only n={n} observations — p-values may not replicate",
            ["sample_size", "p_value"],
            TEST_PERMUTATION_NULL,
        ))

    if n_feat >= 3 and n > 0 and n_feat * 5 > n:
        models.append(_base_artifact(
            ARTIFACT_OVERFITTING,
            "Too many features relative to sample size",
            0.70,
            f"{n_feat} features on n={n} invites spurious fit",
            ["feature_subset", "model"],
            TEST_HELD_OUT_GROUP,
        ))

    if is_sim:
        models.append(_base_artifact(
            ARTIFACT_SIMULATOR,
            "Result is an artifact of one simulator implementation",
            0.65,
            "Single sandbox code path was used to generate evidence",
            ["simulator", "parameters"],
            TEST_ALTERNATE_SIMULATOR,
        ))

    if ctx.p_value is not None and ctx.lofo_r2 is None and not is_lofo:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Significance without cross-group or holdout validation",
            0.85,
            "p-value passed but no adversarial holdout was run",
            ["p_value", "metric_value"],
            TEST_PERMUTATION_NULL,
        ))

    if ctx.effect_size is not None and abs(ctx.effect_size) > 3.0 and (ctx.lofo_r2 or 0) < 0:
        models.append(_base_artifact(
            ARTIFACT_MEASUREMENT,
            "Large effect size with poor generalization suggests measurement bias",
            0.60,
            "Effect size and LOFO disagree — metric may be miscalibrated",
            ["effect_size", "metric"],
            TEST_ROBUSTNESS,
        ))

    if n_feat >= 4:
        models.append(_base_artifact(
            ARTIFACT_FEATURE_REDUNDANCY,
            "Redundant collinear features inflate apparent signal",
            0.50,
            "Multiple correlated structural features submitted together",
            ["feature_subset"],
            TEST_ROBUSTNESS,
        ))

    if not models:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Insufficient metadata to rule out significance-only confirmation",
            0.70,
            "No group structure or holdout metadata in evidence",
            ["evidence"],
            TEST_PERMUTATION_NULL,
        ))

    return models


_PREVALENCE_BOOST: dict[str, float] = {
    ARTIFACT_FAMILY_LEAKAGE: 0.05,
    ARTIFACT_TOPOLOGY_DEPENDENCE: 0.04,
    ARTIFACT_SIGNIFICANCE_ONLY: 0.03,
    ARTIFACT_SAMPLE_SIZE: 0.02,
}


def rank_artifact_models(models: list[ArtifactModel], *, top_k: int = 3) -> list[ArtifactModel]:
    """P2 — sort by plausibility × prevalence × simplicity; keep top K."""

    def score(m: ArtifactModel) -> float:
        simplicity = 1.0 if m.proposed_test in (TEST_LABEL_SHUFFLE_LOFO, TEST_PERMUTATION_NULL) else 0.85
        return m.plausibility_score * simplicity + _PREVALENCE_BOOST.get(m.artifact_id, 0.0)

    ranked = sorted(models, key=score, reverse=True)[:top_k]
    out: list[ArtifactModel] = []
    for i, m in enumerate(ranked, start=1):
        out.append(ArtifactModel(
            artifact_id=m.artifact_id,
            description=m.description,
            plausibility_score=m.plausibility_score,
            why_plausible=m.why_plausible,
            affected_components=list(m.affected_components),
            proposed_test=m.proposed_test,
            artifact_rank=i,
        ))
    return out


def _survives_label_shuffle_lofo(exp: dict[str, Any]) -> ArtifactVerification:
    lofo = exp.get("lofo_r2", exp.get("mean_r2"))
    p95 = exp.get("label_shuffle_null_p95")
    p_ge = exp.get("label_shuffle_permutation_p")
    gap = exp.get("lofo_gap")
    y_perm_p = exp.get("permutation_p")

    if lofo is not None:
        lofo = float(lofo)
    if p95 is not None:
        p95 = float(p95)
    if p_ge is not None:
        p_ge = float(p_ge)
    if y_perm_p is not None:
        y_perm_p = float(y_perm_p)

    survived = False
    rationale = ""
    if lofo is not None and p95 is not None and p_ge is not None:
        survived = lofo > p95 and p_ge < 0.05
        rationale = f"LOFO={lofo:.3f} vs label-shuffle null p95={p95:.3f}, p={p_ge:.3f}"
    elif lofo is not None and gap is not None:
        survived = lofo > 0.05 and float(gap) < 0.85
        rationale = f"LOFO={lofo:.3f} gap={float(gap):.3f} (heuristic; no label-shuffle null)"
    elif lofo is not None and y_perm_p is not None:
        survived = lofo > -0.05 and y_perm_p < 0.05
        rationale = f"LOFO={lofo:.3f} y-perm p={y_perm_p:.3f} (fallback)"

    artifact = _base_artifact(
        ARTIFACT_FAMILY_LEAKAGE, "Group/family leakage", 0.9, "", [], TEST_LABEL_SHUFFLE_LOFO,
    )
    return ArtifactVerification(
        artifact_model=artifact,
        test_used=TEST_LABEL_SHUFFLE_LOFO,
        survived=survived,
        effect_size=float(gap) if gap is not None else lofo,
        confidence=0.85 if survived else 0.65,
        observed_stat=lofo,
        null_p95=p95,
        empirical_p=p_ge or y_perm_p,
        rationale=rationale or "insufficient LOFO null statistics",
    )


def _survives_permutation(ctx: EvidenceContext, exp: dict[str, Any]) -> ArtifactVerification:
    p = ctx.p_value or exp.get("p_value")
    n = ctx.n_samples or exp.get("n_samples") or 0
    lofo = ctx.lofo_r2 or exp.get("lofo_r2") or exp.get("mean_r2")
    survived = False
    rationale = ""
    if lofo is not None:
        survived = float(lofo) > 0.0
        rationale = f"LOFO={float(lofo):.3f} positive under permutation context"
    elif p is not None and n >= 80:
        survived = float(p) < 0.01 and n >= 100
        rationale = f"n={n} large enough for p={float(p):.4f} to be meaningful"
    else:
        rationale = f"n={n} too small or no holdout — significance-only path fails audit"

    artifact = _base_artifact(
        ARTIFACT_SIGNIFICANCE_ONLY, "Significance-only confirmation", 0.85, "", [], TEST_PERMUTATION_NULL,
    )
    return ArtifactVerification(
        artifact_model=artifact,
        test_used=TEST_PERMUTATION_NULL,
        survived=survived,
        effect_size=ctx.effect_size,
        confidence=0.7 if survived else 0.55,
        observed_stat=float(lofo) if lofo is not None else None,
        empirical_p=float(p) if p is not None else None,
        rationale=rationale,
    )


def run_adversarial_test(
    artifact: ArtifactModel,
    ctx: EvidenceContext,
    experiment: dict[str, Any] | None = None,
) -> ArtifactVerification:
    """P3 — cheapest test that would destroy the claim if artifact were responsible."""
    exp = experiment or {}
    test = artifact.proposed_test

    if test == TEST_LABEL_SHUFFLE_LOFO:
        v = _survives_label_shuffle_lofo(exp)
        v.artifact_model = artifact
        return v
    if test in (TEST_PERMUTATION_NULL, TEST_BOOTSTRAP_STABILITY):
        v = _survives_permutation(ctx, exp)
        v.artifact_model = artifact
        return v
    if test == TEST_HELD_OUT_GROUP:
        lofo = exp.get("lofo_r2", exp.get("mean_r2"))
        gap = exp.get("lofo_gap")
        survived = lofo is not None and float(lofo) > -0.15 and (gap is None or float(gap) < 1.0)
        return ArtifactVerification(
            artifact_model=artifact,
            test_used=TEST_HELD_OUT_GROUP,
            survived=survived,
            effect_size=float(gap) if gap is not None else None,
            confidence=0.75 if survived else 0.6,
            observed_stat=float(lofo) if lofo is not None else None,
            rationale=f"held-out group LOFO={lofo} gap={gap}",
        )
    if test == TEST_ALTERNATE_SIMULATOR:
        return ArtifactVerification(
            artifact_model=artifact,
            test_used=TEST_ALTERNATE_SIMULATOR,
            survived=False,
            effect_size=None,
            confidence=0.4,
            rationale="alternate simulator not run — cannot confirm",
        )
    lofo = exp.get("lofo_r2", exp.get("mean_r2"))
    survived = lofo is not None and float(lofo) > 0.0
    return ArtifactVerification(
        artifact_model=artifact,
        test_used=TEST_ROBUSTNESS,
        survived=survived,
        effect_size=ctx.effect_size,
        confidence=0.6,
        observed_stat=float(lofo) if lofo is not None else None,
        rationale="robustness check on generalization metric",
    )


def _second_artifact_trivial(
    second: ArtifactModel,
    first_verification: ArtifactVerification,
    ctx: EvidenceContext,
) -> bool:
    if not first_verification.survived:
        return False
    if second.artifact_id == ARTIFACT_SIGNIFICANCE_ONLY:
        return (
            ctx.p_value is not None
            and ctx.lofo_r2 is None
            and ctx.n_samples is not None
            and ctx.n_samples < 100
        )
    if second.artifact_id == ARTIFACT_SAMPLE_SIZE:
        return (ctx.n_samples or 0) < 60
    if second.artifact_id == ARTIFACT_TOPOLOGY_DEPENDENCE:
        return (
            first_verification.observed_stat is not None
            and first_verification.observed_stat < 0.05
        )
    if second.artifact_id == ARTIFACT_FAMILY_LEAKAGE:
        gap = ctx.lofo_gap
        return gap is not None and float(gap) >= 0.35 and (ctx.lofo_r2 or 0) < 0
    return second.plausibility_score >= 0.75


def apply_two_stage_gate(
    ranked: list[ArtifactModel],
    verifications: list[ArtifactVerification],
    ctx: EvidenceContext,
) -> ArtifactGateResult:
    """P5/P6 — two-stage artifact gate."""
    if not ranked or not verifications:
        return ArtifactGateResult(
            verdict="inconclusive",
            verdict_reason="no artifact models ranked",
            confidence=0.4,
            ranked_artifacts=ranked,
            verifications=verifications,
            top_artifact_survived=False,
            second_artifact_trivial=False,
        )

    top_v = verifications[0]
    top_survived = top_v.survived
    second_trivial = False
    if len(ranked) >= 2 and top_survived:
        second_trivial = _second_artifact_trivial(ranked[1], top_v, ctx)

    if not top_survived:
        verdict = "refuted"
        reason = f"artifact '{ranked[0].artifact_id}' explains result: {top_v.rationale}"
        confidence = top_v.confidence
    elif second_trivial:
        verdict = "inconclusive"
        reason = (
            f"survived top artifact ({ranked[0].artifact_id}) but "
            f"second-ranked '{ranked[1].artifact_id}' still plausibly explains result"
        )
        confidence = 0.55
    else:
        verdict = "confirmed"
        reason = (
            f"survived adversarial test for '{ranked[0].artifact_id}' "
            f"({top_v.test_used}); second artifact not trivial"
        )
        confidence = min(0.95, top_v.confidence + 0.05)

    return ArtifactGateResult(
        verdict=verdict,
        verdict_reason=reason,
        confidence=confidence,
        ranked_artifacts=ranked,
        verifications=verifications,
        top_artifact_survived=top_survived,
        second_artifact_trivial=second_trivial,
        artifact_survival_rate=1.0 if top_survived else 0.0,
    )


def run_artifact_gate(
    ctx: EvidenceContext,
    experiment: dict[str, Any] | None = None,
    *,
    top_k: int = 3,
    tree_nodes: dict[str, Any] | None = None,
    question: str = "",
    payload: dict[str, Any] | None = None,
) -> ArtifactGateResult:
    from propab.domain_profiles import resolve_domain_profile

    profile = resolve_domain_profile(ctx, question=question, payload=payload)
    if profile is not None:
        return profile.run_artifact_gate(ctx, experiment, top_k=top_k, tree_nodes=tree_nodes)

    binding_rejected = 0
    if tree_nodes and ctx.cited_prior_node_ids:
        from propab.evidence_binding import BindingMetrics, filter_node_citations

        bm = BindingMetrics()
        ctx.cited_prior_node_ids = filter_node_citations(
            ctx.hypothesis_text,
            list(ctx.cited_prior_node_ids),
            tree_nodes,
            metrics=bm,
        )
        binding_rejected = bm.binding_rejected_count

    models = generate_artifact_models(ctx)
    ranked = rank_artifact_models(models, top_k=top_k)
    verifications = [run_adversarial_test(a, ctx, experiment) for a in ranked]
    result = apply_two_stage_gate(ranked, verifications, ctx)
    if binding_rejected:
        result.verdict_reason = (
            f"{result.verdict_reason}; binding_rejected_prior={binding_rejected}"
        ).strip("; ")
    return result


def evidence_context_from_hypothesis(
    hypothesis_text: str,
    evidence: dict[str, Any] | None = None,
    *,
    tools_used: list[str] | None = None,
    methodology: str = "",
    domain_bucket: str | None = None,
) -> EvidenceContext:
    ev = evidence or {}
    features = ev.get("feature_subset") or ev.get("features") or []
    if isinstance(features, str):
        features = [features]
    lofo = _float_or_none(ev.get("lofo_r2") or ev.get("mean_r2"))
    metric = _float_or_none(ev.get("metric_value"))
    if lofo is None and ev.get("methodology") == "LOFO":
        lofo = metric
    return EvidenceContext(
        hypothesis_text=hypothesis_text,
        evidence_generation_method=methodology or str(ev.get("methodology") or "unknown"),
        n_samples=_int_or_none(ev.get("n_samples")),
        n_groups=_int_or_none(ev.get("n_families") or ev.get("n_groups")),
        group_column=ev.get("family_column") or ev.get("group_column"),
        feature_count=len(features) if features else _int_or_none(ev.get("n_features")),
        domain_bucket=domain_bucket,
        p_value=_float_or_none(ev.get("p_value")),
        metric_value=metric,
        effect_size=_float_or_none(ev.get("effect_size") or ev.get("lofo_gap")),
        lofo_r2=lofo,
        lofo_gap=_float_or_none(ev.get("lofo_gap")),
        tools_used=list(tools_used or []),
        claim_type=ev.get("claim_type"),
    )


def merge_artifact_into_evidence(
    evidence_obj: dict[str, Any],
    gate: ArtifactGateResult,
) -> dict[str, Any]:
    out = dict(evidence_obj)
    out["artifact_gate"] = gate.to_dict()
    out["artifact_survival_rate"] = gate.artifact_survival_rate
    out["top_artifact"] = gate.ranked_artifacts[0].artifact_id if gate.ranked_artifacts else None
    out["top_artifact_survived"] = gate.top_artifact_survived
    out["second_artifact_check"] = gate.second_artifact_trivial
    out["verdict_reason"] = gate.verdict_reason
    return out


def aggregate_artifact_metrics(gates: list[ArtifactGateResult]) -> dict[str, Any]:
    if not gates:
        return {
            "artifact_survival_rate": 0.0,
            "artifact_failure_distribution": {},
            "n_audited": 0,
            "n_confirmed_under_artifact_gate": 0,
            "n_refuted_by_artifact": 0,
            "n_inconclusive_artifact": 0,
        }
    n = len(gates)
    survival = sum(1 for g in gates if g.top_artifact_survived) / n
    failures: dict[str, int] = {}
    for g in gates:
        if not g.top_artifact_survived and g.ranked_artifacts:
            aid = g.ranked_artifacts[0].artifact_id
            failures[aid] = failures.get(aid, 0) + 1
    return {
        "artifact_survival_rate": round(survival, 4),
        "artifact_failure_distribution": failures,
        "n_audited": n,
        "n_confirmed_under_artifact_gate": sum(1 for g in gates if g.verdict == "confirmed"),
        "n_refuted_by_artifact": sum(1 for g in gates if g.verdict == "refuted"),
        "n_inconclusive_artifact": sum(1 for g in gates if g.verdict == "inconclusive"),
    }


def format_paper_artifact_block(claim: str, gate: ArtifactGateResult) -> str:
    if not gate.ranked_artifacts or not gate.verifications:
        return f"Claim: {claim[:80]}. No artifact audit available."
    top_a = gate.ranked_artifacts[0]
    top_v = gate.verifications[0]
    second = gate.ranked_artifacts[1].artifact_id if len(gate.ranked_artifacts) > 1 else "n/a"
    survived = "Yes" if top_v.survived else "No"
    second_note = "trivially explains" if gate.second_artifact_trivial else "does not trivially explain"
    return (
        f"Claim: {claim[:200]}\n"
        f"Top artifact: {top_a.description}\n"
        f"Adversarial test: {top_v.test_used}\n"
        f"Survived: {survived} ({top_v.rationale})\n"
        f"Second artifact ({second}): {second_note}.\n"
        f"Verdict: {gate.verdict}."
    )


def format_paper_artifact_section(findings: list[tuple[str, ArtifactGateResult]]) -> str:
    parts = ["## Artifact-Adversarial Verification\n"]
    for claim, gate in findings:
        parts.append(format_paper_artifact_block(claim, gate))
        parts.append("")
    return "\n".join(parts)


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_evidence_summary(summary: str) -> dict[str, Any]:
    if not summary:
        return {}
    m = re.search(r"evidence=(\{.*?\});", summary)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def audit_confirmed_row(row: dict[str, Any], experiment: dict[str, Any] | None = None) -> ArtifactGateResult:
    text = str(row.get("text") or row.get("key_finding") or "")
    ev = parse_evidence_summary(str(row.get("evidence_summary") or ""))
    if experiment:
        ev = {**ev, **{k: v for k, v in experiment.items() if v is not None}}
    domain = "graphs" if any(m in text.lower() for m in _NETWORK_MARKERS) else None
    ctx = evidence_context_from_hypothesis(text, ev, domain_bucket=domain)
    return run_artifact_gate(ctx, experiment)


def audit_confirmed_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gates = [audit_confirmed_row(r) for r in rows]
    metrics = aggregate_artifact_metrics(gates)
    return {
        **metrics,
        "findings": [
            {
                "hypothesis_id": rows[i].get("id"),
                "text_snippet": str(rows[i].get("text") or "")[:160],
                "gate": gates[i].to_dict(),
            }
            for i in range(len(rows))
        ],
        "recommendation": (
            f"{metrics['n_confirmed_under_artifact_gate']}/{metrics['n_audited']} survive artifact gate. "
            "Optimize for cross-domain artifact survival, not raw confirmation count."
        ),
    }


def apply_artifact_gate_override(
    preliminary_verdict: str,
    preliminary_reason: str,
    preliminary_confidence: float,
    ctx: EvidenceContext,
    experiment: dict[str, Any] | None = None,
) -> tuple[str, str, float, ArtifactGateResult | None]:
    """If preliminary verdict is confirmed, require artifact gate survival (P5)."""
    if preliminary_verdict != "confirmed":
        return preliminary_verdict, preliminary_reason, preliminary_confidence, None
    gate = run_artifact_gate(ctx, experiment)
    if gate.verdict == "confirmed":
        return gate.verdict, gate.verdict_reason, gate.confidence, gate
    return gate.verdict, gate.verdict_reason, gate.confidence, gate
