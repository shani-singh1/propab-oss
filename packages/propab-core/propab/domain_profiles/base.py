"""Domain profile base — evidence structure handlers for artifact gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from propab.artifact_verification import (
    ArtifactGateResult,
    ArtifactModel,
    ArtifactVerification,
    EvidenceContext,
    apply_two_stage_gate,
    rank_artifact_models,
    run_adversarial_test,
)


@dataclass(frozen=True)
class DomainProfile:
    """
    Tells Propab's verification layer what counts as real evidence in a domain.

    Answers (fixes.md Step 3):
    1. Natural grouping variable for LOFO / holdout
    2. Statistical test matching evidence structure
    3. Permutation null definition
    """

    profile_id: str
    display_name: str
    group_column: str
    group_label: str
    evidence_method: str
    permutation_null: str
    min_samples_per_group: int = 8
    min_groups: int = 3
    min_metric_steps_for_confirm: int = 2
    question_markers: tuple[str, ...] = ()
    artifact_model_builder: Callable[[EvidenceContext], list[ArtifactModel]] | None = None

    def matches_question(self, question: str) -> bool:
        q = (question or "").lower()
        return any(m in q for m in self.question_markers)

    def enrich_context(self, ctx: EvidenceContext) -> EvidenceContext:
        if ctx.group_column is None:
            return EvidenceContext(
                hypothesis_text=ctx.hypothesis_text,
                evidence_generation_method=ctx.evidence_generation_method or self.evidence_method,
                n_samples=ctx.n_samples,
                n_groups=ctx.n_groups,
                group_column=self.group_column,
                feature_count=ctx.feature_count,
                domain_bucket=self.profile_id,
                p_value=ctx.p_value,
                metric_value=ctx.metric_value,
                effect_size=ctx.effect_size,
                lofo_r2=ctx.lofo_r2,
                lofo_gap=ctx.lofo_gap,
                tools_used=list(ctx.tools_used),
                claim_type=ctx.claim_type,
                cited_prior_node_ids=list(ctx.cited_prior_node_ids),
            )
        return ctx

    def generate_artifact_models(self, ctx: EvidenceContext) -> list[ArtifactModel]:
        if self.artifact_model_builder is not None:
            return self.artifact_model_builder(ctx)
        from propab.artifact_verification import generate_artifact_models

        return generate_artifact_models(ctx)

    def run_artifact_gate(
        self,
        ctx: EvidenceContext,
        experiment: dict[str, Any] | None = None,
        *,
        top_k: int = 3,
        tree_nodes: dict[str, Any] | None = None,
    ) -> ArtifactGateResult:
        ctx = self.enrich_context(ctx)
        models = self.generate_artifact_models(ctx)
        ranked = rank_artifact_models(models, top_k=top_k)
        verifications = [run_adversarial_test(a, ctx, experiment) for a in ranked]
        return apply_two_stage_gate(ranked, verifications, ctx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "group_column": self.group_column,
            "group_label": self.group_label,
            "evidence_method": self.evidence_method,
            "permutation_null": self.permutation_null,
            "min_samples_per_group": self.min_samples_per_group,
            "min_groups": self.min_groups,
            "min_metric_steps_for_confirm": self.min_metric_steps_for_confirm,
        }
