"""Real-network graph invariant profile (Candidate C)."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    ARTIFACT_TOPOLOGY_DEPENDENCE,
    TEST_LABEL_SHUFFLE_LOFO,
    TEST_PERMUTATION_NULL,
    TEST_ROBUSTNESS,
    ArtifactModel,
    EvidenceContext,
    _base_artifact,
)
from propab.domain_profiles.base import DomainProfile


def _graph_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    models: list[ArtifactModel] = []

    models.append(_base_artifact(
        ARTIFACT_TOPOLOGY_DEPENDENCE,
        "Invariant holds on one network family but not others",
        0.90,
        "Graph claims must generalize across SNAP network categories",
        ["network_type", "graph_invariant"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    models.append(_base_artifact(
        ARTIFACT_FAMILY_LEAKAGE,
        "Measured property tracks network category metadata rather than structure",
        0.85,
        "Hold out entire network type when testing cross-network invariants",
        ["network_category", "features"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    if ctx.p_value is not None and ctx.lofo_r2 is None:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Empirical correlation without exact combinatorial verification",
            0.75,
            "Prefer exact checks on graph properties where possible",
            ["p_value"],
            TEST_PERMUTATION_NULL,
        ))
    else:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Empirical fit without held-out network family",
            0.70,
            "Cross-network replication required",
            ["metric_value"],
            TEST_ROBUSTNESS,
        ))

    return models


# EvidenceContext has no verified-step counter; deterministic graph checks flow via evidence dict.


GRAPH_INVARIANTS_PROFILE = DomainProfile(
    profile_id="graph_invariants",
    display_name="Graph invariants (SNAP network families)",
    group_column="network_category",
    group_label="SNAP network category or graph family",
    evidence_method="cross_network_lofo",
    permutation_null="hold out network category; shuffle category labels for null",
    min_samples_per_group=30,
    min_groups=4,
    min_metric_steps_for_confirm=1,
    question_markers=(
        "graph invariant",
        "spectral",
        "network repository",
        "snap.stanford",
        "clustering coefficient",
        "graph family",
        "combinatorial",
        "domain_profile:graph_invariants",
    ),
    artifact_model_builder=_graph_artifact_models,
)
