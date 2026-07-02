"""Materials science cross-family generalization profile (Candidate B)."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_FEATURE_REDUNDANCY,
    ARTIFACT_OVERFITTING,
    ARTIFACT_SIGNIFICANCE_ONLY,
    TEST_HELD_OUT_GROUP,
    TEST_LABEL_SHUFFLE_LOFO,
    TEST_PERMUTATION_NULL,
    ArtifactModel,
    EvidenceContext,
    _base_artifact,
)
from propab.domain_profiles.base import DomainProfile


def _materials_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    models: list[ArtifactModel] = []
    n = ctx.n_samples or 0
    n_feat = ctx.feature_count or 0

    models.append(_base_artifact(
        ARTIFACT_FAMILY_LEAKAGE,
        "Property prediction tracks crystal system / composition family identity",
        0.88,
        "Cross-family holdout on space group or composition family tests generalization",
        ["crystal_system", "composition_family", "features"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    if n_feat >= 4:
        models.append(_base_artifact(
            ARTIFACT_FEATURE_REDUNDANCY,
            "Collinear structural descriptors inflate band-gap / formation-energy signal",
            0.55,
            "Materials descriptors are often highly correlated",
            ["feature_subset"],
            TEST_HELD_OUT_GROUP,
        ))

    if n > 0 and n_feat * 3 > n:
        models.append(_base_artifact(
            ARTIFACT_OVERFITTING,
            "High-dimensional composition features on modest n",
            0.68,
            f"{n_feat} descriptors, n={n}",
            ["feature_subset"],
            TEST_HELD_OUT_GROUP,
        ))

    if ctx.p_value is not None and ctx.lofo_r2 is None:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "In-sample fit without cross-family holdout",
            0.80,
            "Materials claims require cross-system validation",
            ["p_value", "metric_value"],
            TEST_PERMUTATION_NULL,
        ))

    return models


MATERIALS_PROFILE = DomainProfile(
    profile_id="materials",
    display_name="Materials properties (crystal-system LOFO)",
    group_column="crystal_system",
    group_label="crystal system or composition family",
    evidence_method="LOFO",
    permutation_null="hold out entire crystal-system family; shuffle family labels for null",
    min_samples_per_group=25,
    min_groups=5,
    min_metric_steps_for_confirm=2,
    question_markers=(
        "materials project",
        "crystal",
        "formation energy",
        "band gap",
        "space group",
        "composition",
        "matbench",
        "domain_profile:materials",
    ),
    artifact_model_builder=_materials_artifact_models,
)
