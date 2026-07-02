"""Enzyme kinetics / biophysical feature LOFO profile (Candidate A)."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_OVERFITTING,
    ARTIFACT_SAMPLE_SIZE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    TEST_HELD_OUT_GROUP,
    TEST_LABEL_SHUFFLE_LOFO,
    TEST_PERMUTATION_NULL,
    ArtifactModel,
    EvidenceContext,
    _base_artifact,
)
from propab.domain_profiles.base import DomainProfile


def _enzyme_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    models: list[ArtifactModel] = []
    n = ctx.n_samples or 0
    n_feat = ctx.feature_count or 0
    n_groups = ctx.n_groups or 0

    models.append(_base_artifact(
        ARTIFACT_FAMILY_LEAKAGE,
        "Signal tracks enzyme family / EC class rather than cross-family biophysics",
        0.90,
        "LOFO holdout on enzyme family is the primary generalization test",
        ["family_column", "features", "target"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    if n > 0 and n < 120:
        models.append(_base_artifact(
            ARTIFACT_SAMPLE_SIZE,
            "Small cohort produces unstable kinetic estimates",
            0.70 if n < 60 else 0.50,
            f"n={n} enzyme entries may not support family-crossing inference",
            ["sample_size"],
            TEST_PERMUTATION_NULL,
        ))

    if n_feat >= 3 and n > 0 and n_feat * 4 > n:
        models.append(_base_artifact(
            ARTIFACT_OVERFITTING,
            "Too many structural features relative to enzyme count",
            0.72,
            f"{n_feat} features on n={n}",
            ["feature_subset"],
            TEST_HELD_OUT_GROUP,
        ))

    if ctx.p_value is not None and ctx.lofo_r2 is None and n_groups < 2:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Within-family significance without cross-family LOFO",
            0.82,
            "p-value without family holdout is insufficient for cross-enzyme claims",
            ["p_value"],
            TEST_LABEL_SHUFFLE_LOFO,
        ))

    return models


ENZYME_KINETICS_PROFILE = DomainProfile(
    profile_id="enzyme_kinetics",
    display_name="Enzyme kinetics (BRENDA / UniProt families)",
    group_column="enzyme_family",
    group_label="enzyme family or EC class",
    evidence_method="LOFO",
    permutation_null="shuffle family labels; require LOFO R² above label-shuffle p95",
    min_samples_per_group=12,
    min_groups=4,
    min_metric_steps_for_confirm=2,
    question_markers=(
        "enzyme",
        "kcat",
        "km",
        "catalytic",
        "brenda",
        "uniprot",
        "ec class",
        "thermal stability",
        "enzyme kinetics",
        "domain_profile:enzyme_kinetics",
    ),
    artifact_model_builder=_enzyme_artifact_models,
)
