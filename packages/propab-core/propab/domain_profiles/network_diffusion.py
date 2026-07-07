"""Network-diffusion profile — real SNAP networks, simulated diffusion outcomes."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_SIGNIFICANCE_ONLY,
    ARTIFACT_SIMULATOR,
    ARTIFACT_TOPOLOGY_DEPENDENCE,
    TEST_ALTERNATE_SIMULATOR,
    TEST_LABEL_SHUFFLE_LOFO,
    TEST_PERMUTATION_NULL,
    ArtifactModel,
    EvidenceContext,
    _base_artifact,
)
from propab.domain_profiles.base import DomainProfile


def _diffusion_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    models: list[ArtifactModel] = []

    models.append(_base_artifact(
        ARTIFACT_TOPOLOGY_DEPENDENCE,
        "Diffusion law holds on one real network family but not the held-out family",
        0.90,
        "Require the structure->outcome correlation to replicate on a held-out real network",
        ["network_family", "structural_feature"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    models.append(_base_artifact(
        ARTIFACT_SIMULATOR,
        "Effect is an artifact of the specific diffusion dynamics (SIR vs cascade)",
        0.85,
        "Replicate the held-out correlation under an alternate simulator",
        ["simulator", "outcome"],
        TEST_ALTERNATE_SIMULATOR,
    ))

    models.append(_base_artifact(
        ARTIFACT_FAMILY_LEAKAGE,
        "Outcome tracks network-level scale (density/size) rather than topology",
        0.80,
        "Within-family shuffle null; hold out whole network family",
        ["network_family", "mean_degree"],
        TEST_LABEL_SHUFFLE_LOFO,
    ))

    if ctx.p_value is not None and ctx.lofo_r2 is None:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Correlation significance without cross-family replication or null",
            0.75,
            "Cross-topology-family holdout + within-family shuffle null required",
            ["p_value"],
            TEST_PERMUTATION_NULL,
        ))

    return models


NETWORK_DIFFUSION_PROFILE = DomainProfile(
    profile_id="network_diffusion",
    display_name="Network diffusion / contagion (real SNAP topology families)",
    group_column="network_family",
    group_label="real network topology family (collaboration / email)",
    evidence_method="cross_topology_family_lofo",
    permutation_null="hold out one real network family; shuffle outcomes within family for null",
    min_samples_per_group=25,
    min_groups=2,
    min_metric_steps_for_confirm=1,
    question_markers=(
        "contagion",
        "diffusion",
        "epidemic spreading",
        "sir model",
        "sis model",
        "independent cascade",
        "outbreak",
        "epidemic threshold",
        "domain_profile:network_diffusion",
    ),
    artifact_model_builder=_diffusion_artifact_models,
)
