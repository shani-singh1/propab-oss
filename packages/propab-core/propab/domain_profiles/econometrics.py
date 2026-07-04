"""Panel econometrics profile for AstaBench DiscoveryBench FE / OLS evidence."""
from __future__ import annotations

from propab.artifact_verification import (
    ARTIFACT_FAMILY_LEAKAGE,
    ARTIFACT_OVERFITTING,
    ARTIFACT_SIGNIFICANCE_ONLY,
    TEST_PANEL_WITHIN_FE,
    TEST_PERMUTATION_NULL,
    ArtifactModel,
    EvidenceContext,
    _base_artifact,
)
from propab.domain_profiles.base import DomainProfile


def _econometrics_artifact_models(ctx: EvidenceContext) -> list[ArtifactModel]:
    models: list[ArtifactModel] = []
    n = ctx.n_samples or 0
    n_feat = ctx.feature_count or 0

    models.append(_base_artifact(
        ARTIFACT_FAMILY_LEAKAGE,
        "Panel signal tracks entity or time fixed-effect identity rather than within-panel variation",
        0.88,
        "Within-group R² must exceed baseline under entity/time FE holdout",
        ["entity_id", "time_period", "features"],
        TEST_PANEL_WITHIN_FE,
    ))

    if n > 0 and n_feat >= 4 and n_feat * 3 > n:
        models.append(_base_artifact(
            ARTIFACT_OVERFITTING,
            "High-dimensional regressors on short panel",
            0.65,
            f"{n_feat} regressors, n={n} panel rows",
            ["feature_subset"],
            TEST_PERMUTATION_NULL,
        ))

    if ctx.p_value is not None and ctx.lofo_r2 is None:
        models.append(_base_artifact(
            ARTIFACT_SIGNIFICANCE_ONLY,
            "Pooled OLS significance without within-group FE comparison",
            0.80,
            "Econometrics claims require within-group R² vs baseline",
            ["p_value"],
            TEST_PANEL_WITHIN_FE,
        ))

    return models


ECONOMETRICS_PROFILE = DomainProfile(
    profile_id="econometrics",
    display_name="Econometrics (panel FE / within-group R²)",
    group_column="entity_id",
    group_label="panel entity or time fixed effect",
    evidence_method="within_group_FE",
    permutation_null="standard permutation p<0.05 on within-group residual; not label-shuffle p95",
    min_samples_per_group=20,
    min_groups=5,
    min_metric_steps_for_confirm=2,
    question_markers=(
        "econometrics",
        "panel data",
        "fixed effect",
        "fixed effects",
        "within-group",
        "within group",
        "ols coefficient",
        "regression coefficient",
        "discoverybench",
        "astabench",
        "domain_profile:econometrics",
    ),
    artifact_model_builder=_econometrics_artifact_models,
)
