"""Domain adapters — dataset-specific verification (fixes.md P0)."""
from propab.domain_adapters.mandrake_adapter import (
    MandrakeAdapter,
    MandrakeExperimentSpec,
    classify_mandrake_verdict,
    infer_biology_theme,
    is_mandrake_campaign,
    resolve_mandrake_features,
)

__all__ = [
    "MandrakeAdapter",
    "MandrakeExperimentSpec",
    "classify_mandrake_verdict",
    "infer_biology_theme",
    "is_mandrake_campaign",
    "resolve_mandrake_features",
]
