"""Domain profile for additive combinatorics (deterministic verification)."""
from __future__ import annotations

from propab.domain_profiles.base import DomainProfile

MATH_COMBINATORICS_PROFILE = DomainProfile(
    profile_id="math_combinatorics",
    display_name="Additive Combinatorics",
    group_column="search_strategy",
    group_label="search strategy",
    evidence_method="combinatorial_computation",
    permutation_null="exhaustive_counterexample_search",
    min_samples_per_group=1,
    min_groups=1,
    min_metric_steps_for_confirm=1,
    question_markers=(
        "domain_profile:math_combinatorics",
        "sidon",
        "cap set",
        "sumset",
        "additive combinator",
        "arithmetic progression",
        "ap-free",
        "extremal",
    ),
)
