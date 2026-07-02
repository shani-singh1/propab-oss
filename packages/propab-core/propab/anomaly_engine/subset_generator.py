"""Feature-subset generation for sweeps (domain-agnostic)."""
from __future__ import annotations

import itertools
from typing import Iterable


def generate_feature_subsets(
    feature_columns: Iterable[str],
    *,
    max_subset_size: int,
) -> list[list[str]]:
    """Exhaustive singles/pairs/… up to max_subset_size within one column list."""
    cols = list(feature_columns)
    subsets: list[list[str]] = []
    for size in range(1, min(max_subset_size, len(cols)) + 1):
        for combo in itertools.combinations(cols, size):
            subsets.append(list(combo))
    return subsets


def generate_grouped_subsets(
    feature_groups: dict[str, list[str]],
    *,
    max_subset_size: int = 3,
    available_columns: set[str] | None = None,
) -> list[list[str]]:
    """
    Singles, pairs, triples within each named group (not cross-group exhaustive).
    Use when the feature space is large and grouped search is intended.
    """
    subsets: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cols in feature_groups.values():
        usable = [c for c in cols if available_columns is None or c in available_columns]
        for subset in generate_feature_subsets(usable, max_subset_size=max_subset_size):
            key = tuple(sorted(subset))
            if key not in seen:
                seen.add(key)
                subsets.append(subset)
    return subsets
