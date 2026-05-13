"""Normalize reported metrics so accuracy-like values stay on a 0–1 fraction scale."""

from __future__ import annotations


def normalize_accuracy_metric(metric_name: str | None, value: float | None) -> float | None:
    """
    If the metric name suggests accuracy and the value looks like a percentage (>1),
    rescale by /100 so downstream breakthrough / best-metric logic uses fractions.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if "accuracy" not in (metric_name or "").lower():
        return v
    if v > 1.0:
        v = v / 100.0
    return v
