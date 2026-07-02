"""Seed hypothesis sources for campaign tree planting."""
from __future__ import annotations

from enum import Enum


class SeedSource(str, Enum):
    """Where initial / regenerated seeds originate."""

    DEFAULT = "default"
    ANOMALY = "anomaly"
