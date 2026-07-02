"""Graph invariants DomainPlugin — configures the artifact gate via GRAPH_INVARIANTS_PROFILE."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.base import DomainPlugin


class GraphInvariantsPlugin(DomainPlugin):
    domain_id = "graph_invariants"
    display_name = "Graph invariants (SNAP network families)"
    version = "0.1"

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        return self.domain_profile().matches_question(question or "")

    def available_features(self) -> list[str]:
        return []

    def domain_profile(self):
        from propab.domain_profiles.graph_invariants import GRAPH_INVARIANTS_PROFILE

        return GRAPH_INVARIANTS_PROFILE
