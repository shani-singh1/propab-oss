"""Enzyme kinetics DomainPlugin — configures the artifact gate via ENZYME_KINETICS_PROFILE.

No dedicated verification adapter yet (run_verification is unsupported); this
plugin makes the enzyme-kinetics artifact profile addressable through the same
interface as fully-implemented domains.
"""
from __future__ import annotations

from typing import Any

from propab.domain_modules.base import DomainPlugin


class EnzymeKineticsPlugin(DomainPlugin):
    domain_id = "enzyme_kinetics"
    display_name = "Enzyme kinetics (BRENDA / UniProt families)"
    version = "0.1"

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        return self.domain_profile().matches_question(question or "")

    def available_features(self) -> list[str]:
        return []

    def domain_profile(self):
        from propab.domain_profiles.enzyme_kinetics import ENZYME_KINETICS_PROFILE

        return ENZYME_KINETICS_PROFILE
