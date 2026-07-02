"""Materials DomainPlugin — delegates to the (already tested) materials adapter/profile.

Adapter imports (which pull pymatgen) are lazy so registering this plugin stays
cheap and cannot break the registry if pymatgen is unavailable.
"""
from __future__ import annotations

from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult


class MaterialsPlugin(DomainPlugin):
    domain_id = "materials"
    display_name = "Materials properties (matbench dielectric, crystal-system LOFO)"
    version = "1.0"
    scope_question_markers = (
        "domain_profile:materials",
        "matbench",
        "dielectric",
        "crystal system",
        "crystal-system",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Matbench dielectric crystals (N≈4,764) with pymatgen structures",
            "distribution": (
                "Training on 6 of 7 crystal systems (cubic, tetragonal, orthorhombic, "
                "monoclinic, trigonal, hexagonal, triclinic)"
            ),
            "claimed_generalization": "Descriptor→dielectric relationship holds on held-out crystal system",
            "expected_failure_modes": (
                "Fails when descriptors proxy crystal-system identity; metallic/high-gap outliers"
            ),
            "ood_test": "Leave-one-crystal-system-out LOFO; lofo_r2 must beat label-shuffle null p95",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        # Mirrors the historical is_materials_campaign: explicit tag/payload only.
        q = (question or "").lower()
        if "domain_profile:materials" in q:
            return True
        if payload:
            if str(payload.get("domain_profile") or "") == "materials":
                return True
            if str(payload.get("domain") or "") == "materials":
                return True
        return False

    def available_features(self) -> list[str]:
        from propab.domain_adapters.materials_adapter import _KNOWN_FEATURES

        return list(_KNOWN_FEATURES)

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        from propab.domain_adapters.materials_adapter import (
            MaterialsAdapter,
            MaterialsExperimentSpec,
            resolve_materials_features,
        )

        feats = list(features or hypothesis.get("feature_subset") or [])
        if not feats:
            feats = resolve_materials_features(str(hypothesis.get("text", "")))
        spec = MaterialsExperimentSpec(feature_subset=feats, methodology="LOFO")
        return MaterialsAdapter().run_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        from propab.domain_adapters.materials_adapter import classify_materials_verdict

        return classify_materials_verdict(hypothesis_text, result)

    def confirmation_criteria(self) -> dict[str, Any]:
        return super().confirmation_criteria()

    def preflight(self) -> PreflightResult:
        try:
            from propab.domain_adapters.materials_adapter import MaterialsAdapter

            df = MaterialsAdapter().load_frame()
            n = int(len(df))
            if n < 500:
                return PreflightResult(False, f"insufficient rows: {n}", {"n_samples": n})
            return PreflightResult(True, "materials frame loaded", {"n_samples": n})
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"materials dataset unavailable: {exc}")

    def domain_profile(self):
        from propab.domain_profiles.materials import MATERIALS_PROFILE

        return MATERIALS_PROFILE
