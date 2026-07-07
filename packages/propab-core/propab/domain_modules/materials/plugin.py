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

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": (
                        "Benchmarking materials property prediction methods: the Matbench "
                        "test set and Automatminer reference algorithm"
                    ),
                    "authors": "A. Dunn, Q. Wang, A. Ganose, et al.",
                    "year": 2020,
                    "doi": "10.1038/s41524-020-00406-3",
                },
            ],
            "search_terms": [
                "matbench", "dielectric constant prediction", "crystal system", "materials informatics",
                "descriptor-property relationship", "leave-one-crystal-system-out",
            ],
            "source_priorities": ["arxiv", "semantic_scholar", "crossref"],
            "classification_codes": {
                "arxiv": ["cond-mat.mtrl-sci"],
            },
            "open_problem_sources": [
                {
                    "name": "Matbench leaderboard (matbench_dielectric task)",
                    "url": "https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_dielectric/",
                },
            ],
            "tabulation_sources": [
                {
                    "name": "Matbench matbench_dielectric",
                    "identifiers": ["matbench_v0.1/matbench_dielectric"],
                    # Best-known-value baseline for rediscovery rejection. The
                    # matbench_dielectric task (predict refractive index n from
                    # structure, 4764 compounds, 5-fold nested CV) has a public
                    # per-task leaderboard; the Automatminer reference algorithm
                    # scores MAE = 0.3057 (log10 refractive index units) /
                    # ~29.09 in the raw-refractive-index reporting, and the
                    # current SOTA (coGN / MODNet) sits near MAE ~0.27. A model
                    # that does not beat these is rediscovery, not novelty.
                    "best_known_metric": "MAE (refractive index)",
                    "automatminer_baseline_mae": 0.3057,
                    "sota_reference_mae": 0.27,
                    "source": "Dunn et al. 2020, npj Comput. Mater. (10.1038/s41524-020-00406-3)",
                },
                {
                    "name": "Materials Project dielectric tensor dataset",
                    "identifiers": ["MP:dielectric"],
                    # DFT (DFPT) dielectric-tensor tabulation underlying the
                    # Matbench task — the authoritative per-material best-known
                    # eps_electronic / refractive-index values.
                    "url": "https://next-gen.materialsproject.org/",
                    "source": "Petousis et al. 2017, Sci. Data (10.1038/sdata.2016.134)",
                },
            ],
            "canonical_surveys": [
                {
                    "title": (
                        "Benchmarking materials property prediction methods: the Matbench "
                        "test set and Automatminer reference algorithm"
                    ),
                    "doi": "10.1038/s41524-020-00406-3",
                },
                {
                    "title": (
                        "High-throughput screening of inorganic compounds for the discovery "
                        "of novel dielectric and optical materials"
                    ),
                    "doi": "10.1038/sdata.2016.134",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a descriptor-to-dielectric relationship "
                "that survives leave-one-crystal-system-out holdout AND beats the Matbench "
                "matbench_dielectric reference baselines (Automatminer MAE ~0.31, SOTA MAE ~0.27 "
                "on log-refractive-index) rather than merely reproducing a per-material value "
                "already tabulated in the Materials Project dielectric-tensor dataset."
            ),
        }
