"""Proteomics / protein-stability DomainPlugin — leave-one-protein-family-out."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.proteomics.adapter import (
    KNOWN_FEATURES,
    ProteomicsAdapter,
    ProteomicsExperimentSpec,
    dataset_is_synthetic,
)
from propab.domain_modules.proteomics.verifier import (
    classify_proteomics_verdict,
    run_proteomics_experiment,
)


class ProteomicsPlugin(DomainPlugin):
    domain_id = "proteomics"
    display_name = "Proteomics (protein thermostability, leave-one-family-out)"
    version = "1.0"
    scope_question_markers = (
        "protein stability",
        "thermostability",
        "melting temperature",
        "proteomics",
        "protein family",
        "denaturation",
        "protein engineering",
    )
    artifact_question_markers = (
        "protein stability",
        "thermostability",
        "melting temperature",
        "proteomics",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Protein thermostability subset: ~640 proteins across 8 fold families",
            "distribution": "Leave-one-protein-family-out holdout across fold families",
            "claimed_generalization": "Sequence-property stability rule survives an unseen fold family",
            "expected_failure_modes": "Family leakage; length/MW collinearity; fold-surrogate confounds",
            "ood_test": "Leave-family-out R² + within-family Tm-shuffle null p<0.05",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "proteomics":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:proteomics]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "proteomics":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:proteomics]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def uses_synthetic_data(self) -> bool:
        ProteomicsAdapter().ensure_cache()
        return dataset_is_synthetic()

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.85,
            "requires_holdout": True,
            "holdout_type": "leave_family_out",
            "null_test": "within_family_tm_shuffle",
            "verification_type": "statistical",
        }

    def objective_spec(self) -> dict[str, Any]:
        """Scored by a held-out leave-one-family-out R² (``lofo_r2``) gated by a
        within-family Tm-shuffle null — not a trained ML metric. ``is_ml=False``
        keeps core off the ML baseline path; the metric label carries no ML token
        and the baseline is ``"measured"`` from the domain's own holdout."""
        return {
            "metric_name": "lofo_r2",
            "direction": "higher_is_better",
            "is_ml": False,
            "baseline_kind": "measured",
        }

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        spec = ProteomicsExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = ProteomicsExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_proteomics_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_proteomics_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            df = ProteomicsAdapter().load_frame()
            run_proteomics_experiment(
                ProteomicsExperimentSpec(feature_subset=["frac_charged", "frac_proline"])
            )
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"proteomics LOFO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "protein stability subset loaded, leave-family-out ok",
                {"n_proteins": len(df), "n_families": df["family"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"proteomics preflight failed: {exc}")

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "Meltome atlas—thermal proteome stability across the tree of life",
                    "authors": "A. Jarzab, N. Kurzawa, T. Hopf, et al.",
                    "year": 2020,
                    "doi": "10.1038/s41592-020-0801-4",
                },
            ],
            "search_terms": [
                "protein thermostability", "melting temperature", "protein stability",
                "meltome", "thermal proteome", "ddG", "protein fold family", "thermophile",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {"mesh": ["Protein Stability", "Protein Folding", "Temperature"]},
            "open_problem_sources": [{"name": "Meltome Atlas / ProThermDB", "url": "https://meltomeatlas.proteomics.wzw.tum.de/"}],
            "tabulation_sources": [
                {
                    "name": "Thermophile vs mesophile stability heuristics",
                    "identifiers": ["thermostability_heuristic"],
                    # Known trends — a claim that merely restates "thermophilic
                    # proteins have more charged residues / salt bridges and higher
                    # Tm" is a rediscovery of an established stability heuristic.
                    "known_trends": ["charged_residue_enrichment", "salt_bridges", "proline_rigidification"],
                    "source": "Kumar & Nussinov 2001, Cell. Mol. Life Sci. (10.1007/PL00000854)",
                },
            ],
            "canonical_surveys": [
                {"title": "Meltome atlas—thermal proteome stability across the tree of life", "doi": "10.1038/s41592-020-0801-4"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a sequence-property->thermostability "
                "relationship that survives leave-one-protein-family-out holdout and is not a "
                "restatement of the known thermophile charged-residue / proline-rigidification "
                "stability heuristics."
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return ["leave-family-out", "lofo", "family holdout", "tm shuffle", "cross-family"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
