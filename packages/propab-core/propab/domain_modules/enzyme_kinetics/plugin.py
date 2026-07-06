"""Enzyme kinetics DomainPlugin — BRENDA-style LOFO across EC classes."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.enzyme_kinetics.adapter import KNOWN_FEATURES, EnzymeExperimentSpec, EnzymeKineticsAdapter
from propab.domain_modules.enzyme_kinetics.verifier import classify_enzyme_verdict, run_enzyme_experiment


class EnzymeKineticsPlugin(DomainPlugin):
    domain_id = "enzyme_kinetics"
    display_name = "Enzyme kinetics (BRENDA / UniProt families)"
    version = "1.0"
    scope_question_markers = (
        "enzyme",
        "kcat",
        "km",
        "brenda",
        "uniprot",
        "ec class",
        "catalytic turnover",
    )
    artifact_question_markers = (
        "enzyme",
        "kcat",
        "km",
        "brenda",
        "ec class",
        "catalytic",
    )

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "enzyme_kinetics":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:enzyme_kinetics]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        # Score = count of distinct enzyme markers present, so a question rich in
        # enzyme vocabulary outranks a colliding domain (e.g. genomics) instead of
        # losing on registration order.
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "enzyme_kinetics":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:enzyme_kinetics]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "BRENDA subset: ~400 enzymes across 6 EC classes",
            "distribution": "Leave-one-EC-class-out holdout",
            "claimed_generalization": "Kinetic parameter survives held-out enzyme family",
            "expected_failure_modes": "EC-class leakage; collinear MW/pH proxies",
            "ood_test": "LOFO on EC class + label-shuffle null p<0.05",
        }

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def uses_synthetic_data(self) -> bool:
        # The BRENDA/UniProt-style frame is seed-generated (adapter meta
        # ``synthetic: True``), not a real BRENDA subset. Findings must be labelled
        # synthetic (DOM2).
        return True

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 2,
            "requires_holdout": True,
            "holdout_type": "leave_ec_out",
            "null_test": "ec_label_shuffle",
            "verification_type": "statistical",
        }

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        spec = EnzymeExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = EnzymeExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_enzyme_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_enzyme_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            adapter = EnzymeKineticsAdapter()
            df = adapter.load_frame()
            run_enzyme_experiment(EnzymeExperimentSpec(feature_subset=["log_km", "molecular_weight"]))
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"enzyme LOFO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "BRENDA subset loaded, EC LOFO ok",
                {"n_enzymes": len(df), "n_ec_classes": df["ec_class"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"enzyme kinetics preflight failed: {exc}")

    def domain_profile(self):
        from propab.domain_profiles.enzyme_kinetics import ENZYME_KINETICS_PROFILE

        return ENZYME_KINETICS_PROFILE

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": (
                        "The Moderately Efficient Enzyme: Evolutionary and Physicochemical "
                        "Trends Shaping Enzyme Parameters"
                    ),
                    "authors": "A. Bar-Even, E. Noor, Y. Savir, et al.",
                    "year": 2011,
                    "doi": "10.1021/bi2002289",
                },
            ],
            "search_terms": [
                "enzyme kinetics", "kcat", "Km", "catalytic turnover number", "BRENDA",
                "EC class", "enzyme evolution", "Michaelis-Menten", "catalytic efficiency",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {
                "mesh": ["Kinetics", "Enzymes", "Catalysis", "Substrate Specificity"],
            },
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [
                {
                    "title": (
                        "The Moderately Efficient Enzyme: Evolutionary and Physicochemical "
                        "Trends Shaping Enzyme Parameters"
                    ),
                    "doi": "10.1021/bi2002289",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a kcat/Km relationship that survives "
                "leave-EC-class-out holdout and is not already accounted for by the general "
                "evolutionary/physicochemical trends (e.g. catalytic efficiency ceilings) "
                "documented in the enzyme-parameter literature above."
            ),
        }

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem")):
            return False
        return any(m in combined for m in self.scope_question_markers)
