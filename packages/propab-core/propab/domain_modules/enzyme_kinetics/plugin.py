"""Enzyme kinetics DomainPlugin — BRENDA-style LOFO across EC classes."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.enzyme_kinetics.adapter import (
    KNOWN_FEATURES,
    EnzymeExperimentSpec,
    EnzymeKineticsAdapter,
    dataset_is_synthetic,
)
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
        # Real DLKcat (BRENDA + SABIO-RK derived) kcat data is served when it can
        # be fetched/cached; ``dataset_is_synthetic()`` reads the on-disk meta so
        # findings are labelled honestly (DOM2). Only the network-unavailable
        # fallback frame reports synthetic.
        EnzymeKineticsAdapter().ensure_cache()
        return dataset_is_synthetic()

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 2,
            "requires_holdout": True,
            "holdout_type": "leave_ec_out",
            "null_test": "ec_label_shuffle",
            "verification_type": "statistical",
        }

    def objective_spec(self) -> dict[str, Any]:
        """Enzyme kinetics is scored by a held-out *statistic*, not a trained ML metric.

        The verifier emits ``metric_name="lofo_r2"`` (``enzyme_kinetics/verifier.py``):
        a leave-one-EC-class-out R² measuring whether a kcat/turnover relationship
        survives holding out an entire enzyme family, gated by an EC-label-shuffle
        null. Statistical holdout evidence, not MLP training.

        ``is_ml=False`` stops core from measuring a meaningless trained baseline for
        this family-LOFO fit (the 1ae74abd mis-scoring). The label carries no ML
        token; there is no external best-known table for a subset LOFO R², so the
        baseline is ``"measured"`` from the domain's own holdout.
        """
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
            run_enzyme_experiment(EnzymeExperimentSpec(feature_subset=["molecular_weight", "sequence_length"]))
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
            "open_problem_sources": [
                {
                    "name": "BRENDA enzyme database (kcat/Km frontier)",
                    "url": "https://www.brenda-enzymes.org/",
                },
            ],
            "tabulation_sources": [
                {
                    "name": "BRENDA kinetic parameters",
                    "identifiers": ["BRENDA:kcat", "BRENDA:km", "BRENDA:kcat_km"],
                    # Authoritative per-EC-number tabulation of measured kcat,
                    # Km and turnover values — the reference for whether a
                    # predicted parameter is already a measured, tabulated value.
                    "url": "https://www.brenda-enzymes.org/",
                    "source": "Chang et al. 2021, Nucleic Acids Res. (10.1093/nar/gkaa1025)",
                },
                {
                    "name": "SABIO-RK reaction kinetics",
                    "identifiers": ["SABIO-RK"],
                    "url": "http://sabiork.h-its.org/",
                    "source": "Wittig et al. 2018, Nucleic Acids Res. (10.1093/nar/gkx1065)",
                },
                {
                    "name": "Bar-Even 2011 kcat/Km distribution ceilings",
                    "identifiers": ["kcat_km_ceiling"],
                    # Best-known-value anchors for rediscovery rejection, from
                    # analysis of ~1900 enzymes (Bar-Even et al. 2011):
                    #   median kcat/Km   ~ 1e5   M^-1 s^-1
                    #   median kcat      ~ 10    s^-1
                    #   median Km        ~ 1e-4  M (~130 uM)
                    #   diffusion limit  ~ 1e8 - 1e9 M^-1 s^-1 (hard ceiling)
                    # A "novel high-efficiency enzyme" claiming kcat/Km above the
                    # diffusion limit, or restating the ~1e5 median, is not novel.
                    "median_kcat_km_M_inv_s_inv": 1e5,
                    "median_kcat_s_inv": 10.0,
                    "median_km_M": 1e-4,
                    "diffusion_limit_kcat_km_M_inv_s_inv": [1e8, 1e9],
                    "source": "Bar-Even et al. 2011, Biochemistry (10.1021/bi2002289)",
                },
            ],
            "canonical_surveys": [
                {
                    "title": (
                        "The Moderately Efficient Enzyme: Evolutionary and Physicochemical "
                        "Trends Shaping Enzyme Parameters"
                    ),
                    "doi": "10.1021/bi2002289",
                },
                {
                    "title": "BRENDA, the ELIXIR core data resource in 2021",
                    "doi": "10.1093/nar/gkaa1025",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a kcat/Km relationship that survives "
                "leave-EC-class-out holdout and is not already accounted for by the tabulated "
                "BRENDA/SABIO-RK parameters or by the Bar-Even 2011 distribution anchors "
                "(median kcat/Km ~1e5 M^-1 s^-1, diffusion ceiling 1e8-1e9 M^-1 s^-1). A "
                "claimed kcat/Km above the diffusion limit, or a mere restatement of the ~1e5 "
                "median, is rediscovery, not novelty."
            ),
        }

    def known_value_check(
        self, claim_text: str, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Flag a claim that merely rediscovers an established enzyme-kinetics fact.

        Consumes this plugin's own ``literature_profile()`` Bar-Even 2011 anchors
        (median kcat/Km ~1e5, diffusion-limit ceiling 1e8-1e9) and returns a
        verdict dict with ``trivial_rediscovery=True`` / ``discovery_worthy=False``
        when the claim restates a known value or violates the known ceiling, else
        None. These are the flags ``paper_narrative._is_rediscovery`` reads.
        """
        from propab.domain_modules.enzyme_kinetics.rediscovery import check_rediscovery

        return check_rediscovery(claim_text, evidence, self.literature_profile())

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem")):
            return False
        return any(m in combined for m in self.scope_question_markers)
