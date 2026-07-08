"""QSAR / drug-target bioactivity DomainPlugin — leave-one-scaffold-out."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.qsar.adapter import (
    KNOWN_FEATURES,
    QSARAdapter,
    QSARExperimentSpec,
    dataset_is_synthetic,
)
from propab.domain_modules.qsar.verifier import classify_qsar_verdict, run_qsar_experiment


class QSARPlugin(DomainPlugin):
    domain_id = "qsar"
    display_name = "QSAR (drug-target bioactivity, leave-one-scaffold-out)"
    version = "1.0"
    scope_question_markers = (
        "qsar",
        "bioactivity",
        "ic50",
        "pic50",
        "structure-activity",
        "molecular descriptor",
        "scaffold",
        "potency",
        "chembl",
    )
    artifact_question_markers = (
        "qsar",
        "bioactivity",
        "ic50",
        "scaffold",
        "descriptor",
        "potency",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Compound-activity subset: ~720 molecules across 8 chemical scaffolds",
            "distribution": "Leave-one-scaffold-out holdout across Bemis-Murcko scaffolds",
            "claimed_generalization": "Structure-activity relationship survives an unseen scaffold",
            "expected_failure_modes": "Scaffold leakage; descriptor collinearity; activity-cliff tautologies",
            "ood_test": "Leave-scaffold-out R² + within-scaffold activity-shuffle null p<0.05",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "qsar":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:qsar]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "qsar":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:qsar]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def uses_synthetic_data(self) -> bool:
        QSARAdapter().ensure_cache()
        return dataset_is_synthetic()

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.85,
            "requires_holdout": True,
            "holdout_type": "leave_scaffold_out",
            "null_test": "within_scaffold_activity_shuffle",
            "verification_type": "statistical",
        }

    def objective_spec(self) -> dict[str, Any]:
        """QSAR is scored by a held-out *statistic*, not a trained ML metric.

        The verifier measures a leave-one-scaffold-out R² (``loso_r2``) gated by a
        within-scaffold activity-shuffle null. ``is_ml=False`` keeps core from
        measuring a meaningless trained baseline; the metric label carries no ML
        token and the baseline is ``"measured"`` from the domain's own holdout.
        """
        return {
            "metric_name": "loso_r2",
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
        spec = QSARExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = QSARExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_qsar_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_qsar_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            df = QSARAdapter().load_frame()
            run_qsar_experiment(QSARExperimentSpec(feature_subset=["clogp", "mol_weight"]))
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"QSAR LOSO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "QSAR compound subset loaded, leave-scaffold-out ok",
                {"n_compounds": len(df), "n_scaffolds": df["scaffold"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"qsar preflight failed: {exc}")

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "Comparative Molecular Field Analysis (CoMFA): effect of shape on binding of steroids",
                    "authors": "R. D. Cramer, D. E. Patterson, J. D. Bunce",
                    "year": 1988,
                    "doi": "10.1021/ja00226a005",
                },
            ],
            "search_terms": [
                "QSAR", "structure-activity relationship", "pIC50", "molecular descriptors",
                "scaffold hopping", "ChEMBL", "activity cliff", "Lipinski rule of five",
            ],
            "source_priorities": ["pubmed", "europepmc", "chemrxiv", "semantic_scholar", "crossref"],
            "classification_codes": {"mesh": ["Structure-Activity Relationship", "Drug Design"]},
            "open_problem_sources": [{"name": "ChEMBL bioactivity database", "url": "https://www.ebi.ac.uk/chembl/"}],
            "tabulation_sources": [
                {
                    "name": "Lipinski rule-of-five property ceilings",
                    "identifiers": ["ro5"],
                    # Known drug-likeness thresholds — a claim that merely restates
                    # MW<=500, cLogP<=5, HBD<=5, HBA<=10 is a rediscovery.
                    "mw_max": 500.0,
                    "clogp_max": 5.0,
                    "hbd_max": 5,
                    "hba_max": 10,
                    "source": "Lipinski et al. 2001, Adv. Drug Deliv. Rev. (10.1016/S0169-409X(00)00129-0)",
                },
            ],
            "canonical_surveys": [
                {"title": "QSAR modeling: where have you been? Where are you going to?", "doi": "10.1021/jm4004285"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a descriptor->potency relationship that "
                "survives leave-one-scaffold-out holdout and is not a mere restatement of the "
                "Lipinski rule-of-five drug-likeness thresholds."
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return ["leave-scaffold-out", "loso", "scaffold holdout", "activity shuffle", "cross-scaffold"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
