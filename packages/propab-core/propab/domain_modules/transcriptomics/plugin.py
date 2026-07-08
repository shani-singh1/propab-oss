"""Transcriptomics / gene-regulation DomainPlugin — leave-one-condition-out."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.transcriptomics.adapter import (
    KNOWN_FEATURES,
    TranscriptomicsAdapter,
    TranscriptomicsExperimentSpec,
    dataset_is_synthetic,
)
from propab.domain_modules.transcriptomics.verifier import (
    classify_transcriptomics_verdict,
    run_transcriptomics_experiment,
)


class TranscriptomicsPlugin(DomainPlugin):
    domain_id = "transcriptomics"
    display_name = "Transcriptomics (gene regulation, leave-one-condition-out)"
    version = "1.0"
    scope_question_markers = (
        "gene regulation",
        "transcriptomics",
        "differential expression",
        "fold change",
        "promoter",
        "transcription factor",
        "regulatory",
        "perturbation",
        "expression response",
    )
    artifact_question_markers = (
        "gene regulation",
        "transcriptomics",
        "differential expression",
        "promoter",
        "fold change",
        "regulatory",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Gene-regulation subset: ~880 gene-condition rows across 8 perturbation conditions",
            "distribution": "Leave-one-condition-out holdout across experimental perturbations",
            "claimed_generalization": "Promoter-feature regulatory rule survives an unseen condition",
            "expected_failure_modes": "Condition leakage; batch confounds; promoter-feature collinearity",
            "ood_test": "Leave-condition-out R² + within-condition fold-change-shuffle null p<0.05",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "transcriptomics":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:transcriptomics]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "transcriptomics":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:transcriptomics]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def uses_synthetic_data(self) -> bool:
        TranscriptomicsAdapter().ensure_cache()
        return dataset_is_synthetic()

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.85,
            "requires_holdout": True,
            "holdout_type": "leave_condition_out",
            "null_test": "within_condition_fold_change_shuffle",
            "verification_type": "statistical",
        }

    def objective_spec(self) -> dict[str, Any]:
        """Scored by a held-out leave-one-condition-out R² (``loco_r2``) gated by a
        within-condition fold-change-shuffle null — not a trained ML metric.
        ``is_ml=False`` keeps core off the ML baseline path; the metric label
        carries no ML token and the baseline is ``"measured"`` from the domain's
        own holdout."""
        return {
            "metric_name": "loco_r2",
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
        spec = TranscriptomicsExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = TranscriptomicsExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_transcriptomics_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_transcriptomics_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            df = TranscriptomicsAdapter().load_frame()
            run_transcriptomics_experiment(
                TranscriptomicsExperimentSpec(feature_subset=["tf_motif_count", "chromatin_accessibility"])
            )
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"transcriptomics LOCO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "gene-regulation subset loaded, leave-condition-out ok",
                {"n_rows": len(df), "n_conditions": df["condition"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"transcriptomics preflight failed: {exc}")

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "The accessible chromatin landscape of the human genome",
                    "authors": "R. E. Thurman, E. Rynes, R. Humbert, et al.",
                    "year": 2012,
                    "doi": "10.1038/nature11232",
                },
            ],
            "search_terms": [
                "gene regulation", "differential expression", "transcription factor binding",
                "promoter", "chromatin accessibility", "ATAC-seq", "cis-regulatory", "perturbation response",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {"mesh": ["Gene Expression Regulation", "Promoter Regions, Genetic", "Transcription Factors"]},
            "open_problem_sources": [{"name": "GEO / ENCODE regulatory catalogues", "url": "https://www.ncbi.nlm.nih.gov/geo/"}],
            "tabulation_sources": [
                {
                    "name": "Canonical cis-regulatory element facts",
                    "identifiers": ["tata_box", "cpg_island"],
                    # Known regulatory facts — a claim that merely restates "TATA
                    # boxes sit ~25-30 bp upstream of the TSS" or "CpG islands mark
                    # housekeeping promoters" is a rediscovery.
                    "tata_offset_bp": [-30, -25],
                    "cpg_island_gc_min": 0.5,
                    "source": "Smale & Kadonaga 2003, Annu. Rev. Biochem. (10.1146/annurev.biochem.72.121801.161520)",
                },
            ],
            "canonical_surveys": [
                {"title": "The accessible chromatin landscape of the human genome", "doi": "10.1038/nature11232"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a promoter-feature->expression-response "
                "relationship that survives leave-one-condition-out holdout and is not a "
                "restatement of the canonical TATA-box position / CpG-island promoter facts."
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return ["leave-condition-out", "loco", "condition holdout", "fold-change shuffle", "cross-condition"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
