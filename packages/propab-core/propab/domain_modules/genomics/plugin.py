"""Genomics DomainPlugin — cross-tissue GTEx LOFO verification."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.genomics.adapter import KNOWN_FEATURES
from propab.domain_modules.genomics.problems import get_literature_prior
from propab.domain_modules.genomics.verifier import (
    classify_genomics_verdict,
    run_genomics_experiment,
)
from propab.domain_modules.genomics.adapter import GenomicsAdapter, GenomicsExperimentSpec


class GenomicsPlugin(DomainPlugin):
    domain_id = "genomics"
    display_name = "Genomics (GTEx cross-tissue expression)"
    version = "1.0"
    scope_question_markers = (
        "gene expression",
        "gtex",
        "cross-tissue",
        "tissue specificity",
        "genomics",
        "eqtl",
    )
    artifact_question_markers = (
        "gene expression",
        "gtex",
        "tissue",
        "housekeeping",
        "tau index",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "GTEx v8 subset: 1000 variable genes × 10 tissues",
            "distribution": "Leave-one-tissue-out holdout across tissue types",
            "claimed_generalization": "Expression pattern survives held-out tissue",
            "expected_failure_modes": "Tissue-label leakage; housekeeping-only tautologies",
            "ood_test": "Leave-tissue-out LOFO + tissue-label shuffle null p<0.05",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "genomics":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:genomics]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        # Score = count of distinct genomics markers present, so an
        # enzyme-vs-genomics collision resolves to whichever domain's vocabulary
        # is more prevalent rather than to registration order.
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "genomics":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:genomics]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 2,
            "min_confidence": 0.85,
            "requires_holdout": True,
            "holdout_type": "leave_tissue_out",
            "null_test": "tissue_label_shuffle",
            "verification_type": "statistical",
        }

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        spec = GenomicsExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = GenomicsExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_genomics_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_genomics_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            import numpy as np
            from scipy import stats  # noqa: F401

            t0 = time.time()
            adapter = GenomicsAdapter()
            df = adapter.load_frame()
            n_genes = df["gene_id"].nunique()
            n_tissues = df["tissue"].nunique()
            spec = GenomicsExperimentSpec(feature_subset=["expression_variance", "mean_expression"])
            run_genomics_experiment(spec)
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"LOFO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "GTEx subset loaded, LOFO preflight ok",
                {"n_genes": n_genes, "n_tissues": n_tissues, "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"genomics preflight failed: {exc}")

    def literature_prior(self, question: str) -> dict[str, Any]:
        return get_literature_prior(question)

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "The GTEx Consortium atlas of genetic regulatory effects across human tissues",
                    "authors": "GTEx Consortium",
                    "year": 2020,
                    "doi": "10.1126/science.aaz1776",
                },
                {
                    "title": (
                        "Genome-wide midrange transcription profiles reveal expression level "
                        "relationships in human tissue specification"
                    ),
                    "authors": "I. Yanai, H. Benjamin, M. Shmoish, et al.",
                    "year": 2005,
                    "doi": "10.1093/bioinformatics/bti042",
                },
            ],
            "search_terms": [
                "GTEx", "cross-tissue gene expression", "tissue specificity", "tau index",
                "housekeeping gene", "eQTL", "leave-tissue-out", "tissue-specific expression",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {
                "mesh": ["Gene Expression Profiling", "Organ Specificity", "Transcriptome"],
            },
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [
                {
                    "title": "The GTEx Consortium atlas of genetic regulatory effects across human tissues",
                    "doi": "10.1126/science.aaz1776",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a cross-tissue expression relationship "
                "(e.g. a tissue-specificity/tau-index effect surviving leave-tissue-out holdout) "
                "that is not a restatement of housekeeping-vs-tissue-specific classification "
                "already established in the GTEx atlas and tau-index literature above."
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return ["leave-tissue-out", "lofo", "tissue label shuffle", "cross-tissue"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
