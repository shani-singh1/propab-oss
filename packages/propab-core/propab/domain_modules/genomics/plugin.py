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
from propab.domain_modules.genomics.adapter import (
    GenomicsAdapter,
    GenomicsExperimentSpec,
    dataset_is_synthetic,
)


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

    def uses_synthetic_data(self) -> bool:
        # Real GTEx v8 median-TPM data is served when it can be fetched/cached;
        # ``dataset_is_synthetic()`` reads the on-disk meta so findings are
        # labelled honestly (DOM2). Only the network-unavailable fallback frame
        # reports synthetic.
        GenomicsAdapter().ensure_cache()
        return dataset_is_synthetic()

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
            "open_problem_sources": [
                {
                    "name": "GTEx portal (tissue-specificity frontier)",
                    "url": "https://gtexportal.org/home/",
                },
            ],
            "tabulation_sources": [
                {
                    "name": "Tau tissue-specificity benchmark (Kryuchkova-Mostacci & Robinson-Rechavi 2017)",
                    "identifiers": ["tau", "tissue_specificity_index"],
                    # Best-known reference values for rediscovery rejection. Tau
                    # ranges [0,1]; the human transcriptome tau distribution is
                    # bimodal with a validated split at ~0.5 (below = broadly
                    # expressed / housekeeping, above = tissue-specific). Tau and
                    # the Gini index were benchmarked as the best-performing
                    # tissue-specificity metrics. A "novel housekeeping vs
                    # tissue-specific" split that just reproduces the tau~0.5
                    # threshold is rediscovery.
                    "tau_range": [0.0, 1.0],
                    "housekeeping_threshold_tau": 0.5,
                    "source": "Kryuchkova-Mostacci & Robinson-Rechavi 2017, Brief. Bioinform. (10.1093/bib/bbw008)",
                },
                {
                    "name": "GTEx v8 median gene-level TPM by tissue",
                    "identifiers": ["GTEx:median_tpm"],
                    # Authoritative per-gene, per-tissue expression tabulation the
                    # tau values are computed from — the ground-truth table for
                    # whether a cross-tissue expression value is already known.
                    "url": "https://gtexportal.org/home/datasets",
                    "source": "GTEx Consortium 2020, Science (10.1126/science.aaz1776)",
                },
                {
                    "name": "Housekeeping gene reference set (Eisenberg & Levanon 2013)",
                    "identifiers": ["HRT_Atlas", "housekeeping_genes"],
                    # Canonical human housekeeping genes — the textbook constitutive
                    # set used as the ground truth for "this gene is broadly/constant-
                    # ly expressed". A claim that one of these is a housekeeping /
                    # low-tissue-specificity / constitutively-expressed gene is a
                    # rediscovery of an established fact, not a novel finding. Both
                    # HGNC symbols and their Ensembl accessions are listed so a claim
                    # keyed by either id form is caught.
                    "housekeeping_genes": [
                        "ACTB", "GAPDH", "TUBB", "B2M", "PPIA", "HPRT1", "TBP",
                        "GUSB", "RPL13A", "RPLP0", "PGK1", "YWHAZ", "SDHA", "UBC",
                    ],
                    "housekeeping_gene_ensembl": [
                        "ENSG00000075624", "ENSG00000111640", "ENSG00000196230",
                        "ENSG00000166710", "ENSG00000196262", "ENSG00000165704",
                        "ENSG00000112592", "ENSG00000169919", "ENSG00000161016",
                        "ENSG00000089157", "ENSG00000102144", "ENSG00000164924",
                        "ENSG00000073578", "ENSG00000150991",
                    ],
                    "source": "Eisenberg & Levanon 2013, Trends Genet. (10.1016/j.tig.2013.05.010)",
                },
            ],
            "canonical_surveys": [
                {
                    "title": "The GTEx Consortium atlas of genetic regulatory effects across human tissues",
                    "doi": "10.1126/science.aaz1776",
                },
                {
                    "title": "A benchmark of gene expression tissue-specificity metrics",
                    "doi": "10.1093/bib/bbw008",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a cross-tissue expression relationship "
                "(e.g. a tissue-specificity/tau-index effect surviving leave-tissue-out holdout) "
                "that is not a restatement of the housekeeping-vs-tissue-specific split already "
                "captured by the tau~0.5 threshold (Kryuchkova-Mostacci & Robinson-Rechavi 2017) "
                "and not merely a per-gene value already tabulated in the GTEx v8 median-TPM "
                "table or the Eisenberg-Levanon housekeeping reference set."
            ),
        }

    def known_value_check(
        self, claim_text: str, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Flag a claim that merely rediscovers an established genomics fact.

        Consumes this plugin's own ``literature_profile()`` reference anchors
        (canonical housekeeping-gene set + tau~0.5 tissue-specificity threshold)
        and returns a verdict dict with ``trivial_rediscovery=True`` /
        ``discovery_worthy=False`` when the claim restates a known value, else
        None. The flags are the ones ``paper_narrative._is_rediscovery`` reads,
        so a flagged finding is labelled "rediscovery (known value)" and dropped
        from the headline discovery count.
        """
        from propab.domain_modules.genomics.rediscovery import check_rediscovery

        return check_rediscovery(claim_text, evidence, self.literature_profile())

    def implementable_methodologies(self) -> list[str]:
        return ["leave-tissue-out", "lofo", "tissue label shuffle", "cross-tissue"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
