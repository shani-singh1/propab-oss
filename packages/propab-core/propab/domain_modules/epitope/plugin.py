"""Immunology / epitope DomainPlugin — leave-one-MHC-allele-out."""
from __future__ import annotations

import time
from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.epitope.adapter import (
    KNOWN_FEATURES,
    EpitopeAdapter,
    EpitopeExperimentSpec,
    dataset_is_synthetic,
)
from propab.domain_modules.epitope.verifier import classify_epitope_verdict, run_epitope_experiment


class EpitopePlugin(DomainPlugin):
    domain_id = "epitope"
    display_name = "Immunology / epitope (peptide-MHC, leave-one-allele-out)"
    version = "1.0"
    scope_question_markers = (
        "epitope",
        "mhc",
        "hla",
        "peptide binding",
        "immunogenic",
        "t-cell",
        "neoantigen",
        "binding affinity",
        "allele",
    )
    artifact_question_markers = (
        "epitope",
        "mhc",
        "hla",
        "peptide",
        "immunogenic",
        "allele",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "Peptide-MHC binding subset: ~720 peptides across 8 HLA alleles",
            "distribution": "Leave-one-MHC-allele-out holdout across HLA alleles",
            "claimed_generalization": "Peptide-property binding rule survives an unseen allele",
            "expected_failure_modes": "Allele leakage; anchor-motif memorization; length confounds",
            "ood_test": "Leave-allele-out R² + within-allele binding-shuffle null p<0.05",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "epitope":
            return True
        q = (question or "").lower()
        hits = sum(1 for m in self.scope_question_markers if m in q)
        return hits >= 2 or "[domain_profile:epitope]" in q

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        if payload and str(payload.get("domain") or payload.get("domain_profile") or "") == "epitope":
            return float(len(self.scope_question_markers))
        q = (question or "").lower()
        if "[domain_profile:epitope]" in q:
            return float(len(self.scope_question_markers))
        return float(sum(1 for m in self.scope_question_markers if m in q))

    def available_features(self) -> list[str]:
        return list(KNOWN_FEATURES)

    def uses_synthetic_data(self) -> bool:
        EpitopeAdapter().ensure_cache()
        return dataset_is_synthetic()

    def confirmation_criteria(self) -> dict[str, Any]:
        return {
            "min_metric_steps_for_confirm": 1,
            "min_confidence": 0.85,
            "requires_holdout": True,
            "holdout_type": "leave_allele_out",
            "null_test": "within_allele_binding_shuffle",
            "verification_type": "statistical",
        }

    def objective_spec(self) -> dict[str, Any]:
        """Scored by a held-out leave-one-allele-out R² (``laoo_r2``) gated by a
        within-allele binding-shuffle null — not a trained ML metric. ``is_ml=False``
        keeps core off the ML baseline path; the label carries no ML token and the
        baseline is ``"measured"`` from the domain's own holdout."""
        return {
            "metric_name": "laoo_r2",
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
        spec = EpitopeExperimentSpec.from_hypothesis(hypothesis)
        if features:
            spec = EpitopeExperimentSpec(feature_subset=list(features), target_column=spec.target_column)
        return run_epitope_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        return classify_epitope_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            t0 = time.time()
            df = EpitopeAdapter().load_frame()
            run_epitope_experiment(
                EpitopeExperimentSpec(feature_subset=["anchorC_hydrophobicity", "net_charge"])
            )
            elapsed = time.time() - t0
            if elapsed > 60:
                return PreflightResult(False, f"epitope LAOO too slow: {elapsed:.1f}s", {"elapsed_sec": elapsed})
            return PreflightResult(
                True,
                "epitope subset loaded, leave-allele-out ok",
                {"n_peptides": len(df), "n_alleles": df["allele"].nunique(), "elapsed_sec": round(elapsed, 2)},
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"epitope preflight failed: {exc}")

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "NetMHCpan-4.1 and NetMHCIIpan-4.0: improved predictions of MHC antigen presentation",
                    "authors": "B. Reynisson, B. Alvarez, S. Paul, et al.",
                    "year": 2020,
                    "doi": "10.1093/nar/gkaa379",
                },
            ],
            "search_terms": [
                "MHC binding", "peptide-MHC", "HLA", "epitope prediction", "immunogenicity",
                "neoantigen", "NetMHCpan", "T-cell epitope", "anchor residue",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {"mesh": ["Histocompatibility Antigens Class I", "Epitopes, T-Lymphocyte", "Protein Binding"]},
            "open_problem_sources": [{"name": "IEDB (immune epitope database)", "url": "https://www.iedb.org/"}],
            "tabulation_sources": [
                {
                    "name": "Canonical HLA class-I anchor motifs / length preference",
                    "identifiers": ["anchor_motif", "peptide_length_9mer"],
                    # Known immunology facts — a claim that HLA-A*02:01 prefers
                    # 9-mers with anchors at P2 (Leu/Met) and PΩ (Val/Leu) is a
                    # rediscovery of a textbook binding motif.
                    "preferred_length": 9,
                    "a0201_anchor_p2": ["L", "M"],
                    "a0201_anchor_pomega": ["V", "L"],
                    "source": "Rammensee et al. 1999, Immunogenetics (SYFPEITHI); IEDB",
                },
            ],
            "canonical_surveys": [
                {"title": "The Immune Epitope Database (IEDB): 2018 update", "doi": "10.1093/nar/gky1006"},
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a peptide-property->binding relationship "
                "that survives leave-one-MHC-allele-out holdout and is not a restatement of the "
                "canonical 9-mer anchor-motif preferences already tabulated for common HLA alleles."
            ),
        }

    def implementable_methodologies(self) -> list[str]:
        return ["leave-allele-out", "laoo", "allele holdout", "binding shuffle", "cross-allele"]

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        combined = f"{text} {methodology or ''}".lower()
        if any(x in combined for x in ("sidon", "cap-set", "docker", "filesystem", "subprocess")):
            return False
        return any(m in combined for m in self.scope_question_markers)
