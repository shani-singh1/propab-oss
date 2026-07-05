"""Mandrake DomainPlugin — delegates to the (already tested) mandrake adapter."""
from __future__ import annotations

from typing import Any

from propab.belief_state import CampaignBeliefState
from propab.domain_modules.base import DomainPlugin, PreflightResult

CONTRARIAN_QUESTION = (
    "Is RT activity, as measured across these 7 evolutionary families, one shared biophysical "
    "mechanism currently confounded by family-correlated nuisance variables — or are there "
    "genuinely distinct, family-specific activity mechanisms, such that no single feature set "
    "could ever predict activity across families, because \"activity\" does not refer to the same "
    "underlying physical process in each family?"
)

CONTRARIAN_BELIEF_FAMILY_SPECIFIC = (
    "RT activity in each family is governed by mechanisms specific to that family's evolutionary "
    "and structural context. Predictive signals exist within families even when sequence redundancy "
    "(nearest-neighbor effects) is controlled via clustered splitting."
)

CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT = (
    "Observed intra-family predictive signals are artifacts of sequence redundancy; model performance "
    "will collapse (R2 < 0) when the test set is restricted to sequences with <50% identity to the "
    "training set."
)

CONTRARIAN_ORCHESTRATOR_DIRECTIVE = (
    "Primary critical-experiment criterion: choose the next test because its result would move "
    "Belief 1 (family-specific signal under clustered split) and Belief 2 (sequence-redundancy "
    "artifact under low-identity holdout) in opposite directions — not because it refines either belief in isolation. "
    "Prioritize within-family models that discriminate between these rivals over further "
    "cross-family LOFO feature-combination searches, which have already run exhaustively under "
    "the prior framing. Do not silently revert to cross-family LOFO search as a fallback. "
    "Belief 2 must clear the same artifact-verification bar as Belief 1."
)


class MandrakePlugin(DomainPlugin):
    domain_id = "mandrake"
    display_name = "Mandrake Retroviral Wall (RT-family LOFO)"
    version = "1.0"
    scope_question_markers = (
        "rt activity",
        "retroviral",
        "biophysical",
        "evolutionary family",
        "mandrake",
    )
    theme_rules = (
        ("thermal_stability", ("t55_raw", "t70_raw", "t75_raw", "t80_raw", "thermal", "thermophilicity", "denaturation", "melting")),
        ("catalytic_geometry", ("triad_best_rmsd", "d1_d2_dist", "d2_d3_dist", "ramachandran", "catalytic triad", "geometry", "yxdd")),
        ("electrostatics", ("mean_pot", "net_charge", "isoelectric", "salt_bridge", "electrostatic", "pocket_hbond")),
        ("fold_similarity", ("foldseek", "tm_score", "lddt", "structural similarity")),
        ("surface_properties", ("camsol", "sasa", "hydrophobic", "surface area")),
        ("motif_structure", ("dgr_motif", "qg_motif", "sp_motif", "motif", "yxdd")),
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "56 retroviral RT sequences with handcrafted biophysical features",
            "distribution": "7 rt_family groups with ≥4 sequences each",
            "claimed_generalization": "Signal must survive leave-one-family-out across held-out families",
            "expected_failure_modes": (
                "Collapses when geometry/fold features proxy family ID; thermal-only axis"
            ),
            "ood_test": "LOFO on held-out family; label-shuffle permutation p<0.05 required before confirm",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        # Mirrors the historical is_mandrake_campaign.
        if payload and str(payload.get("domain") or "") == "mandrake":
            return True
        q = (question or "").lower()
        markers = ("rt activity", "reverse transcriptase", "evolutionary family", "biophysical propert")
        return sum(1 for m in markers if m in q) >= 2

    def available_features(self) -> list[str]:
        from propab.domain_adapters.mandrake_adapter import _KNOWN_FEATURES

        return list(_KNOWN_FEATURES)

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        from propab.domain_adapters.mandrake_adapter import (
            MandrakeAdapter,
            MandrakeExperimentSpec,
        )

        hyp = dict(hypothesis)
        if features:
            hyp["feature_subset"] = list(features)
        spec = MandrakeExperimentSpec.from_hypothesis(hyp, question=str(hyp.get("question", "")))
        return MandrakeAdapter().run_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        from propab.domain_adapters.mandrake_adapter import classify_mandrake_verdict

        return classify_mandrake_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            from propab.domain_adapters.mandrake_adapter import MandrakeAdapter

            df = MandrakeAdapter().load_frame()
            n = int(len(df))
            if n < 20:
                return PreflightResult(False, f"insufficient rows: {n}", {"n_samples": n})
            return PreflightResult(True, "mandrake frame loaded", {"n_samples": n})
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"mandrake dataset unavailable: {exc}")

    def domain_profile(self):
        # Mandrake uses the generic artifact gate (no dedicated profile registered);
        # the enzyme-kinetics profile is the closest family-LOFO analogue but the
        # adapter historically applies the generic gate override, so keep None.
        return None

    def literature_profile(self) -> dict[str, Any]:
        return {
            "seed_papers": [
                {
                    "title": "Origin and evolution of retroelements based upon their reverse transcriptase sequences.",
                    "authors": "Y. Xiong, T. H. Eickbush",
                    "year": 1990,
                    "doi": "10.1002/j.1460-2075.1990.tb07536.x",
                },
            ],
            "search_terms": [
                "reverse transcriptase", "retroelement evolution", "RT phylogeny",
                "catalytic triad geometry", "thermal stability enzyme", "protein structural family",
                "sequence identity clustered split",
            ],
            "source_priorities": ["pubmed", "europepmc", "biorxiv", "semantic_scholar", "crossref"],
            "classification_codes": {
                "mesh": ["RNA-Directed DNA Polymerase", "Retroelements", "Evolution, Molecular"],
            },
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [
                {
                    "title": "Origin and evolution of retroelements based upon their reverse transcriptase sequences.",
                    "doi": "10.1002/j.1460-2075.1990.tb07536.x",
                },
            ],
            "novelty_criteria": (
                "A finding is novel if it establishes a within-family predictive signal for RT "
                "activity that survives both leave-one-family-out holdout and a low-sequence-"
                "identity split (ruling out the sequence-redundancy artifact), rather than "
                "restating the RT phylogenetic family structure already established by Xiong & "
                "Eickbush (1990)."
            ),
        }

    def apply_contrarian_belief_reset(
        self,
        belief_state: CampaignBeliefState,
        *,
        orchestrator_directive: str | None = None,
        close_prior_reason: str = "superseded by contrarian reframing (fixes.md)",
    ) -> CampaignBeliefState:
        """Data-preserving resume: close prior active beliefs, seed two rival beliefs."""
        for belief in list(belief_state.active_beliefs):
            belief_state.abandon_belief(belief, close_prior_reason)

        belief_state.apply_synthesis_beliefs([
            {
                "statement": CONTRARIAN_BELIEF_FAMILY_SPECIFIC,
                "confidence": "weak",
                "status": "active",
                "supporting_nodes": [],
                "contradicting_nodes": [],
            },
            {
                "statement": CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT,
                "confidence": "weak",
                "status": "active",
                "supporting_nodes": [],
                "contradicting_nodes": [],
            },
        ], allow_ungrounded=True)
        belief_state.exhaustion_rounds = 0
        belief_state.branch_exhausted = False
        belief_state.rival_exhaustion_mode = True
        belief_state.results_since_last_synthesis = 0
        belief_state.recent_activity_summary = (
            "Contrarian reframing: discriminate unified-mechanism vs family-specific-mechanism rivals."
        )
        directive = (orchestrator_directive or CONTRARIAN_ORCHESTRATOR_DIRECTIVE).strip()
        if directive:
            belief_state.add_human_message(directive)
        return belief_state
