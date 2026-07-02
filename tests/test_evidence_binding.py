"""Evidence Binding — regression tests (fixes.md Phase 1)."""
from __future__ import annotations

from propab.belief_state import CampaignBeliefState
from propab.evidence_binding import (
    BindingMetrics,
    belief_falsifiable_in_dataset,
    binding_check_statement_to_node,
    extract_subject_from_statement,
    filter_node_citations,
    infer_test_targets,
)


INSUFFICIENCY_BELIEF = (
    "The current 98-feature biophysical set is fundamentally insufficient to capture RT activity "
    "variance; intra-family models will fail to achieve positive R2 even in random-split scenarios."
)

LOFO_NODE = {
    "id": "00d9bcda-5ae8-49ed-8435-50a89d362b2d",
    "text": (
        "Does a binary classifier achieve positive LOFO AUC using biophysical features? "
        "Population: 56 RTs. OOD test: leave-one-family-out."
    ),
    "verdict": "refuted",
    "claim_scope": {
        "population": "56 RTs across 7 families",
        "distribution": "all families",
        "claimed_generalization": "cross-family LOFO",
        "ood_test": "leave-one-family-out",
    },
}

WITHIN_FAMILY_NODE = {
    "id": "within-1",
    "text": (
        "Intra-family clustered vs random split benchmark for Ridge on 98 features. "
        "Population: single family holdout."
    ),
    "verdict": "refuted",
    "claim_scope": {
        "population": "single family",
        "claimed_generalization": "within-family only",
        "ood_test": "clustered split holdout",
    },
}


def test_fabricated_lofo_citation_rejected_for_insufficiency_belief() -> None:
    result = binding_check_statement_to_node(INSUFFICIENCY_BELIEF, LOFO_NODE)
    assert not result.match
    assert "lofo" in result.reason.lower() or "insufficient" in result.reason.lower() or "overlap" in result.reason.lower()


def test_within_family_citation_accepts_for_redundancy_belief() -> None:
    belief = (
        "Observed intra-family predictive signals are artifacts of sequence redundancy; "
        "model performance collapses under clustered split holdout."
    )
    result = binding_check_statement_to_node(belief, WITHIN_FAMILY_NODE)
    assert result.match


def test_filter_node_citations_strips_fabricated_support() -> None:
    nodes = {LOFO_NODE["id"]: LOFO_NODE, WITHIN_FAMILY_NODE["id"]: WITHIN_FAMILY_NODE}
    metrics = BindingMetrics()
    accepted = filter_node_citations(
        INSUFFICIENCY_BELIEF,
        [LOFO_NODE["id"], WITHIN_FAMILY_NODE["id"]],
        nodes,
        metrics=metrics,
    )
    assert LOFO_NODE["id"] not in accepted
    assert metrics.binding_rejected_count >= 1


def test_belief_apply_rejects_unfalsifiable_insufficiency() -> None:
    state = CampaignBeliefState()
    state.rival_exhaustion_mode = True
    metrics = BindingMetrics()
    state.apply_synthesis_beliefs(
        [{"statement": INSUFFICIENCY_BELIEF, "confidence": "weak", "status": "active"}],
        tree_nodes={LOFO_NODE["id"]: LOFO_NODE},
        metrics=metrics,
    )
    assert len(state.active_beliefs) == 0
    assert metrics.falsifiability_rejected_count == 1


def test_rival_mode_hard_cap_two_beliefs() -> None:
    state = CampaignBeliefState()
    state.rival_exhaustion_mode = True
    metrics = BindingMetrics()
    state.apply_synthesis_beliefs(
        [
            {"statement": "Rival A within-family signal exists under clustered split.", "status": "active"},
            {"statement": "Rival B redundancy artifact collapses under low identity holdout.", "status": "active"},
            {"statement": "Rival C should be rejected by cap.", "status": "active"},
        ],
        metrics=metrics,
    )
    assert len(state.active_beliefs) == 2
    assert metrics.belief_cap_rejected_count == 1


def test_feature_insufficiency_not_falsifiable() -> None:
    ok, reason = belief_falsifiable_in_dataset(INSUFFICIENCY_BELIEF)
    assert not ok
    assert "unfalsifiable" in reason


def test_insufficiency_subject_tags() -> None:
    subj = extract_subject_from_statement(INSUFFICIENCY_BELIEF)
    assert "feature_sufficiency" in subj.test_targets


def test_lofo_node_tags() -> None:
    tags = infer_test_targets(LOFO_NODE["text"])
    assert "cross_family_lofo" in tags
