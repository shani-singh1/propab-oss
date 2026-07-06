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
            {"statement": "Rival A within-family signal exists under clustered split.", "status": "active", "supporting_nodes": ["n1"]},
            {"statement": "Rival B redundancy artifact collapses under low identity holdout.", "status": "active", "supporting_nodes": ["n1"]},
            {"statement": "Rival C should be rejected by cap.", "status": "active", "supporting_nodes": ["n1"]},
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


# --- Checklist 3: Evidence Binding at WRITE TIME on a real synthesis pass ---

REDUNDANCY_BELIEF = (
    "Observed intra-family predictive signals are artifacts of sequence redundancy; "
    "model performance collapses under clustered split holdout."
)

# A citation that has nothing to do with the belief's claim — a fabricated
# "supporting" node from an unrelated subject. Binding must strip it before the
# belief is ever persisted to supporting_nodes.
FABRICATED_UNRELATED_NODE = {
    "id": "fabricated-vit-1",
    "text": (
        "Does adding dropout to a vision transformer improve ImageNet top-1 accuracy? "
        "Population: ImageNet images. OOD test: held-out image classes."
    ),
    "verdict": "confirmed",
    "claim_scope": {
        "population": "ImageNet images",
        "claimed_generalization": "vision models",
        "ood_test": "held-out image classes",
    },
}


def test_synthesis_pass_rejects_fabricated_citation_before_persist() -> None:
    """A fabricated citation must be removed at write time, on the real synthesis pass.

    This drives ``apply_synthesis_to_frontier`` (the orchestrator write path) rather
    than calling the binding check in isolation, proving rejection happens before the
    BeliefObject is persisted to ``supporting_nodes`` (ownership contract: Evidence
    Binding — "citations that fail binding are removed before the citing object is
    persisted").
    """
    from propab.campaign_synthesis import apply_synthesis_to_frontier
    from propab.hypothesis_tree import HypothesisTree

    tree = HypothesisTree()
    # apply_synthesis_to_frontier accepts nodes that are dicts or objects with
    # to_dict(); inject the relevant + fabricated nodes directly.
    tree.nodes = {  # type: ignore[assignment]
        WITHIN_FAMILY_NODE["id"]: WITHIN_FAMILY_NODE,
        FABRICATED_UNRELATED_NODE["id"]: FABRICATED_UNRELATED_NODE,
    }
    state = CampaignBeliefState()
    parsed = {
        "beliefs": [
            {
                "statement": REDUNDANCY_BELIEF,
                "confidence": "weak",
                "status": "active",
                "supporting_nodes": [
                    WITHIN_FAMILY_NODE["id"],
                    FABRICATED_UNRELATED_NODE["id"],
                ],
            },
        ],
        "frontier_candidates": [],
    }

    _added, metrics = apply_synthesis_to_frontier(
        tree,
        state,
        parsed,
        question="RT activity sequence redundancy under clustered split",
        generation=1,
    )

    assert len(state.active_beliefs) == 1
    belief = state.active_beliefs[0]
    # Relevant citation survives; fabricated one is stripped BEFORE persistence.
    assert WITHIN_FAMILY_NODE["id"] in belief.supporting_nodes
    assert FABRICATED_UNRELATED_NODE["id"] not in belief.supporting_nodes
    assert int(metrics.get("binding_rejected_count") or 0) >= 1


# --- Audit A2: binding is conservative-by-design (CLEARED), pinned on examples ---
#
# The binding is surface-lexical: it accepts a citation only when it can find
# shared deterministic subject tags / population scope. The audit asked whether a
# genuinely-supporting node with few surface words in common is WRONGLY rejected.
# The answer is that such a node is indeed rejected — but so is a fabricated node,
# and the two are INDISTINGUISHABLE from surface text alone. Loosening the
# "untyped cited node" rule to admit the plain-language supporter would necessarily
# admit fabricated citations too, breaking the integrity contract ("empty honest
# fields beat populated false ones"). So the behavior is left unchanged; these
# tests pin the correct accept/reject outcomes.

REDUNDANCY_TAGGED_BELIEF = (
    "Sequence redundancy inflates apparent within-family predictive performance."
)

# A node that genuinely supports the belief AND happens to contain the subject
# vocabulary (redundancy / clustered split) → shares a real tag → accepted.
REAL_SUPPORT_TAGGED_NODE = {
    "id": "n-real-support-tagged",
    "text": (
        "Ridge regression on a single evolutionary family collapses to near-zero R2 once "
        "nearest-neighbor sequence identity is capped below 50 percent under clustered split."
    ),
    "verdict": "confirmed",
}

# A node that genuinely supports the belief but is written in PLAIN language with
# none of the trigger vocabulary → no tags → rejected as untyped-for-claim. This
# is the conservative default: from surface text alone it cannot be told apart
# from a fabricated node, so binding must not admit it.
REAL_SUPPORT_PLAIN_NODE = {
    "id": "n-real-support-plain",
    "text": "We grouped homologous proteins, held one group out entirely, and the model did no better than chance.",
    "verdict": "confirmed",
}


def test_cleared_tagged_supporter_is_accepted() -> None:
    result = binding_check_statement_to_node(REDUNDANCY_TAGGED_BELIEF, REAL_SUPPORT_TAGGED_NODE)
    assert result.match
    assert "sequence_redundancy" in result.reason


def test_cleared_plain_supporter_rejected_same_as_fabricated() -> None:
    """A plain-language supporter and a fabricated node are rejected identically.

    This is the crux of the CLEARED verdict: both are 'untyped for the claim', so
    admitting the honest one would admit the fabricated one. Behavior is correct.
    """
    plain = binding_check_statement_to_node(REDUNDANCY_TAGGED_BELIEF, REAL_SUPPORT_PLAIN_NODE)
    fabricated = binding_check_statement_to_node(REDUNDANCY_TAGGED_BELIEF, FABRICATED_UNRELATED_NODE)
    assert not plain.match
    assert not fabricated.match
    assert plain.reason == fabricated.reason == "cited_node_untyped_for_citing_claim"


def test_ungrounded_belief_goes_to_proposed_list() -> None:
    """Beliefs with zero supporting nodes cannot enter active_beliefs."""
    state = CampaignBeliefState()
    metrics = BindingMetrics()
    state.apply_synthesis_beliefs(
        [{"statement": "Phase transition between n=20000 and n=50000", "confidence": "weak", "status": "active"}],
        tree_nodes={LOFO_NODE["id"]: LOFO_NODE},
        metrics=metrics,
    )
    assert len(state.active_beliefs) == 0
    assert len(state.proposed_ungrounded_beliefs) == 1
    assert metrics.ungrounded_belief_count == 1


# --- A4: domain-general binding — genuine cross-domain supporters bind, ----------
# --- fabricated/irrelevant ones are still rejected, using NONE of the ------------
# --- hardcoded biology tag vocab (LOFO / within-family / feature tokens). --------

ECON_BELIEF = (
    "Binding minimum wage floors reduce teenage employment in low-income urban labor "
    "markets; the disemployment effect strengthens as the wage floor becomes binding."
)

# Genuine economics supporter. Its structured claim_scope + finding are about the
# SAME subject as the belief, yet it matches none of the biology tag regexes, so
# under the old surface-tag-only logic it produced empty test_targets and was
# rejected as ``cited_node_untyped_for_citing_claim`` / ``both_subjects_untyped``.
ECON_SUPPORT_NODE = {
    "id": "econ-support-1",
    "text": (
        "Does a binding minimum wage floor reduce teenage employment in urban labor markets? "
        "Population: 240 metropolitan labor markets. OOD test: hold out one census region."
    ),
    "verdict": "confirmed",
    "claim_scope": {
        "population": "240 metropolitan low-income labor markets",
        "distribution": "urban teenage employment records",
        "claimed_generalization": "binding minimum wage floors reduce teenage employment",
        "ood_test": "hold out one census region",
    },
    "finding": {
        "claim": "Binding minimum wage floors reduced teenage employment by 3.1% in low-income urban markets.",
        "metric_name": "employment_elasticity",
    },
    "mechanism_id": "minimum_wage_disemployment",
}

# Fabricated / unrelated node cited for the SAME econ belief — an unrelated physics
# claim. It shares NO subject-specific term with the belief, only generic scope
# scaffolding ("hold out one ...", "Population:"), which is stripped before
# comparison, so it must stay rejected.
FABRICATED_PHYSICS_NODE = {
    "id": "phys-fab-1",
    "text": (
        "Does increasing lattice temperature raise superconductor critical current? "
        "Population: 40 niobium samples. OOD test: hold out one fabrication batch."
    ),
    "verdict": "confirmed",
    "claim_scope": {
        "population": "40 niobium thin-film samples",
        "distribution": "cryogenic transport measurements",
        "claimed_generalization": "lattice temperature raises critical current",
        "ood_test": "hold out one fabrication batch",
    },
    "finding": {
        "claim": "Critical current fell 12% per kelvin of lattice temperature increase.",
        "metric_name": "critical_current",
    },
    "mechanism_id": "cooper_pair_breaking",
}


def test_cross_domain_supporter_binds_via_structured_overlap() -> None:
    """(a) A genuine economics supporter binds through structured overlap,
    despite using none of the biology tag vocab (both sides have empty tags)."""
    citing = extract_subject_from_statement(ECON_BELIEF)
    assert not citing.test_targets, "econ belief must have no biology tags"
    result = binding_check_statement_to_node(ECON_BELIEF, ECON_SUPPORT_NODE)
    assert result.match, result.reason
    # Bound on shared claim/scope subject terms, not on any biology tag path.
    assert result.reason.startswith(("shared_claim_terms", "shared_scope_subject",
                                     "shared_mechanism_or_feature"))


def test_cross_domain_fabricated_node_still_rejected() -> None:
    """(b) A fabricated physics node cited for the econ belief shares no
    subject-specific term and must be rejected — integrity line held."""
    result = binding_check_statement_to_node(ECON_BELIEF, FABRICATED_PHYSICS_NODE)
    assert not result.match
    assert result.reason in ("no_subject_overlap", "both_subjects_untyped")


def test_cross_domain_filter_keeps_supporter_strips_fabrication() -> None:
    """(a)+(b) together on the real filter path: the genuine supporter survives,
    the fabricated node is stripped, no biology vocab involved."""
    nodes = {
        ECON_SUPPORT_NODE["id"]: ECON_SUPPORT_NODE,
        FABRICATED_PHYSICS_NODE["id"]: FABRICATED_PHYSICS_NODE,
    }
    metrics = BindingMetrics()
    accepted = filter_node_citations(
        ECON_BELIEF,
        [ECON_SUPPORT_NODE["id"], FABRICATED_PHYSICS_NODE["id"]],
        nodes,
        metrics=metrics,
    )
    assert ECON_SUPPORT_NODE["id"] in accepted
    assert FABRICATED_PHYSICS_NODE["id"] not in accepted
    assert metrics.binding_accepted_count >= 1
    assert metrics.binding_rejected_count >= 1


def test_single_shared_word_is_incidental_not_a_bind() -> None:
    """A node sharing only ONE content word with the belief (and no scope /
    mechanism / feature overlap) is incidental and must NOT bind — this is what
    keeps loosely-related citations out."""
    belief = "Rising interest rates cool housing demand in coastal metros."
    # Shares only the single content word 'housing' — nothing else in common.
    incidental_node = {
        "id": "incidental-1",
        "text": (
            "Does solar panel efficiency degrade with rooftop housing age? "
            "Population: 500 rooftop installations. OOD test: hold out one manufacturer."
        ),
        "verdict": "confirmed",
        "claim_scope": {
            "population": "500 rooftop solar installations",
            "claimed_generalization": "panel efficiency degrades with age",
            "ood_test": "hold out one manufacturer",
        },
        "finding": {"claim": "Rooftop housing panels lost 0.5% efficiency per year."},
    }
    result = binding_check_statement_to_node(belief, incidental_node)
    assert not result.match, result.reason


def test_shared_mechanism_id_binds_across_domain() -> None:
    """A node sharing an explicit mechanism_id with a belief that names that
    mechanism binds — an unambiguous, domain-general subject anchor."""
    belief = "The credit-channel mechanism amplifies monetary tightening on small firms."
    node = {
        "id": "mech-1",
        "text": "Bank lending contracts sharply after rate hikes for small firms.",
        "verdict": "confirmed",
        "claim_scope": {
            "population": "small firms",
            "claimed_generalization": "credit channel amplifies tightening",
            "ood_test": "hold out one lending region",
        },
        "mechanism_id": "credit-channel",
        "feature_subset": ["credit-channel"],
    }
    result = binding_check_statement_to_node(belief, node)
    assert result.match, result.reason
    # 'credit-channel' appears in both statement salient terms and node ids, so it
    # binds either on shared_mechanism_or_feature or shared_claim_terms.
    assert "credit" in result.reason or result.reason.startswith("shared")


def test_biology_tagged_supporter_still_binds() -> None:
    """(c) Existing biology behavior preserved: the within-family redundancy
    belief still binds to the within-family node via the tag path."""
    result = binding_check_statement_to_node(REDUNDANCY_BELIEF, WITHIN_FAMILY_NODE)
    assert result.match


def test_biology_insufficiency_vs_lofo_still_rejected() -> None:
    """(c) Existing biology behavior preserved: the unfalsifiable feature-
    insufficiency belief still rejects the cross-family LOFO node (the structured
    path must not override the incompatible-tag guard)."""
    result = binding_check_statement_to_node(INSUFFICIENCY_BELIEF, LOFO_NODE)
    assert not result.match
    assert "lofo" in result.reason.lower() or "insufficient" in result.reason.lower() or "overlap" in result.reason.lower()
