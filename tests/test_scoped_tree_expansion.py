"""Scoped tree expansion — fixes.md P0-P3."""
from __future__ import annotations

import json

from propab.hypothesis_tree import HypothesisTree
from propab.scoped_claim import (
    EXPANSION_MISSING_SCOPE,
    EXPANSION_VALID_INHERITANCE,
    EXPANSION_VALID_MUTATION,
    ScopedClaim,
    compute_scope_delta,
    validate_expansion_child,
)


def _parent_scope() -> ScopedClaim:
    return ScopedClaim(
        text="parent",
        population="BA graphs N=500-2000",
        distribution="i.i.d. BA with m=3",
        claimed_generalization="WS with matched degree",
        expected_failure_modes="dense ER graphs",
        ood_test="hold out WS family",
    )


def test_compute_scope_delta_detects_ood_change() -> None:
    parent = _parent_scope()
    child = ScopedClaim(
        text="child",
        population=parent.population,
        distribution="BA + WS training mix",
        claimed_generalization=parent.claimed_generalization,
        expected_failure_modes=parent.expected_failure_modes,
        ood_test="hold out ER family",
    )
    delta = compute_scope_delta(parent, child)
    assert delta is not None
    assert "distribution changed" in delta
    assert "OOD target changed" in delta


def test_expansion_gate_rejects_missing_scope() -> None:
    parent = _parent_scope()
    ok, _, reason = validate_expansion_child(
        {"text": "Spectral radius predicts SIS prevalence.", "test_methodology": "stats"},
        parent=parent,
        question="network contagion spreading",
    )
    assert not ok
    assert reason == "missing_scope"


def test_expansion_gate_accepts_full_scope() -> None:
    parent = _parent_scope()
    ok, enriched, reason = validate_expansion_child(
        {
            "text": "k-shell index beats degree for LT thresholds.",
            "population": parent.population,
            "distribution": parent.distribution,
            "claimed_generalization": parent.claimed_generalization,
            "expected_failure_modes": parent.expected_failure_modes,
            "ood_test": parent.ood_test,
            "test_methodology": "statistical_significance",
        },
        parent=parent,
        question="network contagion spreading",
    )
    assert ok
    assert reason is None
    assert enriched.get("claim_scope")
    assert enriched.get("scope_delta") is None


def test_parse_expanded_nodes_applies_gate() -> None:
    tree = HypothesisTree()
    parent = tree.add_seeds(
        [{
            "text": (
                "Parent claim.\n"
                "Population: BA graphs N=500-2000\n"
                "Distribution: i.i.d. BA m=3\n"
                "Claimed generalization: WS matched degree\n"
                "Expected failure modes: ER dense\n"
                "OOD test: hold out WS"
            ),
            "test_methodology": "baseline",
        }],
        generation=0,
    )[0]
    tree.update_node(parent.id, "confirmed", 0.9, "evidence")

    llm_json = json.dumps([
        {
            "id": "c1",
            "text": "Child without scope fields",
            "test_methodology": "stats",
            "expansion_type": "boundary",
        },
        {
            "id": "c2",
            "text": "Child with full scope",
            "population": "BA graphs N=500-2000",
            "distribution": "i.i.d. BA m=3",
            "claimed_generalization": "WS matched degree",
            "expected_failure_modes": "ER dense",
            "ood_test": "hold out ER instead of WS",
            "test_methodology": "statistical_significance",
            "expansion_type": "generalization",
        },
    ])
    children, metrics = tree.parse_expanded_nodes(
        parent.id, llm_json, generation=1, question="contagion networks",
    )
    assert metrics["n_children_generated"] == 2
    assert metrics["n_children_rejected"] == 1
    assert metrics["n_children_passed"] == 1
    assert len(children) == 1
    assert children[0].claim_scope is not None
    assert children[0].scope_delta is not None


def test_build_expand_prompt_includes_structured_failure_fields() -> None:
    tree = HypothesisTree()
    node = tree.add_seeds([{
        "text": (
            "Thermal t55 predicts RT.\nPopulation: 7 families\nDistribution: mandrake\n"
            "Claimed generalization: cross-family\nExpected failure modes: leakage\nOOD test: LOFO"
        ),
        "test_methodology": "mandrake_verification",
    }], generation=0)[0]
    tree.update_node(
        node.id,
        "inconclusive",
        0.5,
        'evidence={"metric_value": -0.2, "verdict_reason": "lofo gap large", '
        '"artifact_gate": {"verdict": "inconclusive", "ranked_artifacts": '
        '[{"artifact_id": "family_leakage"}]}}; steps=1.',
    )
    node.inconclusive_reason = "verification_failure"
    node.failure_signature = "lofo_gap_large"
    prompt = tree.build_expand_prompt(node.id)
    assert prompt is not None
    assert "verdict_reason: lofo gap large" in prompt
    assert "inconclusive_reason: verification_failure" in prompt
    assert "family_leakage" in prompt


def test_build_expand_prompt_includes_parent_scope() -> None:
    tree = HypothesisTree()
    node = tree.add_seeds([{
        "text": (
            "Seed.\nPopulation: BA N=500\nDistribution: BA m=3\n"
            "Claimed generalization: WS\nExpected failure modes: ER\nOOD test: WS holdout"
        ),
        "test_methodology": "x",
    }], generation=0)[0]
    tree.update_node(node.id, "refuted", 0.5)
    prompt = tree.build_expand_prompt(node.id)
    assert prompt is not None
    assert "ParentScope" in prompt
    assert "Population: BA N=500" in prompt
    assert "population, distribution, claimed_generalization" in prompt
