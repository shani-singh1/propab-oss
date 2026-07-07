"""Tests for fixes.md campaign redesign (belief state + synthesis + prompts)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from propab.belief_state import (
    BeliefObject,
    CampaignBeliefState,
    ClosedBelief,
    EXHAUSTION_ROUNDS_REQUIRED,
    MAX_ACTIVE_BELIEFS,
)
from propab.campaign import (
    ResearchCampaign,
    STOP_REASON_HYPOTHESIS_CAP_REACHED,
    STOP_REASON_TIME_BUDGET_EXHAUSTED,
)
from propab.campaign_synthesis import (
    apply_synthesis_to_frontier,
    parse_synthesis_response,
    should_trigger_synthesis,
    text_similarity,
)
from propab.hypothesis_tree import HypothesisTree
from propab.prompt_composer import compose_synthesis_prompt, compress_node_history, load_prompt


ROOT = Path(__file__).resolve().parents[1]


def test_prompts_directory_exists() -> None:
    role = load_prompt("orchestrator_role.md")
    task = load_prompt("synthesis_task.md")
    assert "Campaign Orchestrator" in role
    assert "BeliefObjects" in role or "belief" in role.lower()
    assert "critical_experiment" in task
    assert "parent_id" in task


def test_belief_cap_and_abandonment() -> None:
    state = CampaignBeliefState()
    state.apply_synthesis_beliefs([
        {"statement": "A", "confidence": "weak", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "B", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "C", "confidence": "strong", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "D", "confidence": "weak", "status": "active", "supporting_nodes": ["n1"]},
    ])
    assert len(state.active_beliefs) == MAX_ACTIVE_BELIEFS
    assert state.active_beliefs[0].statement == "A"

    state.abandon_belief(state.active_beliefs[0], "LOFO gap too large")
    assert len(state.active_beliefs) == 2
    assert len(state.closed_beliefs) == 1
    assert "LOFO" in state.closed_beliefs[0].reason


def test_exhaustion_requires_three_rounds() -> None:
    state = CampaignBeliefState()
    state.apply_synthesis_beliefs([
        {"statement": "only unclear", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
    ])
    for _ in range(EXHAUSTION_ROUNDS_REQUIRED - 1):
        state.record_exhaustion_round(True)
        assert not state.branch_exhausted
    state.record_exhaustion_round(True)
    assert state.branch_exhausted


def test_exhaustion_resets_on_new_strong_belief() -> None:
    state = CampaignBeliefState()
    state.exhaustion_rounds = 2
    state.apply_synthesis_beliefs([
        {"statement": "revived", "confidence": "strong", "status": "strengthened", "supporting_nodes": ["n1"]},
    ])
    state.record_exhaustion_round(False)
    assert state.exhaustion_rounds == 0


def test_parse_synthesis_response_json() -> None:
    raw = json.dumps({
        "beliefs": [{"statement": "family proxy", "confidence": "weak", "status": "active"}],
        "frontier_candidates": [{
            "id": "lofo_null",
            "text": (
                "Label-shuffle LOFO null test.\n"
                "Population: 7 RT families\nDistribution: mandrake\n"
                "Claimed generalization: none\nExpected failure modes: leakage\n"
                "OOD test: held-out family"
            ),
            "test_methodology": "mandrake_verification",
            "expansion_type": "diagnostic",
        }],
        "recent_activity_summary": "Testing family leakage",
        "direction_exhausted": False,
    })
    parsed = parse_synthesis_response(raw)
    assert parsed["beliefs"][0]["statement"] == "family proxy"
    assert len(parsed["frontier_candidates"]) == 1


def test_apply_synthesis_adds_frontier_candidates() -> None:
    tree = HypothesisTree()
    parent = tree.add_seeds(
        [{
            "text": (
                "Thermal features predict RT activity.\n"
                "Population: RT enzymes\nDistribution: mandrake 7 families\n"
                "Claimed generalization: cross-family signal\n"
                "Expected failure modes: family leakage\nOOD test: leave-one-family-out"
            ),
            "test_methodology": "mandrake_verification",
        }],
        generation=0,
    )[0]
    tree.update_node(parent.id, "confirmed", 0.9, 'evidence={"verdict_reason": "lofo positive"};')
    state = CampaignBeliefState()
    parsed = {
        "beliefs": [
            {
                "statement": "Thermal features predict RT activity under cross-family LOFO.",
                "confidence": "strong",
                "status": "strengthened",
                "supporting_nodes": [parent.id],
            }
        ],
        "frontier_candidates": [{
            "id": "diag1",
            "parent_id": parent.id,
            "text": (
                "Finer family-split LOFO degradation test.\n"
                "Population: RT enzymes\nDistribution: mandrake 7 families\n"
                "Claimed generalization: cross-family only\n"
                "Expected failure modes: family leakage\nOOD test: leave-one-family-out"
            ),
            "test_methodology": "mandrake_verification",
            "expansion_type": "diagnostic",
        }],
        "recent_activity_summary": "Discriminating biology vs proxy",
        "direction_exhausted": False,
    }
    added, metrics = apply_synthesis_to_frontier(
        tree,
        state,
        parsed,
        question="Which biophysical properties predict RT activity independently of family?",
        generation=1,
        relevance_threshold=0.0,
    )
    assert metrics["n_added"] >= 1
    assert metrics["n_added_as_children"] == 1
    assert metrics.get("n_added_as_roots", 0) == 0
    assert added[0].parent_id == parent.id
    assert added[0].depth == 1
    assert added[0].id in tree.nodes[parent.id].children
    assert len(state.active_beliefs) == 1
    assert state.active_beliefs[0].confidence == "strong"
    assert tree.next_dispatch_candidate(frozenset()) is not None
    # Lineage-derivation quality (§3.2): the candidate named an explicit parent.
    assert metrics["n_lineage_explicit"] == 1
    assert metrics["n_lineage_inferred"] == 0
    assert metrics["lineage_derivation_rate"] == 1.0
    # Convergence metric (§3.3): one confirmed node (the parent) → depth 1.0.
    assert metrics["confirmed_lineage_depth"] == 1.0


def test_apply_synthesis_can_bootstrap_root_when_no_parent_exists() -> None:
    tree = HypothesisTree()
    state = CampaignBeliefState()
    parsed = {
        "beliefs": [],
        "frontier_candidates": [{
            "id": "root1",
            "text": (
                "Initial scoped diagnostic.\n"
                "Population: RT enzymes\nDistribution: mandrake 7 families\n"
                "Claimed generalization: cross-family signal\n"
                "Expected failure modes: family leakage\nOOD test: leave-one-family-out"
            ),
            "test_methodology": "mandrake_verification",
            "expansion_type": "diagnostic",
        }],
        "direction_exhausted": False,
    }
    added, metrics = apply_synthesis_to_frontier(
        tree,
        state,
        parsed,
        question="Which biophysical properties predict RT activity independently of family?",
        generation=1,
        relevance_threshold=0.0,
    )
    assert len(added) == 1
    assert added[0].parent_id is None
    assert added[0].depth == 0
    assert metrics["n_added_as_roots"] == 1


def test_synthesis_infers_parent_when_llm_omits_parent_id() -> None:
    tree = HypothesisTree()
    parent = tree.add_seeds(
        [{
            "text": (
                "Global hydrophobicity fails under family holdout.\n"
                "Population: RT enzymes\nDistribution: mandrake 7 families\n"
                "Claimed generalization: cross-family signal\n"
                "Expected failure modes: motif-specific effects\nOOD test: leave-one-family-out"
            ),
            "test_methodology": "mandrake_verification",
        }],
        generation=0,
    )[0]
    tree.update_node(parent.id, "refuted", 0.8, 'evidence={"verdict_reason": "lofo negative"};')

    parsed = {
        "beliefs": [],
        "frontier_candidates": [{
            "id": "alt1",
            "text": (
                "Family-local motif features replace global hydrophobicity under holdout.\n"
                "Population: RT enzymes\nDistribution: mandrake 7 families\n"
                "Claimed generalization: cross-family signal\n"
                "Expected failure modes: global averages remain sufficient\nOOD test: leave-one-family-out"
            ),
            "test_methodology": "mandrake_verification",
            "expansion_type": "alternative",
        }],
        "direction_exhausted": False,
    }
    added, metrics = apply_synthesis_to_frontier(
        tree,
        CampaignBeliefState(),
        parsed,
        question="Which biophysical properties predict RT activity independently of family?",
        generation=1,
        relevance_threshold=0.0,
    )
    assert len(added) == 1
    assert metrics["n_added_as_children"] == 1
    assert added[0].parent_id == parent.id
    assert tree.nodes[parent.id].children == [added[0].id]
    # Lineage was INFERRED (LLM omitted parent_id) — the metric must show it, so a
    # campaign whose lineage is mostly inferred (structural depth without real
    # derivation) is visible rather than silent (§3.2).
    assert metrics["n_lineage_inferred"] == 1
    assert metrics["n_lineage_explicit"] == 0
    assert metrics["lineage_derivation_rate"] == 0.0


def test_compose_synthesis_includes_pinned_context() -> None:
    tree = HypothesisTree()
    node = tree.add_seeds([{"text": "Seed claim", "test_methodology": "x"}], generation=0)[0]
    tree.update_node(node.id, "refuted", 0.8, 'evidence={"verdict_reason": "lofo negative"};')
    node.inconclusive_reason = "replication_failed"
    node.failure_signature = "lofo_gap"

    state = CampaignBeliefState()
    state.human_messages.append("Focus on family leakage tests")
    state.active_beliefs.append(BeliefObject(statement="family proxy", confidence="weak"))
    state.closed_beliefs.append(ClosedBelief(statement="thermal universal", reason="abandoned"))

    prompt = compose_synthesis_prompt(
        question="Test question verbatim?",
        belief_state=state,
        tree=tree,
    )
    assert "Test question verbatim?" in prompt
    assert "Focus on family leakage tests" in prompt
    assert "family proxy" in prompt
    assert "thermal universal" in prompt
    assert "replication_failed" in prompt
    assert "verdict_reason: lofo negative" in prompt
    assert "Open expansion targets" in prompt
    assert f"target_id={node.id}" in prompt
    assert "choose parent_id" in prompt


def test_compress_drops_old_prose_keeps_structured_fields() -> None:
    tree = HypothesisTree()
    for i in range(5):
        n = tree.add_seeds([{"text": f"Hypothesis number {i} with long prose " * 5, "test_methodology": "x"}], generation=i)[0]
        tree.update_node(n.id, "refuted", 0.5, f'evidence={{"verdict_reason": "r{i}"}};')
        n.failure_signature = f"sig{i}"
    blob = compress_node_history(tree, max_prose_nodes=2)
    assert "sig4" in blob
    assert "verdict_reason: r4" in blob


@pytest.mark.parametrize(
    ("results", "queued", "max_c", "expected"),
    [
        (3, 5, 3, True),   # batch threshold met
        (0, 0, 3, False),  # no results yet
        (1, 1, 3, False),  # queue not fully dry — no premature synthesis
        (2, 0, 3, True),   # queue empty + partial batch (>= batch//2)
    ],
)
def test_should_trigger_synthesis(results: int, queued: int, max_c: int, expected: bool) -> None:
    state = CampaignBeliefState()
    state.results_since_last_synthesis = results
    assert should_trigger_synthesis(state, results_since=results, max_concurrent=max_c, queued_candidates=queued) is expected


def test_campaign_belief_state_roundtrip() -> None:
    state = CampaignBeliefState(
        active_beliefs=[BeliefObject(statement="x", confidence="weak")],
        closed_beliefs=[ClosedBelief(statement="y", reason="z")],
        human_messages=["guidance"],
        recent_activity_summary="exploring",
        results_since_last_synthesis=2,
    )
    restored = CampaignBeliefState.from_dict(state.to_dict())
    assert restored.active_beliefs[0].statement == "x"
    assert restored.closed_beliefs[0].reason == "z"
    assert restored.human_messages == ["guidance"]


def test_frontier_dedup_rejects_similar_candidates() -> None:
    tree = HypothesisTree()
    existing_text = (
        "LOFO label-shuffle null test on RT enzyme families.\n"
        "Population: RT families\nDistribution: mandrake\n"
        "Claimed generalization: none\nExpected failure modes: leakage\n"
        "OOD test: held-out family"
    )
    existing = tree.add_seeds(
        [{"text": existing_text, "test_methodology": "mandrake_verification"}],
        generation=1,
    )[0]
    tree.update_node(existing.id, "refuted", 0.9, "evidence=lofo negative;")

    state = CampaignBeliefState()
    near_dup = existing_text.replace("label-shuffle", "label shuffle")
    parsed = {
        "beliefs": [],
        "frontier_candidates": [
            {"id": "dup", "text": near_dup, "test_methodology": "mandrake_verification"},
            {"id": "novel", "text": (
                "Plate row/column artifact diagnostic.\n"
                "Population: all wells\nDistribution: mandrake\n"
                "Claimed generalization: none\nExpected failure modes: batch effect\n"
                "OOD test: held-out plate"
            ), "test_methodology": "mandrake_verification"},
        ],
        "direction_exhausted": False,
    }
    added, metrics = apply_synthesis_to_frontier(
        tree, state, parsed, question="RT activity question?", generation=2, relevance_threshold=0.0,
    )
    assert metrics["n_rejected_duplicate"] >= 1
    assert len(added) == 1
    assert text_similarity(near_dup, existing.text) >= 0.85


def test_stop_reason_and_should_stop_use_wall_clock() -> None:
    c = ResearchCampaign(
        id="test",
        question="q?",
        compute_budget_seconds=3600,
        started_at="2020-01-01T00:00:00+00:00",
    )
    assert c.should_stop()
    c.finalize_stop(STOP_REASON_TIME_BUDGET_EXHAUSTED)
    assert c.stop_reason == STOP_REASON_TIME_BUDGET_EXHAUSTED
    assert c.status == "budget_exhausted"

    c2 = ResearchCampaign(id="t2", question="q?", max_hypotheses_cap=5)
    c2.total_hypotheses = 5
    assert c2.hypothesis_cap_reached()
    c2.finalize_stop(STOP_REASON_HYPOTHESIS_CAP_REACHED)
    assert c2.stop_reason == STOP_REASON_HYPOTHESIS_CAP_REACHED


def test_campaign_meta_roundtrip_in_breakthrough_blob() -> None:
    from propab.campaign_db import (
        _apply_campaign_meta_from_db,
        _breakthrough_criteria_for_db,
    )

    c = ResearchCampaign(
        id="meta-test",
        question="q?",
        max_hypotheses_cap=50,
        seed_source="anomaly",
    )
    c.belief_state.human_messages.append("focus LOFO")
    blob = _breakthrough_criteria_for_db(c)
    assert blob["_campaign_meta"]["max_hypotheses_cap"] == 50
    data: dict = {"breakthrough_criteria": {k: v for k, v in blob.items() if k != "_campaign_meta"}}
    _apply_campaign_meta_from_db(data, dict(blob))
    assert data["max_hypotheses_cap"] == 50
    assert data["belief_state"]["human_messages"] == ["focus LOFO"]


def test_belief_backfill_from_synthesis_events() -> None:
    from propab.campaign_resume import backfill_belief_state_if_empty

    events = [
        {
            "step": "campaign.synthesis",
            "payload": {
                "active_beliefs": [
                    {"statement": "family proxy", "confidence": "weak", "status": "active"},
                    {"statement": "noise floor", "confidence": "unclear", "status": "active"},
                ],
                "critical_experiment": {"title": "LOFO shuffle test"},
            },
        },
    ]
    state, changed = backfill_belief_state_if_empty(
        CampaignBeliefState(),
        events=events,
        tree_nodes={"n1": {"verdict": "refuted"}},
    )
    assert changed
    assert len(state.active_beliefs) == 2
    assert state.active_beliefs[0].statement == "family proxy"
    assert "n1" in state.last_synthesis_node_ids


def test_contrarian_belief_reset_closes_prior_and_seeds_two_rivals() -> None:
    from propab.campaign_resume import (
        CONTRARIAN_BELIEF_FAMILY_SPECIFIC,
        CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT,
        apply_contrarian_belief_reset,
    )

    state = CampaignBeliefState()
    state.apply_synthesis_beliefs([
        {"statement": "old belief A", "confidence": "strong", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "old belief B", "confidence": "weak", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "old belief C", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
    ])
    apply_contrarian_belief_reset(state)
    assert len(state.active_beliefs) == 2
    assert state.active_beliefs[0].statement == CONTRARIAN_BELIEF_FAMILY_SPECIFIC
    assert state.active_beliefs[1].statement == CONTRARIAN_BELIEF_REDUNDANCY_ARTIFACT
    assert state.rival_exhaustion_mode is True
    assert state.branch_exhausted is False
    assert len(state.closed_beliefs) == 3
    assert state.human_messages


def test_rival_exhaustion_requires_both_beliefs_three_rounds() -> None:
    state = CampaignBeliefState()
    state.rival_exhaustion_mode = True
    state.apply_synthesis_beliefs([
        {"statement": "rival A", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "rival B", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
    ])
    for _ in range(EXHAUSTION_ROUNDS_REQUIRED - 1):
        state.record_synthesis_exhaustion(True)
        assert not state.branch_exhausted
    state.record_synthesis_exhaustion(True)
    assert state.branch_exhausted


def test_rival_exhaustion_does_not_stop_when_one_belief_progresses() -> None:
    state = CampaignBeliefState()
    state.rival_exhaustion_mode = True
    state.apply_synthesis_beliefs([
        {"statement": "rival A", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
        {"statement": "rival B", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
    ])
    for _ in range(EXHAUSTION_ROUNDS_REQUIRED):
        state.record_synthesis_exhaustion(True)
    assert state.branch_exhausted

    state = CampaignBeliefState()
    state.rival_exhaustion_mode = True
    state.apply_synthesis_beliefs([
        {"statement": "rival A", "confidence": "strong", "status": "strengthened", "supporting_nodes": ["n1"]},
        {"statement": "rival B", "confidence": "unclear", "status": "active", "supporting_nodes": ["n1"]},
    ])
    for _ in range(EXHAUSTION_ROUNDS_REQUIRED):
        state.record_synthesis_exhaustion(True)
    assert not state.branch_exhausted
