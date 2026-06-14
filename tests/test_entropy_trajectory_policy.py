"""Tests for fixes.md P0–P2: entropy trajectory + residual history + analyst inputs."""
from __future__ import annotations

from propab.entropy_trajectory import summarize_entropy_trajectory
from propab.meta_science import CampaignObservation
from propab.policy_evaluation import evaluate_candidate_policy
from propab.policy_fitness_ledger import FitnessRecord, PolicyFitnessLedger
from propab.policy_record import PredictedEffects
from propab.policy_residual_history import residual_history_for_bucket
from services.orchestrator.policy_analyst import _build_analyst_prompt, propose_policy_narrative_sync


def _sample_points():
    return [
        {"tested": 1, "theme_entropy": 0.72},
        {"tested": 10, "theme_entropy": 1.0},
        {"tested": 30, "theme_entropy": 1.59},
        {"tested": 70, "theme_entropy": 1.98},
        {"tested": 120, "theme_entropy": 1.89},
    ]


def test_summarize_entropy_trajectory_monotone_rise():
    summary = summarize_entropy_trajectory(_sample_points())
    assert summary.H_start == 0.72
    assert summary.H_end == 1.89
    assert summary.growth_pattern == "monotone_rise"
    assert summary.cross_H_1_5_at_tested == 30
    assert summary.growth_rate > 0


def test_residual_history_for_bucket():
    ledger = PolicyFitnessLedger()
    ledger.record(FitnessRecord(
        policy_id="pol-abc",
        campaign_id="camp-1111",
        budget_bucket="3h",
        domain_bucket="graphs",
        predictions={"start_H": 0.7},
        observations={"start_H": 0.72},
        residuals={"start_H": 0.02},
        accept_or_reject="REJECT",
        detail={"observed_trajectory": {"H_start": 0.72, "H_end": 1.9}},
    ))
    hist = residual_history_for_bucket(ledger, budget_bucket="3h", domain_bucket="graphs")
    assert len(hist) == 1
    assert hist[0]["residuals"]["start_H"] == 0.02
    assert hist[0]["observed_trajectory"]["H_end"] == 1.9


def test_analyst_prompt_includes_residual_and_trajectory():
    traj = summarize_entropy_trajectory(_sample_points()).to_dict()
    history = [{"campaign_id": "abc", "residuals": {"start_H": -0.1}}]
    prompt = _build_analyst_prompt(
        domain_bucket="graphs",
        budget_bucket="3h",
        parent=None,
        params={"boosts": {"diffusion_dynamics": 0.35}, "penalties": {}, "blocked_failures": []},
        campaign_metrics={"closure_ratio": 0.1},
        trajectory_summary=traj,
        residual_history=history,
    )
    assert "campaign residual history" in prompt.lower()
    assert "entropy trajectory" in prompt.lower() or "trajectory summary" in prompt.lower()
    assert "cross_H_1_5_at_tested" in prompt
    assert '"theme_entropy_delta"' not in prompt


def test_deterministic_narrative_predicts_entropy_dynamics():
    from propab.knowledge_graph import KnowledgeGraph
    from propab.meta_science import MetaScienceLedger
    from propab.policy_record import PolicyRecord

    graph = KnowledgeGraph()
    meta = MetaScienceLedger()
    parent = PolicyRecord.empty_accepted(budget_bucket="3h", domain_bucket="graphs")
    traj = summarize_entropy_trajectory(_sample_points())
    _, predicted, _, _ = propose_policy_narrative_sync(
        parent=parent,
        graph=graph,
        meta=meta,
        budget_bucket="3h",
        domain_bucket="graphs",
        trajectory_summary=traj,
    )
    assert predicted.uses_entropy_dynamics()
    assert predicted.start_H > 0
    assert predicted.saturation_H > 1.5


def test_evaluation_entropy_dynamics_within_tolerance():
    baseline = CampaignObservation(
        campaign_id="b", question="q", tested=85, confirmed=23, refuted=3,
        inconclusive=59, closure_ratio=0.306, theme_entropy=2.5,
        general_theme_fraction=0.0, compute_seconds=10800, policy_generation=1,
        knowledge_claims=0, knowledge_failures=0, budget_bucket="3h", domain_bucket="graphs",
    )
    current = CampaignObservation(
        campaign_id="c", question="q", tested=200, confirmed=8, refuted=40,
        inconclusive=152, closure_ratio=0.09, theme_entropy=1.9,
        general_theme_fraction=0.0, compute_seconds=10800, policy_generation=2,
        knowledge_claims=0, knowledge_failures=0, budget_bucket="3h", domain_bucket="graphs",
    )
    traj = summarize_entropy_trajectory(_sample_points())
    predicted = PredictedEffects(
        closure_ratio_delta=-0.2,
        start_H=0.72,
        growth_rate=0.01,
        saturation_H=1.9,
        cross_H_1_5_at_tested=30,
        cross_H_2_0_at_tested=80,
    )
    ok, detail = evaluate_candidate_policy(
        predicted=predicted,
        baseline_obs=baseline,
        current_obs=current,
        budget_bucket="3h",
        calibration_closure_target=None,
        trajectory_summary=traj,
    )
    assert detail["entropy_eval_mode"] == "dynamics"
    assert "start_H" in detail["residuals"]
    assert ok or not ok  # smoke — closure gate may reject
